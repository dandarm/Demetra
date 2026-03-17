#!/usr/bin/env python3
"""First-pass + tracking inference from a folder of Mediterranean frames.

Pipeline:
1. Build stretched SxS frames for first-pass inference.
2. Run first-pass on full Mediterranean frames.
3. Build 16-frame high-resolution tile folders centered on first-pass estimates.
4. Run VideoMAE tracking on positive tile folders.
5. Export final per-timeframe CSV: datetime, has_cyclone, pred_lat, pred_lon.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
VIDEOMAE_ROOT = REPO_ROOT / "moduli" / "videomae"
FIRSTPASS_ROOT_DEFAULT = REPO_ROOT / "moduli" / "firstpass"
FIRSTPASS_MODEL_DEFAULT = REPO_ROOT / "trained_models" / "firstpass_model.ckpt"
TRACKING_MODEL_DEFAULT = REPO_ROOT / "trained_models" / "checkpoint-tracking-best_1.pth"
if str(VIDEOMAE_ROOT) not in sys.path:
    sys.path.insert(0, str(VIDEOMAE_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, DistributedSampler

import utils
from arguments import prepare_tracking_args
from dataset.datasets import MedicanesTrackDataset
import engine_for_tracking as tracking_engine
from ffmpeg_utils import resolve_ffmpeg_executable
from inference_tracking import run_tracking_inference, load_checkpoint, set_seeds
from models.tracking_model import create_tracking_model
from track_from_folder import (
    _parse_tile_folder_name,
    _build_gt_map_from_tracks,
    _build_tracking_csv_for_folders,
)
from utils import setup_for_distributed

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


IMG_EXTS = {".png"}
TS_RE = re.compile(r"(\d{8})_(\d{4})")
FIRSTPASS_CLIP_STRIDE_MINUTES = 15
STANDARD_TILE_STRIDE_X = 213
STANDARD_TILE_STRIDE_Y = 196
STANDARD_IMAGE_WIDTH = 1290
STANDARD_IMAGE_HEIGHT = 420


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline globale: firstpass (stretched full basin) + crop videotile + "
            "tracking ad alta risoluzione."
        )
    )
    parser.add_argument("--input_dir", required=True, help="Cartella immagini originali full-resolution.")
    parser.add_argument("--output_dir", required=True, help="Cartella output della pipeline.")
    parser.add_argument(
        "--firstpass_model_path",
        default=str(FIRSTPASS_MODEL_DEFAULT),
        help="Checkpoint first-pass.",
    )
    parser.add_argument(
        "--tracking_model_path",
        default=str(TRACKING_MODEL_DEFAULT),
        help="Checkpoint tracking VideoMAE.",
    )
    parser.add_argument(
        "--firstpass_root",
        default=None,
        help="Root modulo firstpass (default: <repo>/moduli/firstpass).",
    )
    parser.add_argument(
        "--firstpass_config",
        default=None,
        help="Config firstpass (default: <firstpass_root>/config/default.yml).",
    )
    parser.add_argument(
        "--firstpass_threshold",
        type=float,
        default=0.2,
        help="Soglia presenza first-pass per selezionare le clip da tracciare.",
    )
    parser.add_argument(
        "--firstpass_image_size",
        type=int,
        default=224,
        help="Lato SxS delle immagini stretched date al first-pass.",
    )
    parser.add_argument(
        "--firstpass_batch_size",
        type=int,
        default=40,
        help="Batch size first-pass inference.",
    )
    parser.add_argument(
        "--firstpass_num_workers",
        type=int,
        default=4,
        help="Num workers first-pass inference.",
    )
    parser.add_argument(
        "--firstpass_device",
        default=None,
        help="Device first-pass (cuda|cpu). Default: cuda se disponibile.",
    )
    parser.add_argument(
        "--firstpass_soft_argmax",
        action="store_true",
        help="Abilita soft-argmax nel first-pass.",
    )
    parser.add_argument(
        "--firstpass_soft_argmax_tau",
        type=float,
        default=None,
        help="Tau per soft-argmax first-pass (opzionale).",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
        help="Numero frame per videotile.",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=224,
        help="Dimensione lato tile crop sull'immagine originale.",
    )
    parser.add_argument(
        "--standard_tiling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Se attivo, per ogni centro first-pass usa la tile della griglia standard "
            "(stride default) che contiene il centro, invece del crop centrato."
        ),
    )
    parser.add_argument(
        "--max_contiguous_gap_minutes",
        type=float,
        default=60.0,
        help="Gap massimo per considerare contiguo un gruppo temporale.",
    )
    # Compat: kept as hidden no-op to avoid breaking old command lines.
    parser.add_argument(
        "--end_on_hour",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--manos_file",
        default="medicane_data_input/medicanes_new_windows.csv",
        help="CSV Manos opzionale per GT tracking.",
    )
    parser.add_argument(
        "--on",
        default=None,
        help="Preset macchina per arguments.py (es. leonardo).",
    )
    parser.add_argument(
        "--make_video",
        action="store_true",
        help="Se presente, genera il video Mediterraneo con tracce.",
    )
    parser.add_argument(
        "--only_video",
        action="store_true",
        help="Se presente, genera solo MP4 da frame gia presenti (richiede --make_video).",
    )
    parser.add_argument(
        "--video_name",
        default="mediterraneo_predizioni",
        help="Nome base del video MP4 (senza estensione).",
    )
    parser.add_argument(
        "--ffmpeg_path",
        default=None,
        help="Path da aggiungere al PATH per trovare ffmpeg (opzionale).",
    )
    return parser.parse_args()


def _setup_distributed():
    if not torch.cuda.is_available():
        setup_for_distributed(True)
        return 0, 0, 1, False

    rank, local_rank, world_size, _, _ = utils.get_resources()
    if world_size > 1:
        if not dist.is_initialized():
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
        distributed = True
    else:
        distributed = False
    torch.cuda.set_device(local_rank)
    setup_for_distributed(rank == 0, silence_non_master=True)
    return rank, local_rank, world_size, distributed


def _parse_datetime_from_filename(path: Path) -> Optional[pd.Timestamp]:
    match = TS_RE.search(path.name)
    if not match:
        return None
    try:
        dt = datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M")
    except ValueError:
        return None
    return pd.Timestamp(dt)


def _collect_frames(input_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    skipped = 0
    for p in sorted(input_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
            continue
        dt = _parse_datetime_from_filename(p)
        if dt is None:
            skipped += 1
            continue
        rows.append({"orig_path": str(p.resolve()), "datetime": dt})
    if not rows:
        raise RuntimeError(f"Nessun frame valido trovato in {input_dir}")
    if skipped:
        print(f"[WARN] Frame saltati per timestamp non parsabile: {skipped}")
    df = pd.DataFrame(rows).drop_duplicates(subset=["orig_path"])
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df[df["datetime"].notna()].sort_values(["datetime", "orig_path"]).reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"Nessun frame con datetime valido in {input_dir}")
    print(
        f"[INFO] Frames raccolti: {len(df)} "
        f"({df['datetime'].min()} -> {df['datetime'].max()})"
    )
    return df


def _make_stretched_manifest(
    frames_df: pd.DataFrame,
    input_dir: Path,
    stretched_root: Path,
    manifest_csv: Path,
    image_size: int,
) -> pd.DataFrame:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) non disponibile. Installa opencv-python nell'ambiente.")
    if stretched_root.exists():
        shutil.rmtree(stretched_root)
    stretched_root.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, object]] = []
    skipped = 0
    for row in frames_df.itertuples(index=False):
        orig_path = Path(row.orig_path)
        # Force 3 channels to keep first-pass temporal fusion shape stable (16*3=48).
        # Using IMREAD_UNCHANGED can mix RGB(3) and RGBA(4) across samples.
        img = cv2.imread(str(orig_path), cv2.IMREAD_COLOR)
        if img is None:
            skipped += 1
            continue
        orig_h, orig_w = img.shape[:2]
        stretched = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)

        try:
            rel = orig_path.relative_to(input_dir)
            dst_path = stretched_root / rel
        except ValueError:
            dst_path = stretched_root / orig_path.name
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(dst_path), stretched)
        if not ok:
            skipped += 1
            continue

        scale_x = image_size / float(orig_w)
        scale_y = image_size / float(orig_h)
        records.append(
            {
                "image_path": str(dst_path.resolve()),
                "orig_path": str(orig_path.resolve()),
                "datetime": row.datetime,
                "orig_w": int(orig_w),
                "orig_h": int(orig_h),
                "scale_x": float(scale_x),
                "scale_y": float(scale_y),
                "pad_x": 0.0,
                "pad_y": 0.0,
            }
        )

    if not records:
        raise RuntimeError("Impossibile generare copie stretched per first-pass.")
    if skipped:
        print(f"[WARN] Frame saltati nella creazione stretched: {skipped}")

    manifest_df = pd.DataFrame(records).sort_values("datetime").reset_index(drop=True)
    manifest_df.to_csv(manifest_csv, index=False)
    print(f"[INFO] Manifest first-pass salvato in {manifest_csv} ({len(manifest_df)} righe)")
    return manifest_df


def _run_firstpass_inference(
    args_cli: argparse.Namespace,
    firstpass_root: Path,
    firstpass_config: Path,
    manifest_csv: Path,
    firstpass_preds_raw_csv: Path,
    firstpass_out_dir: Path,
) -> None:
    device = args_cli.firstpass_device or ("cuda" if torch.cuda.is_available() else "cpu")
    cmd = [
        sys.executable,
        "-m",
        "cyclone_locator.infer",
        "--config",
        str(firstpass_config),
        "--checkpoint",
        str(args_cli.firstpass_model_path),
        "--manifest_csv",
        str(manifest_csv),
        "--save-preds",
        str(firstpass_preds_raw_csv),
        "--out_dir",
        str(firstpass_out_dir),
        "--threshold",
        str(args_cli.firstpass_threshold),
        "--batch-size",
        str(args_cli.firstpass_batch_size),
        "--num-workers",
        str(args_cli.firstpass_num_workers),
        "--device",
        device,
    ]
    if args_cli.firstpass_soft_argmax:
        cmd.append("--soft-argmax")
    if args_cli.firstpass_soft_argmax_tau is not None:
        cmd.extend(["--soft-argmax-tau", str(args_cli.firstpass_soft_argmax_tau)])

    print("[INFO] Eseguo first-pass inference...")
    env = os.environ.copy()
    src_dir = str((firstpass_root / "src").resolve())
    existing_pythonpath = env.get("PYTHONPATH", "")
    if existing_pythonpath:
        env["PYTHONPATH"] = src_dir + os.pathsep + existing_pythonpath
    else:
        env["PYTHONPATH"] = src_dir
    subprocess.run(cmd, cwd=str(firstpass_root), env=env, check=True)
    if not firstpass_preds_raw_csv.exists():
        raise RuntimeError(f"Predizioni first-pass non trovate: {firstpass_preds_raw_csv}")


def _build_firstpass_predictions(
    firstpass_preds_raw_csv: Path,
    manifest_df: pd.DataFrame,
    firstpass_threshold: float,
    out_csv: Path,
) -> pd.DataFrame:
    preds = pd.read_csv(firstpass_preds_raw_csv)
    if "image_path" not in preds.columns:
        raise RuntimeError("CSV first-pass senza colonna image_path.")
    for col in ["presence_prob", "x_g", "y_g"]:
        if col not in preds.columns:
            raise RuntimeError(f"CSV first-pass senza colonna {col}.")
    preds["image_path"] = preds["image_path"].astype(str).apply(lambda p: str(Path(p).resolve()))

    manifest = manifest_df.copy()
    manifest["image_path"] = manifest["image_path"].astype(str).apply(lambda p: str(Path(p).resolve()))
    merged = preds.merge(manifest, on="image_path", how="inner")
    if merged.empty:
        raise RuntimeError("Join vuoto tra predizioni first-pass e manifest stretched.")

    merged["presence_prob"] = pd.to_numeric(merged["presence_prob"], errors="coerce")
    merged["x_g"] = pd.to_numeric(merged["x_g"], errors="coerce")
    merged["y_g"] = pd.to_numeric(merged["y_g"], errors="coerce")
    merged["scale_x"] = pd.to_numeric(merged["scale_x"], errors="coerce")
    merged["scale_y"] = pd.to_numeric(merged["scale_y"], errors="coerce")
    merged["orig_w"] = pd.to_numeric(merged["orig_w"], errors="coerce")
    merged["orig_h"] = pd.to_numeric(merged["orig_h"], errors="coerce")

    merged["x_orig"] = (merged["x_g"] - merged["pad_x"]) / merged["scale_x"]
    merged["y_orig"] = (merged["y_g"] - merged["pad_y"]) / merged["scale_y"]
    merged["x_orig"] = np.clip(merged["x_orig"], 0, np.maximum(merged["orig_w"] - 1, 0))
    merged["y_orig"] = np.clip(merged["y_orig"], 0, np.maximum(merged["orig_h"] - 1, 0))
    merged["presence_pred"] = (merged["presence_prob"] >= float(firstpass_threshold)).astype(int)
    merged["datetime"] = pd.to_datetime(merged["datetime"], errors="coerce")
    merged = (
        merged.sort_values("datetime")
        .drop_duplicates(subset=["orig_path"], keep="last")
        .reset_index(drop=True)
    )

    keep_cols = [
        "orig_path",
        "image_path",
        "datetime",
        "presence_prob",
        "presence_pred",
        "x_g",
        "y_g",
        "x_orig",
        "y_orig",
    ]
    merged[keep_cols].to_csv(out_csv, index=False)
    print(f"[INFO] Predizioni first-pass (+backprojection) salvate in {out_csv}")
    return merged


def _split_contiguous_groups(frames_df: pd.DataFrame, max_gap_minutes: float) -> List[pd.DataFrame]:
    df = frames_df.copy().sort_values("datetime").reset_index(drop=True)
    df["delta_min"] = df["datetime"].diff().dt.total_seconds().div(60.0)
    df["new_group"] = df["delta_min"] > float(max_gap_minutes)
    df["group_id"] = df["new_group"].cumsum()
    return [g.drop(columns=["delta_min", "new_group", "group_id"]) for _, g in df.groupby("group_id")]


def _build_clip_candidates(
    frames_df: pd.DataFrame,
    firstpass_df: pd.DataFrame,
    num_frames: int,
    tile_size: int,
    threshold: float,
    max_gap_minutes: float,
    standard_tiling: bool = False,
) -> pd.DataFrame:
    fp = firstpass_df.copy()
    fp["orig_path"] = fp["orig_path"].astype(str).apply(lambda p: str(Path(p).resolve()))
    fp_map = fp.set_index("orig_path")[["presence_prob", "x_orig", "y_orig"]]

    groups = _split_contiguous_groups(frames_df, max_gap_minutes=max_gap_minutes)
    rows: List[Dict[str, object]] = []
    half = tile_size / 2.0

    for group in groups:
        g = group.reset_index(drop=True).copy()
        if g.empty:
            continue
        base_ts = pd.Timestamp(g["datetime"].iloc[0])
        delta_min = (g["datetime"] - base_ts).dt.total_seconds().div(60.0).to_numpy(dtype=float)
        aligned = np.isclose(
            np.mod(delta_min, float(FIRSTPASS_CLIP_STRIDE_MINUTES)),
            0.0,
            atol=1e-6,
        )
        end_idxs = g.index[aligned]
        if len(end_idxs) == 0:
            end_idxs = g.index
        for end_idx in end_idxs:
            start_idx = int(end_idx) - (num_frames - 1)
            if start_idx < 0:
                continue
            block = g.iloc[start_idx : int(end_idx) + 1].copy()
            if len(block) != num_frames:
                continue
            end_row = block.iloc[-1]
            end_path = str(Path(end_row["orig_path"]).resolve())
            presence_prob = np.nan
            x_orig = np.nan
            y_orig = np.nan
            if end_path in fp_map.index:
                info = fp_map.loc[end_path]
                presence_prob = float(info["presence_prob"])
                x_orig = float(info["x_orig"])
                y_orig = float(info["y_orig"])

            is_positive = bool(np.isfinite(presence_prob) and presence_prob >= float(threshold))
            tile_offset_x = np.nan
            tile_offset_y = np.nan
            tile_folder = ""
            if is_positive and np.isfinite(x_orig) and np.isfinite(y_orig):
                if standard_tiling:
                    std_offset = _select_standard_offset_for_center(
                        x=float(x_orig),
                        y=float(y_orig),
                        tile_size=tile_size,
                    )
                    tile_offset_x = int(std_offset[0])
                    tile_offset_y = int(std_offset[1])
                else:
                    tile_offset_x = int(round(x_orig - half))
                    tile_offset_y = int(round(y_orig - half))
                tile_folder = (
                    f"{pd.Timestamp(end_row['datetime']).strftime('%d-%m-%Y_%H%M')}"
                    f"_{int(tile_offset_x)}_{int(tile_offset_y)}"
                )

            rows.append(
                {
                    "end_datetime": pd.Timestamp(end_row["datetime"]),
                    "end_orig_path": end_path,
                    "firstpass_presence_prob": presence_prob,
                    "firstpass_x_orig": x_orig,
                    "firstpass_y_orig": y_orig,
                    "is_positive": int(is_positive),
                    "tile_offset_x": tile_offset_x,
                    "tile_offset_y": tile_offset_y,
                    "tile_folder": tile_folder,
                    "frame_paths": json.dumps(block["orig_path"].tolist()),
                }
            )

    candidates = pd.DataFrame(rows).sort_values("end_datetime").reset_index(drop=True)
    if candidates.empty:
        raise RuntimeError("Nessuna clip candidata generata. Verifica num_frames/input timeline.")
    return candidates


def _standard_offsets(tile_size: int) -> List[Tuple[int, int]]:
    offsets: List[Tuple[int, int]] = []
    for oy in range(0, STANDARD_IMAGE_HEIGHT - tile_size + 1, STANDARD_TILE_STRIDE_Y):
        for ox in range(0, STANDARD_IMAGE_WIDTH - tile_size + 1, STANDARD_TILE_STRIDE_X):
            offsets.append((ox, oy))
    return offsets


def _select_standard_offset_for_center(x: float, y: float, tile_size: int) -> Tuple[int, int]:
    """Return the standard-grid tile offset containing point (x, y)."""
    offsets = _standard_offsets(tile_size=tile_size)

    containing = [
        (ox, oy)
        for ox, oy in offsets
        if (ox <= x < ox + tile_size) and (oy <= y < oy + tile_size)
    ]
    if containing:
        # In overlap regions choose the tile whose center is closest to the point.
        return min(
            containing,
            key=lambda off: (off[0] + tile_size / 2.0 - x) ** 2 + (off[1] + tile_size / 2.0 - y) ** 2,
        )

    # Defensive fallback for malformed coordinates: clamp centered crop in-bounds.
    ox = int(np.clip(round(x - tile_size / 2.0), 0, STANDARD_IMAGE_WIDTH - tile_size))
    oy = int(np.clip(round(y - tile_size / 2.0), 0, STANDARD_IMAGE_HEIGHT - tile_size))
    return ox, oy


def _create_tile_folders(
    candidates_df: pd.DataFrame,
    tile_root: Path,
    tile_size: int,
) -> List[Path]:
    if tile_root.exists():
        shutil.rmtree(tile_root)
    tile_root.mkdir(parents=True, exist_ok=True)

    created: List[Path] = []
    for row in candidates_df.itertuples(index=False):
        if int(row.is_positive) != 1:
            continue
        folder_name = str(row.tile_folder).strip()
        if not folder_name:
            continue
        if not np.isfinite(row.tile_offset_x) or not np.isfinite(row.tile_offset_y):
            continue
        frame_paths = json.loads(row.frame_paths)
        if not isinstance(frame_paths, list) or not frame_paths:
            continue

        offset_x = int(row.tile_offset_x)
        offset_y = int(row.tile_offset_y)
        folder = tile_root / folder_name
        folder.mkdir(parents=True, exist_ok=True)

        for idx, frame_path in enumerate(frame_paths):
            src = Path(frame_path)
            if not src.exists():
                continue
            with Image.open(src) as img:
                img = img.convert("RGB")
                tile = img.crop((offset_x, offset_y, offset_x + tile_size, offset_y + tile_size))
                out_name = folder / f"img_{idx + 1:05d}.png"
                tile.save(out_name)
        created.append(folder)
    return created


def _create_tiles_with_tracking_overlay(
    tile_root: Path,
    tracking_df: Optional[pd.DataFrame],
    output_root: Path,
    dot_radius: int = 4,
) -> None:
    if not tile_root.exists():
        print(f"[WARN] Tile root non trovata, salto overlay tracking: {tile_root}")
        return

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    pred_map: Dict[str, Tuple[float, float]] = {}
    if tracking_df is not None and not tracking_df.empty and "path" in tracking_df.columns:
        tdf = tracking_df.copy()
        tdf["path_base"] = tdf["path"].astype(str).apply(lambda p: os.path.basename(str(p)).strip())
        tdf["pred_x"] = pd.to_numeric(tdf.get("pred_x"), errors="coerce")
        tdf["pred_y"] = pd.to_numeric(tdf.get("pred_y"), errors="coerce")
        tdf = tdf.sort_values("path_base")
        for _, row in tdf.iterrows():
            px = row.get("pred_x")
            py = row.get("pred_y")
            if not (np.isfinite(px) and np.isfinite(py)):
                continue
            key = str(row.get("path_base", "")).strip()
            if key:
                pred_map[key] = (float(px), float(py))

    tile_dirs = sorted([p for p in tile_root.iterdir() if p.is_dir()])
    folders_total = 0
    folders_with_pred = 0
    frames_written = 0
    frames_with_dot = 0

    for folder in tile_dirs:
        folders_total += 1
        out_folder = output_root / folder.name
        out_folder.mkdir(parents=True, exist_ok=True)
        pred_xy = pred_map.get(folder.name)
        if pred_xy is not None:
            folders_with_pred += 1

        frame_files = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])
        for frame_path in frame_files:
            with Image.open(frame_path) as img:
                img = img.convert("RGB")
                if pred_xy is not None:
                    draw = ImageDraw.Draw(img)
                    x, y = pred_xy
                    draw.ellipse(
                        (
                            x - dot_radius,
                            y - dot_radius,
                            x + dot_radius,
                            y + dot_radius,
                        ),
                        fill=(255, 0, 0),
                    )
                    frames_with_dot += 1
                img.save(out_folder / frame_path.name)
                frames_written += 1

    print(
        f"[INFO] Tile overlay tracking salvate in {output_root} | "
        f"folder: {folders_total}, con_pred: {folders_with_pred}, "
        f"frame: {frames_written}, frame_con_dot: {frames_with_dot}"
    )


def _build_timeframe_csv_from_candidates(
    candidates_df: pd.DataFrame,
    tracking_df: pd.DataFrame,
    output_csv: Path,
) -> None:
    cands = candidates_df.copy()
    cands["end_datetime"] = pd.to_datetime(cands["end_datetime"], errors="coerce")
    cands = cands[cands["end_datetime"].notna()].copy()
    if cands.empty:
        raise RuntimeError("Nessuna clip candidata valida per costruire CSV finale.")

    track_map: Dict[str, pd.Series] = {}
    if tracking_df is not None and not tracking_df.empty and "path" in tracking_df.columns:
        tdf = tracking_df.copy()
        tdf["path"] = tdf["path"].astype(str).apply(lambda p: os.path.basename(str(p)))
        tdf = tdf.sort_values("path")
        for _, row in tdf.iterrows():
            track_map[row["path"]] = row

    rows: List[Dict[str, object]] = []
    for dt in sorted(cands["end_datetime"].unique()):
        subset = cands[cands["end_datetime"] == dt].copy()
        subset = subset.sort_values(["is_positive", "tile_folder"], ascending=[False, True])

        # Coarse fallback from first-pass (available even for negative clips)
        coarse_lat = np.nan
        coarse_lon = np.nan
        subset_by_prob = subset.sort_values(
            ["firstpass_presence_prob", "tile_folder"], ascending=[False, True]
        )
        for _, cand_fp in subset_by_prob.iterrows():
            x_fp = pd.to_numeric(cand_fp.get("firstpass_x_orig"), errors="coerce")
            y_fp = pd.to_numeric(cand_fp.get("firstpass_y_orig"), errors="coerce")
            if not (np.isfinite(x_fp) and np.isfinite(y_fp)):
                continue
            try:
                lat_arr, lon_arr = tracking_engine._pixels_to_latlon(
                    np.array([float(x_fp)]), np.array([float(y_fp)])
                )
                coarse_lat = float(lat_arr[0])
                coarse_lon = float(lon_arr[0])
                break
            except Exception:
                continue

        chosen = None
        for _, cand in subset.iterrows():
            folder = str(cand.get("tile_folder", "")).strip()
            if folder and folder in track_map:
                chosen = track_map[folder]
                break
        if chosen is None:
            rows.append(
                {
                    "datetime": dt,
                    "has_cyclone": 0,
                    "pred_lat": coarse_lat,
                    "pred_lon": coarse_lon,
                }
            )
        else:
            pred_lat = pd.to_numeric(chosen.get("pred_lat"), errors="coerce")
            pred_lon = pd.to_numeric(chosen.get("pred_lon"), errors="coerce")
            if not np.isfinite(pred_lat):
                pred_lat = coarse_lat
            if not np.isfinite(pred_lon):
                pred_lon = coarse_lon
            rows.append(
                {
                    "datetime": dt,
                    "has_cyclone": 1,
                    "pred_lat": pred_lat,
                    "pred_lon": pred_lon,
                }
            )

    out_df = pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)
    out_df["has_cyclone"] = out_df["has_cyclone"].astype("Int8")
    out_df["pred_lat"] = pd.to_numeric(out_df["pred_lat"], errors="coerce").round(2)
    out_df["pred_lon"] = pd.to_numeric(out_df["pred_lon"], errors="coerce").round(2)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)


def _first_numeric_value(val: object) -> float:
    if val is None:
        return float("nan")
    if isinstance(val, (list, tuple, np.ndarray, pd.Series)):
        if len(val) == 0:
            return float("nan")
        try:
            return float(val[0])
        except Exception:
            return float("nan")
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return float("nan")
        if s.startswith("[") and s.endswith("]"):
            try:
                arr = json.loads(s)
                if isinstance(arr, list) and arr:
                    return float(arr[0])
            except Exception:
                try:
                    arr = ast.literal_eval(s)
                    if isinstance(arr, (list, tuple)) and len(arr) > 0:
                        return float(arr[0])
                except Exception:
                    pass
        try:
            return float(s)
        except Exception:
            return float("nan")
    try:
        return float(val)
    except Exception:
        return float("nan")


def _as_naive_timestamp(ts: object) -> pd.Timestamp:
    out = pd.Timestamp(ts)
    if out.tzinfo is not None:
        try:
            out = out.tz_localize(None)
        except Exception:
            try:
                out = out.tz_convert(None)
            except Exception:
                pass
    return out


def _parse_datetime_series(values: pd.Series) -> pd.Series:
    # Try generic parse first, then dayfirst fallback if it parses more rows.
    s1 = pd.to_datetime(values, errors="coerce", utc=False)
    s2 = pd.to_datetime(values, errors="coerce", dayfirst=True, utc=False)
    if s2.notna().sum() > s1.notna().sum():
        out = s2
    else:
        out = s1
    try:
        if hasattr(out.dt, "tz") and out.dt.tz is not None:
            out = out.dt.tz_localize(None)
    except Exception:
        pass
    return out


def _load_gt_points(manos_file: Optional[str]) -> pd.DataFrame:
    if not manos_file:
        return pd.DataFrame(columns=["time", "x_pix", "y_pix", "start_time", "end_time"])
    manos_path = Path(manos_file)
    if not manos_path.exists():
        print(f"[WARN] manos_file non trovato per overlay GT: {manos_file}")
        return pd.DataFrame(columns=["time", "x_pix", "y_pix", "start_time", "end_time"])
    try:
        df = pd.read_csv(manos_path)
    except Exception as exc:
        print(f"[WARN] Impossibile leggere manos_file per overlay GT ({manos_file}): {exc}")
        return pd.DataFrame(columns=["time", "x_pix", "y_pix", "start_time", "end_time"])

    time_col = None
    for cand in ("time", "datetime", "timestamp"):
        if cand in df.columns:
            time_col = cand
            break
    if time_col is None or not {"x_pix", "y_pix"}.issubset(df.columns):
        print(
            "[WARN] manos_file senza colonne richieste per overlay GT "
            f"(time/datetime/timestamp + x_pix + y_pix): {manos_file}"
        )
        return pd.DataFrame(columns=["time", "x_pix", "y_pix", "start_time", "end_time"])

    work = df.copy()
    work["time"] = _parse_datetime_series(work[time_col])
    work["x_pix"] = work["x_pix"].apply(_first_numeric_value)
    work["y_pix"] = work["y_pix"].apply(_first_numeric_value)
    # Optional presence window columns (if present in manos_file).
    if "start_time" in work.columns:
        work["start_time"] = _parse_datetime_series(work["start_time"])
    else:
        work["start_time"] = pd.NaT
    if "end_time" in work.columns:
        work["end_time"] = _parse_datetime_series(work["end_time"])
    else:
        work["end_time"] = pd.NaT
    work = work[work["time"].notna()].copy()
    work = work[np.isfinite(work["x_pix"]) & np.isfinite(work["y_pix"])].copy()
    if work.empty:
        print(f"[WARN] Nessun punto GT valido in manos_file: {manos_file}")
        return pd.DataFrame(columns=["time", "x_pix", "y_pix", "start_time", "end_time"])

    work["time"] = work["time"].apply(_as_naive_timestamp)
    work["start_time"] = pd.to_datetime(work["start_time"], errors="coerce")
    work["end_time"] = pd.to_datetime(work["end_time"], errors="coerce")
    # Keep timestamps naive for consistent merge/compare.
    work["start_time"] = work["start_time"].apply(lambda x: _as_naive_timestamp(x) if pd.notna(x) else pd.NaT)
    work["end_time"] = work["end_time"].apply(lambda x: _as_naive_timestamp(x) if pd.notna(x) else pd.NaT)
    work = work.sort_values("time").reset_index(drop=True)
    out = work[["time", "x_pix", "y_pix", "start_time", "end_time"]].copy()
    print(f"[INFO] GT points caricati per overlay video: {len(out)}")
    return out


def _build_gt_time_maps(gt_points: pd.DataFrame) -> Dict[str, Dict[pd.Timestamp, Tuple[float, float]]]:
    if gt_points is None or gt_points.empty:
        return {
            "exact": {},
            "round": {},
            "floor": {},
            "ceil": {},
        }
    exact: Dict[pd.Timestamp, Tuple[float, float]] = {}
    rounded: Dict[pd.Timestamp, Tuple[float, float]] = {}
    floored: Dict[pd.Timestamp, Tuple[float, float]] = {}
    ceiled: Dict[pd.Timestamp, Tuple[float, float]] = {}
    for _, row in gt_points.iterrows():
        ts = _as_naive_timestamp(row["time"])
        xy = (float(row["x_pix"]), float(row["y_pix"]))
        exact[ts] = xy
        rounded[ts.round("h")] = xy
        floored[ts.floor("h")] = xy
        ceiled[ts.ceil("h")] = xy
    return {
        "exact": exact,
        "round": rounded,
        "floor": floored,
        "ceil": ceiled,
    }


def _lookup_gt_xy(ts: pd.Timestamp, gt_maps: Dict[str, Dict[pd.Timestamp, Tuple[float, float]]]) -> Tuple[float, float]:
    tt = _as_naive_timestamp(ts)
    for key in ("exact", "round", "floor", "ceil"):
        m = gt_maps.get(key, {})
        if tt in m:
            return m[tt]
    return float("nan"), float("nan")


def _build_tracking_lookups(
    tracking_df: Optional[pd.DataFrame],
) -> Tuple[Dict[str, Dict[str, float]], Dict[pd.Timestamp, Dict[str, float]]]:
    by_tile: Dict[str, Dict[str, float]] = {}
    by_time: Dict[pd.Timestamp, Dict[str, float]] = {}
    if tracking_df is None or tracking_df.empty or "path" not in tracking_df.columns:
        return by_tile, by_time

    tdf = tracking_df.copy()
    tdf["path_key"] = tdf["path"].astype(str).apply(lambda p: os.path.basename(str(p)).strip())
    for col in ["pred_x_global", "pred_y_global", "target_x_global", "target_y_global"]:
        tdf[col] = pd.to_numeric(tdf.get(col), errors="coerce")
    tdf["tile_dt"] = tdf["path_key"].apply(
        lambda name: _parse_tile_folder_name(name)[0]
        if _parse_tile_folder_name(name) is not None
        else pd.NaT
    )

    for _, row in tdf.iterrows():
        key = str(row.get("path_key", "")).strip()
        if not key:
            continue
        rec = {
            "pred_x_global": row.get("pred_x_global"),
            "pred_y_global": row.get("pred_y_global"),
            "target_x_global": row.get("target_x_global"),
            "target_y_global": row.get("target_y_global"),
        }
        by_tile[key] = rec

    valid_time = tdf[tdf["tile_dt"].notna()].copy()
    if not valid_time.empty:
        valid_time["has_gt"] = valid_time["target_x_global"].notna() & valid_time["target_y_global"].notna()
        valid_time["has_pred"] = valid_time["pred_x_global"].notna() & valid_time["pred_y_global"].notna()
        valid_time = valid_time.sort_values(
            ["tile_dt", "has_gt", "has_pred", "path_key"],
            ascending=[True, False, False, True],
        )
        for dt, group in valid_time.groupby("tile_dt"):
            row = group.iloc[0]
            by_time[pd.Timestamp(dt)] = {
                "pred_x_global": row.get("pred_x_global"),
                "pred_y_global": row.get("pred_y_global"),
                "target_x_global": row.get("target_x_global"),
                "target_y_global": row.get("target_y_global"),
            }
    return by_tile, by_time


def _draw_marker_if_finite(
    img: np.ndarray,
    x: object,
    y: object,
    *,
    marker_type: int,
    color_bgr: Tuple[int, int, int],
    size: int,
    thickness: int,
) -> bool:
    xx = pd.to_numeric(x, errors="coerce")
    yy = pd.to_numeric(y, errors="coerce")
    if not (np.isfinite(xx) and np.isfinite(yy)):
        return False
    cv2.drawMarker(
        img,
        (int(round(float(xx))), int(round(float(yy)))),
        color_bgr,
        markerType=marker_type,
        markerSize=int(size),
        thickness=int(thickness),
    )
    return True


def _run_tracking_inference(
    args_cli: argparse.Namespace,
    positive_folders: Sequence[Path],
    output_dir: Path,
    track_tiles_csv: Path,
    device: torch.device,
    world_size: int,
    rank: int,
    distributed: bool,
) -> None:
    if track_tiles_csv.exists():
        if rank == 0:
            print(f"[INFO] Tracking gia presente in {track_tiles_csv}: salto inferenza.")
        return

    if not positive_folders:
        if rank == 0:
            print("[WARN] Nessuna clip positiva first-pass: salto tracking.")
            empty = pd.DataFrame(columns=["path", "pred_lat", "pred_lon"])
            empty.to_csv(track_tiles_csv, index=False)
        return

    args_tracking = prepare_tracking_args(machine=args_cli.on)
    args_tracking.output_dir = str(output_dir)
    args_tracking.pretrained = False
    args_tracking.init_ckpt = ""
    args_tracking.load_for_test_mode = True

    set_seeds(args_tracking.seed)

    tile_infos: List[Tuple[str, pd.Timestamp, float, float]] = []
    for folder in positive_folders:
        parsed = _parse_tile_folder_name(folder.name)
        if parsed is None:
            print(f"[WARN] Nome tile non valido (saltata GT): {folder.name}")
            continue
        dt_floor, off_x, off_y = parsed
        tile_infos.append((str(folder), dt_floor, off_x, off_y))

    gt_map = _build_gt_map_from_tracks(args_cli.manos_file, tile_infos)
    if rank == 0:
        print(
            f"[INFO] GT match su tile tracking: {len(gt_map)}/{len(positive_folders)} "
            f"(manos_file={args_cli.manos_file})"
        )
    tmp_csv = output_dir / "_tmp_tracking_inference_dataset.csv"
    _build_tracking_csv_for_folders(positive_folders, gt_map, tmp_csv)

    dataset = MedicanesTrackDataset(
        anno_path=str(tmp_csv),
        data_root="",
        clip_len=args_tracking.num_frames,
        transform=None,
    )

    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)

    data_loader = DataLoader(
        dataset,
        batch_size=args_tracking.batch_size,
        num_workers=args_tracking.num_workers,
        pin_memory=args_tracking.pin_mem,
        sampler=sampler,
    )

    model = create_tracking_model(args_tracking.model, **args_tracking.__dict__)
    model.to(device)
    if rank == 0:
        print(f"[INFO] Loading tracking checkpoint: {args_cli.tracking_model_path}")
    load_checkpoint(model, args_cli.tracking_model_path, device)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index], output_device=device.index
        )

    _ = run_tracking_inference(
        model=model,
        data_loader=data_loader,
        device=device,
        output_dir=str(output_dir),
        preds_csv=track_tiles_csv.name,
    )


def _render_firstpass_roi_frames(
    frames_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    tracking_df: Optional[pd.DataFrame],
    manos_file: Optional[str],
    frames_dir: Path,
    tile_size: int,
) -> int:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) non disponibile. Installa opencv-python nell'ambiente.")
    frames_dir.mkdir(parents=True, exist_ok=True)

    cands = candidates_df.copy()
    cands["end_datetime"] = pd.to_datetime(cands["end_datetime"], errors="coerce")
    cands = cands[cands["end_datetime"].notna()].copy()
    if cands.empty or frames_df is None or frames_df.empty:
        return 0

    timeline = frames_df.copy()
    if "orig_path" not in timeline.columns or "datetime" not in timeline.columns:
        raise RuntimeError("frames_df deve contenere colonne orig_path e datetime.")
    timeline["datetime"] = pd.to_datetime(timeline["datetime"], errors="coerce")
    timeline = timeline[timeline["datetime"].notna()].copy()
    timeline["orig_path"] = timeline["orig_path"].astype(str).apply(lambda p: str(Path(p).resolve()))
    timeline = timeline.sort_values(["datetime", "orig_path"]).drop_duplicates(
        subset=["datetime", "orig_path"],
        keep="last",
    )
    if timeline.empty:
        return 0

    track_by_tile, track_by_time = _build_tracking_lookups(tracking_df)
    gt_points = _load_gt_points(manos_file)

    def _candidate_has_tracking(row: pd.Series) -> int:
        folder = str(row.get("tile_folder", "")).strip()
        if folder and folder in track_by_tile:
            rec = track_by_tile[folder]
            px = pd.to_numeric(rec.get("pred_x_global"), errors="coerce")
            py = pd.to_numeric(rec.get("pred_y_global"), errors="coerce")
            if np.isfinite(px) and np.isfinite(py):
                return 1
        return 0

    cands["has_tracking"] = cands.apply(_candidate_has_tracking, axis=1)
    # Una sola entry per timestamp: preferisci candidati con tracking, poi più confidente.
    cands = cands.sort_values(
        ["end_datetime", "has_tracking", "firstpass_presence_prob"],
        ascending=[True, False, False],
    ).drop_duplicates(subset=["end_datetime"], keep="first")

    state_rows: List[Dict[str, object]] = []
    written = 0
    half = float(tile_size) / 2.0
    n_firstpass_marker = 0
    n_tracking_marker = 0
    n_gt_marker = 0
    n_gt_from_tracking = 0
    n_gt_from_manos = 0
    for row in cands.itertuples(index=False):
        draw_box = int(row.is_positive) == 1
        off_x = pd.to_numeric(row.tile_offset_x, errors="coerce")
        off_y = pd.to_numeric(row.tile_offset_y, errors="coerce")
        if draw_box and not (np.isfinite(off_x) and np.isfinite(off_y)):
            cx = pd.to_numeric(row.firstpass_x_orig, errors="coerce")
            cy = pd.to_numeric(row.firstpass_y_orig, errors="coerce")
            if np.isfinite(cx) and np.isfinite(cy):
                off_x = int(round(float(cx) - half))
                off_y = int(round(float(cy) - half))

        ts = pd.Timestamp(row.end_datetime)
        folder_name = str(getattr(row, "tile_folder", "")).strip()
        track_rec = track_by_tile.get(folder_name)
        if track_rec is None:
            track_rec = track_by_time.get(ts)
        if track_rec is None:
            track_rec = track_by_time.get(ts.round("h"))
        track_x = np.nan if track_rec is None else pd.to_numeric(track_rec.get("pred_x_global"), errors="coerce")
        track_y = np.nan if track_rec is None else pd.to_numeric(track_rec.get("pred_y_global"), errors="coerce")

        gt_x = np.nan
        gt_y = np.nan
        gt_from_tracking = 0
        if track_rec is not None:
            gt_x = pd.to_numeric(track_rec.get("target_x_global"), errors="coerce")
            gt_y = pd.to_numeric(track_rec.get("target_y_global"), errors="coerce")
            gt_from_tracking = int(np.isfinite(gt_x) and np.isfinite(gt_y))

        state_rows.append(
            {
                "state_time": ts,
                "draw_box": int(draw_box),
                "tile_offset_x": off_x,
                "tile_offset_y": off_y,
                "firstpass_x_orig": pd.to_numeric(getattr(row, "firstpass_x_orig", np.nan), errors="coerce"),
                "firstpass_y_orig": pd.to_numeric(getattr(row, "firstpass_y_orig", np.nan), errors="coerce"),
                "track_x": track_x,
                "track_y": track_y,
                "gt_x_track": gt_x,
                "gt_y_track": gt_y,
                "gt_from_tracking": gt_from_tracking,
            }
        )

    state_df = pd.DataFrame(state_rows).sort_values("state_time")
    state_df["state_time"] = pd.to_datetime(state_df["state_time"], errors="coerce")
    state_df = state_df[state_df["state_time"].notna()].copy()

    render_df = timeline.rename(columns={"orig_path": "frame_path"}).copy()
    render_df = pd.merge_asof(
        render_df.sort_values("datetime"),
        state_df.sort_values("state_time"),
        left_on="datetime",
        right_on="state_time",
        direction="backward",
    )

    if gt_points is not None and not gt_points.empty:
        gt_src = gt_points.copy().sort_values("time").rename(columns={"time": "gt_time"})
        gt_hold = pd.merge_asof(
            render_df[["datetime"]].copy().sort_values("datetime"),
            gt_src[["gt_time", "x_pix", "y_pix", "start_time", "end_time"]].sort_values("gt_time"),
            left_on="datetime",
            right_on="gt_time",
            direction="backward",
        )
        gt_x = pd.to_numeric(gt_hold["x_pix"], errors="coerce")
        gt_y = pd.to_numeric(gt_hold["y_pix"], errors="coerce")
        gt_start = pd.to_datetime(gt_hold.get("start_time"), errors="coerce")
        gt_end = pd.to_datetime(gt_hold.get("end_time"), errors="coerce")
        # Show GT only inside the cyclone presence window if available.
        valid_mask = np.isfinite(gt_x) & np.isfinite(gt_y)
        has_start = gt_start.notna()
        has_end = gt_end.notna()
        valid_mask = valid_mask & (~has_start | (gt_hold["datetime"] >= gt_start))
        valid_mask = valid_mask & (~has_end | (gt_hold["datetime"] <= gt_end))
        render_df["gt_x_manos"] = gt_x.where(valid_mask, np.nan)
        render_df["gt_y_manos"] = gt_y.where(valid_mask, np.nan)
    else:
        render_df["gt_x_manos"] = np.nan
        render_df["gt_y_manos"] = np.nan

    for row in render_df.itertuples(index=False):
        img_path = Path(str(row.frame_path))
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]

        draw_box_val = pd.to_numeric(getattr(row, "draw_box", 0), errors="coerce")
        draw_box = bool(np.isfinite(draw_box_val) and int(draw_box_val) == 1)
        off_x = pd.to_numeric(getattr(row, "tile_offset_x", np.nan), errors="coerce")
        off_y = pd.to_numeric(getattr(row, "tile_offset_y", np.nan), errors="coerce")
        if draw_box and np.isfinite(off_x) and np.isfinite(off_y):
            x0 = int(np.clip(int(round(float(off_x))), 0, max(0, w - 1)))
            y0 = int(np.clip(int(round(float(off_y))), 0, max(0, h - 1)))
            x1 = int(np.clip(x0 + int(tile_size), 0, max(0, w - 1)))
            y1 = int(np.clip(y0 + int(tile_size), 0, max(0, h - 1)))
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 2)

        if _draw_marker_if_finite(
            img,
            getattr(row, "firstpass_x_orig", np.nan),
            getattr(row, "firstpass_y_orig", np.nan),
            marker_type=cv2.MARKER_DIAMOND,
            color_bgr=(0, 0, 255),
            size=16,
            thickness=2,
        ):
            n_firstpass_marker += 1

        track_x = pd.to_numeric(getattr(row, "track_x", np.nan), errors="coerce")
        track_y = pd.to_numeric(getattr(row, "track_y", np.nan), errors="coerce")
        if np.isfinite(track_x) and np.isfinite(track_y):
            cv2.circle(img, (int(round(float(track_x))), int(round(float(track_y)))), 4, (0, 0, 255), -1)
            cv2.circle(img, (int(round(float(track_x))), int(round(float(track_y)))), 6, (255, 255, 255), 1)
            n_tracking_marker += 1

        gt_x = pd.to_numeric(getattr(row, "gt_x_track", np.nan), errors="coerce")
        gt_y = pd.to_numeric(getattr(row, "gt_y_track", np.nan), errors="coerce")
        gt_from_tracking_val = pd.to_numeric(getattr(row, "gt_from_tracking", 0), errors="coerce")
        gt_from_tracking = bool(np.isfinite(gt_from_tracking_val) and int(gt_from_tracking_val) == 1)
        if not (np.isfinite(gt_x) and np.isfinite(gt_y)):
            gt_x = pd.to_numeric(getattr(row, "gt_x_manos", np.nan), errors="coerce")
            gt_y = pd.to_numeric(getattr(row, "gt_y_manos", np.nan), errors="coerce")
            gt_from_tracking = False
        if np.isfinite(gt_x) and np.isfinite(gt_y):
            cv2.circle(img, (int(round(float(gt_x))), int(round(float(gt_y)))), 4, (0, 255, 0), -1)
            cv2.circle(img, (int(round(float(gt_x))), int(round(float(gt_y)))), 6, (255, 255, 255), 1)
            n_gt_marker += 1
            if gt_from_tracking:
                n_gt_from_tracking += 1
            else:
                n_gt_from_manos += 1

        label = pd.Timestamp(row.datetime).strftime("%Y-%m-%d %H:%M")
        cv2.putText(
            img,
            label,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            label,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        out_png = frames_dir / f"frame_{written:05d}.png"
        if cv2.imwrite(str(out_png), img):
            written += 1
    print(
        f"[INFO] Video overlay markers - firstpass: {n_firstpass_marker}, "
        f"tracking: {n_tracking_marker}, gt: {n_gt_marker}"
    )
    print(
        f"[INFO] GT overlay source - da tracking: {n_gt_from_tracking}, "
        f"da manos_file: {n_gt_from_manos}"
    )
    return written


def _encode_video_from_frames(
    frames_dir: Path,
    output_mp4: Path,
    ffmpeg_path: Optional[str],
    fps: int = 10,
) -> None:
    ffmpeg_exec = resolve_ffmpeg_executable(ffmpeg_path)
    if ffmpeg_exec is None:
        raise RuntimeError(
            "ffmpeg non trovato nel PATH o nel bundle locale. "
            "Specifica --ffmpeg_path."
        )
    frame_list = sorted(frames_dir.glob("frame_*.png"))
    if not frame_list:
        raise RuntimeError(f"Nessun frame PNG trovato in {frames_dir}")
    cmd = [
        ffmpeg_exec,
        "-y",
        "-framerate",
        str(int(fps)),
        "-i",
        str(frames_dir / "frame_%05d.png"),
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        str(output_mp4),
    ]
    subprocess.run(cmd, check=True)


def _make_firstpass_roi_video(
    args_cli: argparse.Namespace,
    frames_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    tracking_df: Optional[pd.DataFrame],
    output_dir: Path,
) -> Path:
    frames_dir = output_dir / f"anim_frames_{args_cli.video_name}"
    output_mp4 = output_dir / f"{args_cli.video_name}.mp4"
    if args_cli.only_video:
        if not frames_dir.exists():
            raise RuntimeError(
                f"Cartella frame non trovata: {frames_dir}. "
                "Rimuovi --only_video o genera prima i frame."
            )
    else:
        n_frames = _render_firstpass_roi_frames(
            frames_df=frames_df,
            candidates_df=candidates_df,
            tracking_df=tracking_df,
            manos_file=args_cli.manos_file,
            frames_dir=frames_dir,
            tile_size=int(args_cli.tile_size),
        )
        if n_frames == 0:
            raise RuntimeError("Nessun frame renderizzato per il video ROI first-pass.")

    _encode_video_from_frames(
        frames_dir=frames_dir,
        output_mp4=output_mp4,
        ffmpeg_path=args_cli.ffmpeg_path,
        fps=10,
    )
    return output_mp4


def main() -> None:
    args_cli = parse_args()
    if args_cli.end_on_hour is not None:
        print(
            "[WARN] --end_on_hour/--no-end_on_hour e' deprecato e ignorato: "
            f"la stride clip first-pass e' fissata a {FIRSTPASS_CLIP_STRIDE_MINUTES} minuti."
        )
    if args_cli.only_video and not args_cli.make_video:
        raise RuntimeError("--only_video richiede --make_video.")
    output_dir = Path(args_cli.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stretched_dir = output_dir / "_tmp_firstpass_stretched"
    manifest_csv = output_dir / "_tmp_firstpass_manifest.csv"
    firstpass_out_dir = output_dir / "_tmp_firstpass_out"
    firstpass_preds_raw_csv = firstpass_out_dir / "preds_firstpass_raw.csv"
    firstpass_preds_csv = output_dir / "_tmp_firstpass_predictions.csv"
    clip_candidates_csv = output_dir / "_tmp_firstpass_clip_candidates.csv"
    tile_root = output_dir / "firstpass_tiles"
    track_tiles_csv = output_dir / "_tmp_tracking_inference_predictions_tiles.csv"
    final_time_csv = output_dir / "tracking_inference_predictions.csv"

    # Fast cache path: if final CSV exists, skip all prediction stages.
    if final_time_csv.exists():
        print(f"[INFO] CSV finale gia presente: {final_time_csv}. Salto inferenze.")
        if not args_cli.make_video:
            return
        if args_cli.only_video:
            video_path = _make_firstpass_roi_video(
                args_cli=args_cli,
                frames_df=pd.DataFrame(columns=["orig_path", "datetime"]),
                candidates_df=pd.DataFrame([]),
                tracking_df=pd.DataFrame([]),
                output_dir=output_dir,
            )
            print(f"[INFO] Video ROI first-pass generato: {video_path}")
            return
        if not clip_candidates_csv.exists():
            print(
                "[WARN] Video richiesto ma candidati clip non trovati; "
                "salto video (nessuna inferenza ricalcolata)."
            )
            return
        frames_df = _collect_frames(Path(args_cli.input_dir).resolve())
        clip_candidates = pd.read_csv(clip_candidates_csv, parse_dates=["end_datetime"])
        track_df = pd.read_csv(track_tiles_csv) if track_tiles_csv.exists() else pd.DataFrame([])
        video_path = _make_firstpass_roi_video(
            args_cli=args_cli,
            frames_df=frames_df,
            candidates_df=clip_candidates,
            tracking_df=track_df,
            output_dir=output_dir,
        )
        print(f"[INFO] Video ROI first-pass generato: {video_path}")
        return

    firstpass_root = (
        Path(args_cli.firstpass_root).resolve()
        if args_cli.firstpass_root
        else FIRSTPASS_ROOT_DEFAULT
    )
    firstpass_config = (
        Path(args_cli.firstpass_config).resolve()
        if args_cli.firstpass_config
        else (firstpass_root / "config" / "config_exp31.yml").resolve()
    )
    if not firstpass_root.exists():
        raise FileNotFoundError(f"firstpass_root non trovato: {firstpass_root}")
    if not firstpass_config.exists():
        raise FileNotFoundError(f"firstpass_config non trovato: {firstpass_config}")

    rank, local_rank, world_size, distributed = _setup_distributed()
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")

    if rank == 0:
        frames_df = _collect_frames(Path(args_cli.input_dir).resolve())
        if firstpass_preds_csv.exists():
            print(f"[INFO] Predizioni first-pass gia presenti: {firstpass_preds_csv}")
            firstpass_df = pd.read_csv(firstpass_preds_csv, parse_dates=["datetime"])
        else:
            manifest_df = _make_stretched_manifest(
                frames_df=frames_df,
                input_dir=Path(args_cli.input_dir).resolve(),
                stretched_root=stretched_dir,
                manifest_csv=manifest_csv,
                image_size=args_cli.firstpass_image_size,
            )
            if firstpass_preds_raw_csv.exists():
                firstpass_preds_raw_csv.unlink()
            _run_firstpass_inference(
                args_cli=args_cli,
                firstpass_root=firstpass_root,
                firstpass_config=firstpass_config,
                manifest_csv=manifest_csv,
                firstpass_preds_raw_csv=firstpass_preds_raw_csv,
                firstpass_out_dir=firstpass_out_dir,
            )
            firstpass_df = _build_firstpass_predictions(
                firstpass_preds_raw_csv=firstpass_preds_raw_csv,
                manifest_df=manifest_df,
                firstpass_threshold=args_cli.firstpass_threshold,
                out_csv=firstpass_preds_csv,
            )
            if firstpass_preds_raw_csv.exists():
                firstpass_preds_raw_csv.unlink()

        clip_candidates = _build_clip_candidates(
            frames_df=frames_df,
            firstpass_df=firstpass_df,
            num_frames=args_cli.num_frames,
            tile_size=args_cli.tile_size,
            threshold=args_cli.firstpass_threshold,
            max_gap_minutes=float(args_cli.max_contiguous_gap_minutes),
            standard_tiling=bool(args_cli.standard_tiling),
        )
        clip_candidates.to_csv(clip_candidates_csv, index=False)
        created_folders = _create_tile_folders(
            candidates_df=clip_candidates,
            tile_root=tile_root,
            tile_size=args_cli.tile_size,
        )
        n_pos = int((clip_candidates["is_positive"] == 1).sum())
        print(
            f"[INFO] Clip candidate: {len(clip_candidates)} | "
            f"positive first-pass: {n_pos} | tile create: {len(created_folders)} | "
            f"stride: {FIRSTPASS_CLIP_STRIDE_MINUTES} minuti"
        )

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    if not clip_candidates_csv.exists():
        raise RuntimeError(f"Candidati clip non trovati: {clip_candidates_csv}")
    clip_candidates = pd.read_csv(clip_candidates_csv, parse_dates=["end_datetime"])
    positive_names = (
        clip_candidates.loc[clip_candidates["is_positive"] == 1, "tile_folder"]
        .dropna()
        .astype(str)
        .tolist()
    )
    positive_folders = [tile_root / name for name in positive_names if name]
    positive_folders = [p for p in positive_folders if p.exists()]

    skip_tracking_due_final = final_time_csv.exists()
    if skip_tracking_due_final:
        if rank == 0:
            print(
                f"[INFO] CSV finale gia presente in {final_time_csv}: "
                "salto tracking inference."
            )
    else:
        _run_tracking_inference(
            args_cli=args_cli,
            positive_folders=positive_folders,
            output_dir=output_dir,
            track_tiles_csv=track_tiles_csv,
            device=device,
            world_size=world_size,
            rank=rank,
            distributed=distributed,
        )

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    if rank == 0:
        if track_tiles_csv.exists():
            track_df = pd.read_csv(track_tiles_csv)
        else:
            track_df = pd.DataFrame([])
        if final_time_csv.exists():
            print(f"[INFO] CSV finale timeframe gia presente: {final_time_csv} (salto rebuild)")
        else:
            _build_timeframe_csv_from_candidates(
                candidates_df=clip_candidates,
                tracking_df=track_df,
                output_csv=final_time_csv,
            )
            print(f"[INFO] CSV finale timeframe salvato in {final_time_csv}")

        if args_cli.make_video:
            video_path = _make_firstpass_roi_video(
                args_cli=args_cli,
                frames_df=frames_df,
                candidates_df=clip_candidates,
                tracking_df=track_df,
                output_dir=output_dir,
            )
            print(f"[INFO] Video ROI first-pass generato: {video_path}")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()
