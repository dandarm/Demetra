from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import os
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader, Dataset

from cyclone_locator.infer import decode_heatmap, spatial_peak_pool
from cyclone_locator.utils.metric import peak_and_width


def build_meta_by_name(meta_map: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    by_name: Dict[str, Dict[str, float]] = {}
    for k, v in meta_map.items():
        name = Path(str(k)).name
        if name and name not in by_name:
            by_name[name] = v
    return by_name


def build_manifest_map(manifest_df) -> Dict[str, Dict[str, object]]:
    manifest_cols = ["presence", "cx", "cy", "x_pix_resized", "y_pix_resized", "datetime"]
    cols_present = [c for c in manifest_cols if c in manifest_df.columns]
    if not cols_present:
        return {}
    tmp = manifest_df.set_index("image_path_abs")[cols_present]
    return tmp.to_dict(orient="index")


def _to_float(v: object) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _get_meta_for_path(
    meta_map: Dict[str, Dict[str, float]],
    meta_by_name: Dict[str, Dict[str, float]],
    path_str: str,
) -> Optional[Dict[str, float]]:
    meta = meta_map.get(path_str)
    if meta is None:
        meta = meta_by_name.get(Path(path_str).name)
    return meta


def _forward_map_xy(x_orig: float, y_orig: float, meta: Dict[str, float]) -> Tuple[float, float]:
    sx = float(meta.get("scale_x", meta["scale"]))
    sy = float(meta.get("scale_y", meta["scale"]))
    x_lb = sx * x_orig + float(meta.get("pad_x", 0.0))
    y_lb = sy * y_orig + float(meta.get("pad_y", 0.0))
    return x_lb, y_lb


def _backproject_xy(x_lb: float, y_lb: float, meta: Dict[str, float]) -> Tuple[float, float]:
    sx = float(meta.get("scale_x", meta["scale"]))
    sy = float(meta.get("scale_y", meta["scale"]))
    x_orig = (x_lb - float(meta.get("pad_x", 0.0))) / sx
    y_orig = (y_lb - float(meta.get("pad_y", 0.0))) / sy
    return x_orig, y_orig


def _get_gt_info(
    image_path_abs: str,
    meta: Dict[str, float],
    manifest_map: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    row = manifest_map.get(image_path_abs)
    if row is None:
        return {
            "gt_orig": None,
            "gt_lb": None,
            "roundtrip_orig": None,
            "roundtrip_lb": None,
            "presence_gt": None,
            "timestamp": None,
        }

    gt_cx = _to_float(row.get("cx", float("nan")))
    gt_cy = _to_float(row.get("cy", float("nan")))
    gt_x_lb = _to_float(row.get("x_pix_resized", float("nan")))
    gt_y_lb = _to_float(row.get("y_pix_resized", float("nan")))
    presence_gt = _to_float(row.get("presence", float("nan")))
    ts = row.get("datetime")

    gt_orig = None
    if np.isfinite(gt_cx) and np.isfinite(gt_cy):
        gt_orig = (gt_cx, gt_cy)
    elif np.isfinite(gt_x_lb) and np.isfinite(gt_y_lb):
        gt_orig = _backproject_xy(gt_x_lb, gt_y_lb, meta)

    gt_lb = None
    if np.isfinite(gt_x_lb) and np.isfinite(gt_y_lb):
        gt_lb = (gt_x_lb, gt_y_lb)
    elif gt_orig is not None:
        gt_lb = _forward_map_xy(gt_orig[0], gt_orig[1], meta)

    roundtrip_orig = None
    roundtrip_lb = None
    if np.isfinite(gt_x_lb) and np.isfinite(gt_y_lb):
        roundtrip_orig = _backproject_xy(gt_x_lb, gt_y_lb, meta)
        roundtrip_lb = (gt_x_lb, gt_y_lb)
    if roundtrip_orig is None and gt_lb is not None:
        roundtrip_orig = _backproject_xy(gt_lb[0], gt_lb[1], meta)
        roundtrip_lb = gt_lb

    return {
        "gt_orig": gt_orig,
        "gt_lb": gt_lb,
        "roundtrip_orig": roundtrip_orig,
        "roundtrip_lb": roundtrip_lb,
        "presence_gt": presence_gt,
        "timestamp": ts,
    }


def _to_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        frame = frame[..., None]
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    return frame


def _as_rgb_uint8(img01: np.ndarray) -> np.ndarray:
    img01 = np.clip(img01, 0.0, 1.0)
    return (img01 * 255.0).round().astype(np.uint8)


def _load_orig(path: Optional[str]) -> Optional[np.ndarray]:
    if not path or not os.path.exists(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def draw_markers_and_roi(
    rgb: np.ndarray,
    pred_xy: Optional[Tuple[float, float]],
    gt_xy: Optional[Tuple[float, float]],
    roundtrip_xy: Optional[Tuple[float, float]],
    radius: float,
) -> np.ndarray:
    img = rgb.copy()
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]

    if pred_xy is not None:
        px, py = pred_xy
        x0 = int(round(px - radius))
        y0 = int(round(py - radius))
        x1 = int(round(px + radius))
        y1 = int(round(py + radius))
        x0 = max(0, min(w - 1, x0))
        y0 = max(0, min(h - 1, y0))
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        cv2.rectangle(bgr, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.drawMarker(
            bgr,
            (int(round(px)), int(round(py))),
            (0, 0, 255),
            markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=18,
            thickness=2,
        )

    if gt_xy is not None:
        gx, gy = gt_xy
        cv2.drawMarker(
            bgr,
            (int(round(gx)), int(round(gy))),
            (0, 255, 0),
            markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=18,
            thickness=2,
        )

    if roundtrip_xy is not None:
        rx, ry = roundtrip_xy
        cv2.drawMarker(
            bgr,
            (int(round(rx)), int(round(ry))),
            (0, 255, 255),
            markerType=cv2.MARKER_DIAMOND,
            markerSize=16,
            thickness=2,
        )

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def render_event_panel(
    frame_rgb01: np.ndarray,
    hm_gt01: np.ndarray,
    hm_pr01: np.ndarray,
    *,
    title: str,
    gt_xy_hm: Tuple[float, float],
    pr_xy_hm: Tuple[float, float],
    pred_lb: Optional[Tuple[float, float]] = None,
    gt_lb: Optional[Tuple[float, float]] = None,
    roundtrip_lb: Optional[Tuple[float, float]] = None,
    heatmap_stride: int = 1,
    hm_pred_vmax: float = 1.0,
    bottom_lines: Optional[Sequence[str]] = None,
    status_text: Optional[str] = None,
    status_color: Tuple[int, int, int] = (0, 255, 0),
    series_vals: Optional[Sequence[float]] = None,
    series_index: Optional[int] = None,
    series_label: str = "",
    series_color: Tuple[int, int, int] = (255, 255, 0),
    series_marks: Optional[Sequence[Tuple[int, float]]] = None,
    gt_alpha: float = 1.0,
    gt_mask_rel: float = 0.0,
    pad: int = 6,
) -> np.ndarray:
    frame_u8 = _as_rgb_uint8(_to_rgb(frame_rgb01))
    frame_u8 = np.ascontiguousarray(frame_u8)
    h, w = frame_u8.shape[:2]

    # Pred heatmap as red-only on black (BGR -> red channel index 2)
    hm_pr_vmax = float(np.clip(hm_pred_vmax, 1e-6, 1.0))
    hm_pr_resized = cv2.resize(hm_pr01.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    hm_pr_rel = np.clip(hm_pr_resized / hm_pr_vmax, 0.0, 1.0)
    hm_pr_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    hm_pr_rgb[..., 2] = (hm_pr_rel * 255.0).round().astype(np.uint8)

    # Overlay GT heatmap on the frame
    gt_alpha = float(np.clip(gt_alpha, 0.0, 1.0))
    gt_mask_rel = float(np.clip(gt_mask_rel, 0.0, 1.0))
    hm_gt_resized = cv2.resize(hm_gt01.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    gt_max = float(np.max(hm_gt_resized)) if hm_gt_resized.size else 0.0
    if gt_max > 0.0 and gt_alpha > 0.0:
        thr = max(0.0, gt_mask_rel * gt_max)
        alpha_map = (hm_gt_resized / max(gt_max, 1e-6)).astype(np.float32) * float(gt_alpha)
        alpha_map[hm_gt_resized <= thr] = 0.0
        alpha3 = np.repeat(alpha_map[..., None], 3, axis=-1)
        base = frame_u8.astype(np.float32)
        green = np.zeros_like(base)
        green[..., 1] = 255.0
        frame_u8 = np.clip(base * (1.0 - alpha3) + green * alpha3, 0.0, 255.0).astype(np.uint8)
        frame_u8 = np.ascontiguousarray(frame_u8)

    def _draw_marker(img: np.ndarray, xy: Optional[Tuple[float, float]], color_rgb: Tuple[int, int, int],
                     marker: int, size: int = 14, thickness: int = 2) -> None:
        if xy is None:
            return
        x, y = xy
        cv2.drawMarker(img, (int(round(x)), int(round(y))), color=color_rgb,
                       markerType=marker, markerSize=size, thickness=thickness)

    def _draw_cross_hm(img: np.ndarray, x_hm: float, y_hm: float, color_rgb: Tuple[int, int, int]) -> None:
        hs, ws = hm_gt01.shape[:2]
        sx = w / float(ws)
        sy = h / float(hs)
        cx = int(round(x_hm * sx))
        cy = int(round(y_hm * sy))
        cv2.drawMarker(img, (cx, cy), color=color_rgb, markerType=cv2.MARKER_TILTED_CROSS, markerSize=6, thickness=2)

    _draw_cross_hm(hm_pr_rgb, pr_xy_hm[0], pr_xy_hm[1], (0, 0, 255))

    # Markers on frame (BGR image)
    if pred_lb is not None:
        _draw_marker(frame_u8, pred_lb, (0, 0, 255), cv2.MARKER_TILTED_CROSS, size=14, thickness=2)
    if gt_lb is not None:
        _draw_marker(frame_u8, gt_lb, (0, 255, 0), cv2.MARKER_TILTED_CROSS, size=14, thickness=2)
    if roundtrip_lb is not None:
        _draw_marker(frame_u8, roundtrip_lb, (0, 255, 255), cv2.MARKER_DIAMOND, size=12, thickness=2)

    pad_col = np.zeros((h, pad, 3), dtype=np.uint8)
    out = np.concatenate([frame_u8, pad_col, hm_pr_rgb], axis=1)

    strip_h = 78
    strip = np.zeros((strip_h, out.shape[1], 3), dtype=np.uint8)
    t1 = title[:110]
    t2 = title[110:220] if len(title) > 110 else ""
    t3 = title[220:330] if len(title) > 220 else ""
    cv2.putText(strip, t1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    if t2:
        cv2.putText(strip, t2, (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    if t3:
        cv2.putText(strip, t3, (10, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    bottom_lines = list(bottom_lines or [])
    has_series = series_vals is not None and len(series_vals) > 1
    if bottom_lines or has_series:
        bottom_h = 140
        bottom = np.zeros((bottom_h, out.shape[1], 3), dtype=np.uint8)
        y = 22
        for line in bottom_lines[:3]:
            cv2.putText(bottom, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            y += 20
        if status_text:
            cv2.putText(bottom, status_text, (bottom.shape[1] - 180, bottom_h - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2, cv2.LINE_AA)

        if has_series:
            vals = np.asarray(series_vals, dtype=np.float32)
            finite = np.isfinite(vals)
            if finite.any():
                vmin = 0.0
                vmax = 12.0
                if vmax <= vmin:
                    vmax = vmin + 1.0
                plot_top = 60
                plot_bottom = bottom_h - 12
                plot_left = 10
                plot_right = bottom.shape[1] - 10
                xs = np.linspace(plot_left, plot_right, len(vals))
                ys = plot_bottom - (vals - vmin) / (vmax - vmin) * float(plot_bottom - plot_top)
                pts = np.stack([xs, ys], axis=1).astype(np.int32)
                cv2.polylines(bottom, [pts], False, series_color, 1, cv2.LINE_AA)
                if series_label:
                    cv2.putText(bottom, series_label, (plot_left, plot_top - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
                if series_marks:
                    for idx, val in series_marks:
                        if 0 <= idx < len(vals) and np.isfinite(val):
                            x = int(round(xs[idx]))
                            yv = plot_bottom - (val - vmin) / (vmax - vmin) * float(plot_bottom - plot_top)
                            yv = int(round(np.clip(yv, plot_top, plot_bottom)))
                            cv2.line(bottom, (x, plot_top), (x, plot_bottom), (120, 120, 120), 1)
                            dash = 6
                            gap = 4
                            xx = plot_left
                            while xx < plot_right:
                                cv2.line(bottom, (xx, yv), (min(xx + dash, plot_right), yv), (120, 120, 120), 1)
                                xx += dash + gap
                if series_index is not None:
                    idx = int(series_index)
                    if 0 <= idx < len(vals):
                        x = int(round(xs[idx]))
                        y = int(round(ys[idx]))
                        cv2.line(bottom, (x, plot_top), (x, plot_bottom), (255, 0, 0), 1)
                        cv2.circle(bottom, (x, y), 3, (255, 0, 0), -1)
        out = np.concatenate([strip, out, bottom], axis=0)
    else:
        out = np.concatenate([strip, out], axis=0)

    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def render_event_video(
    *,
    ds,
    event_indices: Sequence[int],
    model: torch.nn.Module,
    device: torch.device,
    meta_map: Dict[str, Dict[str, float]],
    meta_by_name: Dict[str, Dict[str, float]],
    manifest_map: Dict[str, Dict[str, object]],
    selector,
    heatmap_stride: int,
    center_tau: float,
    roi_base_radius: int,
    roi_sigma_multiplier: float,
    presence_threshold: float,
    peak_pool: str,
    peak_tau: float,
    presence_topk: Optional[int],
    presence_from_peak: bool,
    batch_size: int,
    num_workers: int,
    render_workers: int,
    output_dir: Path,
    output_name: str,
    event_id: str,
    use_soft_argmax: bool,
) -> Tuple[Path, Optional[np.ndarray]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / f"{output_name}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    input_key = "video" if getattr(model, "input_is_video", False) else "image"

    class _IdxWrap(Dataset):
        def __init__(self, base_ds, indices):
            self.base_ds = base_ds
            self.indices = list(indices)
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, i):
            ds_idx = int(self.indices[i])
            s = self.base_ds[ds_idx]
            s["dataset_idx"] = ds_idx
            return s

    wrapped = _IdxWrap(ds, event_indices)

    loader_series = DataLoader(
        wrapped,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(num_workers)),
        pin_memory=(device.type == "cuda"),
    )

    peak_series: List[float] = []
    pred_prob_series: List[float] = []
    gt_prob_series: List[float] = []
    with torch.no_grad():
        for batch in loader_series:
            inp = batch[input_key].to(device, non_blocking=True)
            hm_logits_b, pres_logit_b = model(inp)
            hm_logits_b = hm_logits_b.squeeze(1).detach().float().cpu()
            pres_logit_b = pres_logit_b.squeeze(1).detach().float().cpu()

            peak_logit_pool = spatial_peak_pool(
                hm_logits_b,
                pool=peak_pool,
                tau=float(peak_tau),
                topk=presence_topk,
            )
            peak_logsumexp = spatial_peak_pool(
                hm_logits_b,
                pool="logsumexp",
                tau=float(peak_tau),
                topk=presence_topk,
            )

            if presence_from_peak:
                pred_prob = torch.sigmoid(peak_logit_pool).clamp(0.0, 1.0)
            else:
                pred_prob = torch.sigmoid(pres_logit_b).clamp(0.0, 1.0)

            peak_series.extend(peak_logsumexp.numpy().tolist())
            pred_prob_series.extend(pred_prob.numpy().tolist())
            gt_prob_series.extend(batch["presence"].squeeze(1).numpy().astype(np.float32).tolist())

    series_marks: List[Tuple[int, float]] = []
    gt_pos_idx = [i for i, v in enumerate(gt_prob_series) if float(v) > 0.0]
    if gt_pos_idx:
        start_idx = gt_pos_idx[0]
        end_idx = gt_pos_idx[-1]
        if 0 <= start_idx < len(peak_series):
            series_marks.append((start_idx, float(peak_series[start_idx])))
        if 0 <= end_idx < len(peak_series):
            series_marks.append((end_idx, float(peak_series[end_idx])))

    loader = DataLoader(
        wrapped,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(num_workers)),
        pin_memory=(device.type == "cuda"),
    )

    frame_counter = 0
    sample_counter = 0
    last_preview: Optional[np.ndarray] = None

    pool = ThreadPoolExecutor(max_workers=render_workers) if render_workers and int(render_workers) > 0 else None

    def _render_sample_frame(payload: Dict[str, object]) -> List[np.ndarray]:
        center_frame = payload["center_frame"]
        hm_tgt = payload["hm_tgt"]
        hm_pred_prob = payload["hm_pred_prob"]
        title = payload["title"]
        bottom_lines = payload["bottom_lines"]
        status_text = payload["status_text"]
        status_color = payload["status_color"]
        series_idx = payload["series_idx"]
        pred_lb = payload["pred_lb"]
        gt_lb = payload["gt_lb"]
        roundtrip_lb = payload["roundtrip_lb"]
        tx = payload["tx"]
        ty = payload["ty"]
        px = payload["px"]
        py = payload["py"]

        top_panel = render_event_panel(
            center_frame,
            hm_tgt,
            hm_pred_prob,
            title=title,
            gt_xy_hm=(float(tx), float(ty)),
            pr_xy_hm=(float(px), float(py)),
            pred_lb=pred_lb,
            gt_lb=gt_lb,
            roundtrip_lb=roundtrip_lb,
            heatmap_stride=heatmap_stride,
            hm_pred_vmax=1.0,
            bottom_lines=bottom_lines,
            status_text=status_text,
            status_color=status_color,
            series_vals=peak_series,
            series_index=series_idx,
            series_label="",
            series_marks=series_marks,
        )

        center_orig_path = payload["center_orig_path"]
        orig_rgb = _load_orig(center_orig_path)
        if orig_rgb is None:
            print("[WARN] Cannot load center original frame:", center_orig_path)
            return []

        orig_overlay = draw_markers_and_roi(
            orig_rgb,
            payload["pred_orig"],
            payload["gt_orig"],
            payload["roundtrip_orig"],
            payload["roi_radius"],
        )

        if top_panel.shape[1] != orig_rgb.shape[1]:
            tgt_w = orig_rgb.shape[1]
            pad_canvas = np.zeros((top_panel.shape[0], tgt_w, 3), dtype=np.uint8)
            if top_panel.shape[1] >= tgt_w:
                top_panel = cv2.resize(top_panel, (tgt_w, top_panel.shape[0]), interpolation=cv2.INTER_AREA)
            else:
                x0 = (tgt_w - top_panel.shape[1]) // 2
                pad_canvas[:, x0:x0 + top_panel.shape[1]] = top_panel
                top_panel = pad_canvas

        sep = np.zeros((8, orig_rgb.shape[1], 3), dtype=np.uint8)
        frame_rgb = np.concatenate([top_panel, sep, orig_overlay], axis=0)
        return [frame_rgb]

    try:
        for batch in loader:
            ds_idx_batch = batch["dataset_idx"].tolist()
            inp = batch[input_key].to(device, non_blocking=True)

            with torch.no_grad():
                hm_logits_b, _ = model(inp)

            hm_logits_b = hm_logits_b.squeeze(1).detach().float().cpu()
            hm_pred_prob_b = torch.sigmoid(hm_logits_b).numpy()
            hm_tgt_b = batch["heatmap"].squeeze(1).numpy()
            video_b = batch["video"]
            path_b = batch["image_path_abs"]

            tasks = []
            for bi, ds_idx in enumerate(ds_idx_batch):
                sample_path = str(path_b[bi])
                meta = _get_meta_for_path(meta_map, meta_by_name, sample_path)
                if meta is None:
                    print("[WARN] missing meta for", sample_path)
                    continue

                v = video_b[bi]
                t_total = int(v.shape[1])
                center_i = int(t_total // 2)

                hm_pred_prob = hm_pred_prob_b[bi]
                hm_tgt = hm_tgt_b[bi]
                hm_logits = hm_logits_b[bi]

                x_g, y_g = decode_heatmap(hm_logits.numpy(), stride=heatmap_stride, soft=use_soft_argmax, tau=center_tau)
                pred_lb = (float(x_g), float(y_g))
                pred_orig = _backproject_xy(float(x_g), float(y_g), meta)

                _, _, _, peak_width = peak_and_width(hm_logits.numpy())
                r_dynamic = roi_sigma_multiplier * peak_width * heatmap_stride if peak_width > 0 else 0
                roi_radius = max(roi_base_radius, int(round(r_dynamic)))

                gt_info = _get_gt_info(sample_path, meta, manifest_map)
                gt_orig = gt_info["gt_orig"]
                gt_lb = gt_info["gt_lb"]
                roundtrip_orig = gt_info["roundtrip_orig"]
                roundtrip_lb = gt_info["roundtrip_lb"]

                ty, tx = np.unravel_index(int(np.argmax(hm_tgt)), hm_tgt.shape)
                py, px = np.unravel_index(int(np.argmax(hm_pred_prob)), hm_pred_prob.shape)

                series_idx = sample_counter
                presence_gt = float(gt_prob_series[series_idx]) if series_idx < len(gt_prob_series) else float("nan")
                pred_prob = float(pred_prob_series[series_idx]) if series_idx < len(pred_prob_series) else float("nan")
                peak_logsumexp = float(peak_series[series_idx]) if series_idx < len(peak_series) else float("nan")

                gt_pos = presence_gt > 0.0 if np.isfinite(presence_gt) else False
                pred_pos = pred_prob >= float(presence_threshold) if np.isfinite(pred_prob) else False
                if gt_pos:
                    status_text = "detected" if pred_pos else "missed"
                    status_color = (0, 255, 0) if status_text == "detected" else (255, 0, 0)
                else:
                    status_text = None
                    status_color = (0, 255, 0)

                stride_real = float(v.shape[2]) / float(hm_pred_prob.shape[0])
                x_dsnt, y_dsnt = decode_heatmap(
                    hm_logits.numpy(),
                    stride=int(round(stride_real)),
                    soft=use_soft_argmax,
                    tau=float(center_tau),
                )

                ts_str = str(gt_info["timestamp"]) if gt_info["timestamp"] is not None else ""

                title = f"id={event_id} | idx={int(ds_idx)}"
                if ts_str:
                    title += f" | {ts_str}"

                bottom_lines = [
                    f"gt_p={presence_gt:.2f}  pred_p={pred_prob:.2f}",
                    f"peak_logsumexp_topk={peak_logsumexp:.3f}",
                ]

                center_frame = v[:, center_i].permute(1, 2, 0).float().cpu().numpy()
                center_frame = np.clip(center_frame, 0.0, 1.0)
                payload = {
                    "center_frame": center_frame,
                    "hm_tgt": hm_tgt,
                    "hm_pred_prob": hm_pred_prob,
                    "title": title,
                    "bottom_lines": bottom_lines,
                    "status_text": status_text,
                    "status_color": status_color,
                    "series_idx": series_idx,
                    "pred_lb": pred_lb,
                    "gt_lb": gt_lb,
                    "roundtrip_lb": roundtrip_lb,
                    "tx": float(tx),
                    "ty": float(ty),
                    "px": float(px),
                    "py": float(py),
                    "center_orig_path": meta.get("orig_path"),
                    "pred_orig": pred_orig,
                    "gt_orig": gt_orig,
                    "roundtrip_orig": roundtrip_orig,
                    "roi_radius": roi_radius,
                }

                if pool is None:
                    tasks.append(_render_sample_frame(payload))
                else:
                    tasks.append(pool.submit(_render_sample_frame, payload))

                sample_counter += 1

            for item in tasks:
                frames = item if pool is None else item.result()
                for frame_rgb in frames:
                    out_png = frames_dir / f"frame_{frame_counter:05d}.png"
                    cv2.imwrite(str(out_png), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                    frame_counter += 1
                    last_preview = frame_rgb
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    return frames_dir, last_preview
