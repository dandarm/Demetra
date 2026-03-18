#!/home/isac/miniconda3/envs/videomae/bin/python
"""Download Airmass RGB frames for a date range and run first-pass + tracking."""
from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from contextlib import ExitStack
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from threading import local
from typing import Dict, Iterable, List, Tuple

import eumdac
import gcsfs
import imageio.v2 as imageio
import ocf_blosc2  # noqa: F401 - required to decode the public zarr dataset
import pandas as pd
import requests
import xarray as xr
from dask.distributed import Client, LocalCluster


REPO_ROOT = Path(__file__).resolve().parents[1]
AIRMAss_OUTPUT_ROOT_DEFAULT = Path("/media/isacDisk2/demetra_output")
PYTHON_EXEC_DEFAULT = Path("/home/isac/miniconda3/envs/videomae/bin/python")
PUBLIC_BUCKET_BASE = (
    "public-datasets-eumetsat-solar-forecasting/satellite/EUMETSAT/SEVIRI_RSS/v4"
)
EUMETSAT_COLLECTION_ID_DEFAULT = "EO:EUM:DAT:MSG:MSG15-RSS"
FRAME_RE = re.compile(r"airmass_rgb_(\d{8}_\d{4})\.png$")

FIRSTPASS_MODEL_DEFAULT = REPO_ROOT / "trained_models" / "firstpass_model.ckpt"
TRACKING_MODEL_DEFAULT = Path("/media/isacDisk2/demetra_trained_models/checkpoint-tracking-best_1.pth")
MANOS_FILE_DEFAULT = (
    REPO_ROOT / "moduli" / "videomae" / "medicane_data_input" / "medicanes_new_windows.csv"
)
INFERENCE_SCRIPT = REPO_ROOT / "scripts" / "predict_firstpass_and_track_from_folder.py"

MEDICANE_UTILS_DIR = REPO_ROOT / "moduli" / "videomae" / "medicane_utils"
if str(MEDICANE_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(MEDICANE_UTILS_DIR))

from download_airmassRGB import (  # noqa: E402
    create_rgb_array,
    inverse_rescale_bulk,
    lat_max,
    lat_min,
    lon_max,
    lon_min,
    spatial_cut_geos,
    to_8bit_airmass,
)
from create_airmassRGB_from_hrseviri_local import (  # noqa: E402
    parse_dt_from_zip_name,
    process_one_zip,
    slot_dt_from_zip_name,
)


LOG = logging.getLogger("download_track_range")
_THREAD_LOCAL = local()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scarica i frame Airmass RGB da SEVIRI RSS e lancia la pipeline "
            "first-pass + tracking sul range temporale richiesto."
        )
    )
    parser.add_argument("--start", required=True, help="Data/ora iniziale.")
    parser.add_argument("--end", required=True, help="Data/ora finale.")
    parser.add_argument(
        "--download_source",
        choices=["auto", "public", "eumetsat"],
        default="auto",
        help="Sorgente download: pubblico GCS, EUMETSAT diretto o auto.",
    )
    parser.add_argument(
        "--output_root",
        default=str(AIRMAss_OUTPUT_ROOT_DEFAULT),
        help="Root dove creare la cartella di run (default: /media/isacDisk2/demetra_output).",
    )
    parser.add_argument(
        "--eumetsat_collection",
        default=EUMETSAT_COLLECTION_ID_DEFAULT,
        help="Collection ID EUMETSAT da usare per il download diretto.",
    )
    parser.add_argument(
        "--firstpass_model_path",
        default=str(FIRSTPASS_MODEL_DEFAULT),
        help="Checkpoint first-pass.",
    )
    parser.add_argument(
        "--tracking_model_path",
        default=str(TRACKING_MODEL_DEFAULT),
        help="Checkpoint tracking.",
    )
    parser.add_argument(
        "--manos_file",
        default=str(MANOS_FILE_DEFAULT),
        help="CSV opzionale con GT.",
    )
    parser.add_argument(
        "--python_exec",
        default=str(PYTHON_EXEC_DEFAULT),
        help="Interprete Python da usare per la pipeline di inferenza.",
    )
    parser.add_argument(
        "--dask_workers",
        type=int,
        default=16,
        help="Numero di worker Dask per il download/preprocessing.",
    )
    parser.add_argument(
        "--skip_inference",
        action="store_true",
        help="Scarica solo i frame e non lancia il tracking.",
    )
    parser.add_argument(
        "--eumetsat_download_workers",
        type=int,
        default=4,
        help="Numero di download concorrenti EUMETSAT.",
    )
    parser.add_argument(
        "--eumetsat_download_retries",
        type=int,
        default=3,
        help="Numero massimo di retry per ciascun prodotto EUMETSAT.",
    )
    parser.add_argument(
        "--eumetsat_read_timeout",
        type=int,
        default=180,
        help="Read timeout in secondi per ciascun stream EUMETSAT.",
    )
    return parser.parse_args()


def setup_logging(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="a"),
        ],
        force=True,
    )


def _looks_like_date_only(value: str) -> bool:
    value = value.strip()
    return bool(re.fullmatch(r"\d{2}-\d{2}-\d{4}", value) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", value))


def parse_user_datetime(value: str, *, is_end: bool) -> pd.Timestamp:
    raw = value.strip()
    explicit_formats = (
        "%d-%m-%Y %H:%M",
        "%d-%m-%Y_%H:%M",
        "%d-%m-%Y %H%M",
        "%d-%m-%Y_%H%M",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d_%H:%M",
        "%Y-%m-%d %H%M",
        "%Y-%m-%d_%H%M",
        "%Y%m%d_%H%M",
        "%Y%m%d%H%M",
        "%d-%m-%Y",
        "%Y-%m-%d",
    )
    for fmt in explicit_formats:
        try:
            ts = pd.Timestamp(datetime.strptime(raw, fmt))
            if _looks_like_date_only(raw):
                return ts.replace(hour=23, minute=55) if is_end else ts.replace(hour=0, minute=0)
            return ts
        except ValueError:
            continue

    ts_dayfirst = pd.to_datetime(raw, errors="coerce", dayfirst=True)
    ts_default = pd.to_datetime(raw, errors="coerce", dayfirst=False)
    ts = ts_dayfirst if pd.notna(ts_dayfirst) else ts_default
    if pd.isna(ts):
        raise ValueError(f"Formato data non riconosciuto: {value}")
    ts = pd.Timestamp(ts)
    if _looks_like_date_only(raw):
        return ts.replace(hour=23, minute=55) if is_end else ts.replace(hour=0, minute=0)
    return ts


def format_compact(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%Y%m%d_%H%M")


def collect_existing_frame_map(frames_dir: Path) -> Dict[pd.Timestamp, Path]:
    frame_map: Dict[pd.Timestamp, Path] = {}
    if not frames_dir.exists():
        return frame_map
    for path in sorted(frames_dir.glob("airmass_rgb_*.png")):
        match = FRAME_RE.match(path.name)
        if not match:
            continue
        try:
            ts = pd.Timestamp(datetime.strptime(match.group(1), "%Y%m%d_%H%M"))
        except ValueError:
            continue
        frame_map[ts] = path
    return frame_map


def write_availability_report(
    run_dir: Path,
    requested_start: pd.Timestamp,
    requested_end: pd.Timestamp,
    latest_available: pd.Timestamp,
    latest_year: int,
) -> Path:
    report_path = run_dir / "availability_report.txt"
    text = (
        f"Requested range: {requested_start} -> {requested_end}\n"
        f"Latest available public SEVIRI RSS timestamp: {latest_available} UTC\n"
        f"Latest available yearly store: {latest_year}_nonhrv.zarr\n"
        "No frames overlap the requested interval.\n"
    )
    report_path.write_text(text, encoding="ascii")
    return report_path


def build_run_paths(output_root: Path, requested_start: pd.Timestamp, requested_end: pd.Timestamp) -> Tuple[Path, Path]:
    run_name = f"range_{format_compact(requested_start)}__{format_compact(requested_end)}"
    run_dir = output_root / run_name
    frames_dir = run_dir / "frames"
    return run_dir, frames_dir


def has_eumdac_credentials() -> bool:
    return bool(os.getenv("EUMETSAT_CONSUMER_KEY") and os.getenv("EUMETSAT_CONSUMER_SECRET"))


def _get_thread_datastore(consumer_key: str, consumer_secret: str) -> eumdac.DataStore:
    datastore = getattr(_THREAD_LOCAL, "datastore", None)
    credentials = getattr(_THREAD_LOCAL, "credentials", None)
    current_credentials = (consumer_key, consumer_secret)
    if datastore is not None and credentials == current_credentials:
        return datastore

    token = eumdac.AccessToken(current_credentials)
    datastore = eumdac.DataStore(token)
    _THREAD_LOCAL.datastore = datastore
    _THREAD_LOCAL.credentials = current_credentials
    return datastore


def is_valid_eumetsat_zip(zip_path: Path) -> bool:
    if not zip_path.exists() or zip_path.stat().st_size <= 0:
        return False
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            return any(name.lower().endswith(".nat") for name in zf.namelist())
    except (zipfile.BadZipFile, OSError):
        return False


@lru_cache(maxsize=1)
def fs_client() -> gcsfs.GCSFileSystem:
    return gcsfs.GCSFileSystem(token="anon", max_pool_connections=20)


@lru_cache(maxsize=1)
def list_available_nonhrv_years() -> Dict[int, str]:
    entries = fs_client().ls(PUBLIC_BUCKET_BASE, detail=False)
    year_map: Dict[int, str] = {}
    for entry in entries:
        name = Path(entry).name
        match = re.fullmatch(r"(\d{4})_nonhrv\.zarr", name)
        if match:
            year = int(match.group(1))
            year_map[year] = f"gs://{entry}"
    if not year_map:
        raise RuntimeError(f"Nessun dataset annuale *_nonhrv.zarr trovato in {PUBLIC_BUCKET_BASE}")
    return dict(sorted(year_map.items()))


@lru_cache(maxsize=8)
def open_year_dataset(year: int) -> xr.Dataset:
    year_map = list_available_nonhrv_years()
    if year not in year_map:
        raise KeyError(year)
    mapper = fs_client().get_mapper(year_map[year])
    return xr.open_zarr(mapper, consolidated=True)


def dataset_time_bounds(ds: xr.Dataset) -> Tuple[pd.Timestamp, pd.Timestamp]:
    times = ds["time"]
    return pd.Timestamp(times.isel(time=0).values), pd.Timestamp(times.isel(time=-1).values)


def latest_public_timestamp() -> Tuple[int, pd.Timestamp]:
    year_map = list_available_nonhrv_years()
    latest_year = max(year_map)
    latest_ts = dataset_time_bounds(open_year_dataset(latest_year))[1]
    return latest_year, latest_ts


def requested_segments(
    requested_start: pd.Timestamp,
    requested_end: pd.Timestamp,
) -> List[Tuple[int, pd.Timestamp, pd.Timestamp]]:
    segments: List[Tuple[int, pd.Timestamp, pd.Timestamp]] = []
    for year in range(requested_start.year, requested_end.year + 1):
        try:
            ds = open_year_dataset(year)
        except KeyError:
            LOG.info("Dataset annuale %s_nonhrv.zarr non disponibile nel bucket pubblico.", year)
            continue
        year_start, year_end = dataset_time_bounds(ds)
        seg_start = max(requested_start, year_start)
        seg_end = min(requested_end, year_end)
        if seg_start <= seg_end:
            segments.append((year, seg_start, seg_end))
    return segments


def expected_times_for_segments(
    segments: Iterable[Tuple[int, pd.Timestamp, pd.Timestamp]]
) -> List[pd.Timestamp]:
    out: List[pd.Timestamp] = []
    for year, seg_start, seg_end in segments:
        ds = open_year_dataset(year)
        times = pd.to_datetime(ds["time"].sel(time=slice(seg_start, seg_end)).values)
        out.extend(pd.Timestamp(t) for t in times)
    return sorted(set(out))


def download_frames_for_segments(
    segments: Iterable[Tuple[int, pd.Timestamp, pd.Timestamp]],
    frames_dir: Path,
) -> List[Path]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    created: List[Path] = []

    for year, seg_start, seg_end in segments:
        ds = open_year_dataset(year)
        data_array = ds["data"].sortby("time")
        x_increasing = bool(data_array.x_geostationary[0] < data_array.x_geostationary[-1])
        y_increasing = bool(data_array.y_geostationary[0] < data_array.y_geostationary[-1])

        ds_sub = data_array.sel(
            time=slice(seg_start, seg_end),
            variable=["IR_097", "IR_108", "WV_062", "WV_073"],
        )
        if ds_sub.sizes.get("time", 0) == 0:
            LOG.info("Nessun frame disponibile nel segmento %s -> %s", seg_start, seg_end)
            continue

        ds_sub_cut = spatial_cut_geos(
            ds_sub,
            lat_min,
            lat_max,
            lon_min,
            lon_max,
            x_increasing=x_increasing,
            y_increasing=y_increasing,
            flip_north_up=True,
        )
        ds_tb = inverse_rescale_bulk(ds_sub_cut)
        ds_rgb = create_rgb_array(ds_tb)

        day_values = pd.to_datetime(ds_rgb.time.values).normalize().unique()
        for day in day_values:
            day_str = pd.Timestamp(day).strftime("%Y-%m-%d")
            ds_day = ds_rgb.sel(time=day_str)
            if ds_day.sizes.get("time", 0) == 0:
                LOG.info("Nessun time-step nel giorno %s, skip", day_str)
                continue
            LOG.info("Compute giorno %s (%s frame)...", day_str, ds_day.sizes["time"])
            ds_day_loaded = ds_day.compute()
            for i in range(ds_day_loaded.sizes["time"]):
                da_one = ds_day_loaded.isel(time=i)
                arr_8bit = to_8bit_airmass(da_one.values)
                time_val = pd.Timestamp(ds_day_loaded.time.isel(time=i).values)
                out_png = frames_dir / f"airmass_rgb_{time_val.strftime('%Y%m%d_%H%M')}.png"
                if out_png.exists() and out_png.stat().st_size > 0:
                    continue
                imageio.imwrite(out_png, arr_8bit)
                created.append(out_png)
            LOG.info("Completato giorno %s", day_str)
    return created


def verify_downloaded_frames(
    frames_dir: Path,
    expected_times: Iterable[pd.Timestamp],
) -> List[Path]:
    frame_map = collect_existing_frame_map(frames_dir)
    missing = []
    for ts in expected_times:
        if ts not in frame_map:
            missing.append(frames_dir / f"airmass_rgb_{pd.Timestamp(ts).strftime('%Y%m%d_%H%M')}.png")
    return missing


def public_segments_cover_request(
    requested_start: pd.Timestamp,
    requested_end: pd.Timestamp,
    segments: Iterable[Tuple[int, pd.Timestamp, pd.Timestamp]],
) -> bool:
    segs = list(segments)
    if not segs:
        return False
    return segs[0][1] <= requested_start and segs[-1][2] >= requested_end


def choose_download_source(
    mode: str,
    requested_start: pd.Timestamp,
    requested_end: pd.Timestamp,
    segments: Iterable[Tuple[int, pd.Timestamp, pd.Timestamp]],
) -> str:
    segs = list(segments)
    if mode == "public":
        return "public"
    if mode == "eumetsat":
        return "eumetsat"
    if public_segments_cover_request(requested_start, requested_end, segs):
        return "public"
    if has_eumdac_credentials():
        return "eumetsat"
    return "public"


def download_products_from_eumetsat(
    requested_start: pd.Timestamp,
    requested_end: pd.Timestamp,
    raw_dir: Path,
    collection_id: str,
    download_workers: int,
    max_retries: int,
    read_timeout: int,
) -> List[Path]:
    consumer_key = os.getenv("EUMETSAT_CONSUMER_KEY")
    consumer_secret = os.getenv("EUMETSAT_CONSUMER_SECRET")
    if not consumer_key or not consumer_secret:
        raise RuntimeError(
            "Credenziali EUMETSAT mancanti. Imposta EUMETSAT_CONSUMER_KEY e "
            "EUMETSAT_CONSUMER_SECRET."
        )

    raw_dir.mkdir(parents=True, exist_ok=True)
    token = eumdac.AccessToken((consumer_key, consumer_secret))
    datastore = eumdac.DataStore(token)
    collection = datastore.get_collection(collection_id)

    products = list(
        collection.search(
            dtstart=requested_start.to_pydatetime(),
            dtend=requested_end.to_pydatetime(),
        )
    )
    product_ids = [str(p) for p in products]
    LOG.info(
        "Prodotti EUMETSAT trovati in %s per %s -> %s: %d",
        collection_id,
        requested_start,
        requested_end,
        len(product_ids),
    )
    if not product_ids:
        return []

    cached_paths: List[Path] = []
    pending_product_ids: List[str] = []
    for product_id in product_ids:
        out_path = raw_dir / f"{product_id}.zip"
        if is_valid_eumetsat_zip(out_path):
            cached_paths.append(out_path)
            LOG.debug("Cache hit %s", out_path.name)
            continue
        if out_path.exists():
            LOG.warning("ZIP locale non valido, riscarico %s", out_path.name)
            out_path.unlink()
        pending_product_ids.append(product_id)

    LOG.info(
        "ZIP EUMETSAT gia validi in cache: %d | da scaricare: %d",
        len(cached_paths),
        len(pending_product_ids),
    )
    if not pending_product_ids:
        return sorted(
            cached_paths,
            key=lambda p: (parse_dt_from_zip_name(p.name) or datetime.min, p.name),
        )

    def _download_one(product_id: str) -> Path:
        out_path = raw_dir / f"{product_id}.zip"
        tmp_path = out_path.with_suffix(".zip.part")
        last_error: Exception | None = None
        for attempt in range(1, max_retries + 2):
            try:
                datastore_local = _get_thread_datastore(consumer_key, consumer_secret)
                product = datastore_local.get_product(
                    product_id=product_id,
                    collection_id=collection_id,
                )
                url = product.datastore.urls.get(
                    "datastore",
                    "download product",
                    vars={
                        "collection_id": collection_id,
                        "product_id": product_id,
                    },
                )
                headers = eumdac.common.headers.copy()
                LOG.info(
                    "Downloading %s [attempt %d/%d]",
                    out_path.name,
                    attempt,
                    max_retries + 1,
                )
                with requests.get(
                    url,
                    auth=product.datastore.token.auth,
                    stream=True,
                    headers=headers,
                    timeout=(30, read_timeout),
                ) as response:
                    response.raise_for_status()
                    expected_bytes = response.headers.get("Content-Length")
                    with tmp_path.open("wb") as fdst:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                fdst.write(chunk)
                if expected_bytes is not None and tmp_path.stat().st_size != int(expected_bytes):
                    raise RuntimeError(
                        f"Download incompleto per {out_path.name}: "
                        f"{tmp_path.stat().st_size} != {expected_bytes} byte"
                    )
                tmp_path.replace(out_path)
                if not is_valid_eumetsat_zip(out_path):
                    raise RuntimeError(f"ZIP corrotto o privo di .nat: {out_path.name}")
                LOG.info("Done %s (%.1f MB)", out_path.name, out_path.stat().st_size / 1e6)
                return out_path
            except Exception as exc:
                last_error = exc
                if tmp_path.exists():
                    tmp_path.unlink()
                if attempt > max_retries:
                    break
                sleep_s = min(30, 2 ** (attempt - 1))
                LOG.warning(
                    "Retry %s after error on %s: %s",
                    attempt,
                    out_path.name,
                    exc,
                )
                time.sleep(sleep_s)

        raise RuntimeError(f"Download fallito per {product_id}: {last_error}")

    zip_paths: List[Path] = list(cached_paths)
    workers = max(1, int(download_workers))
    LOG.info(
        "Avvio download concorrente EUMETSAT con %d worker, %d retry max, read timeout %ss",
        workers,
        max_retries,
        read_timeout,
    )
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="eumdac",
    ) as executor:
        future_to_idx = {
            executor.submit(_download_one, product_id): (idx, product_id)
            for idx, product_id in enumerate(pending_product_ids, start=1)
        }
        for future in concurrent.futures.as_completed(future_to_idx):
            idx, product_id = future_to_idx[future]
            out_path = future.result()
            LOG.info("[%d/%d] Completed %s", idx, len(pending_product_ids), product_id)
            zip_paths.append(out_path)

    return sorted(
        zip_paths,
        key=lambda p: (parse_dt_from_zip_name(p.name) or datetime.min, p.name),
    )


def expected_times_for_zip_files(zip_paths: Iterable[Path]) -> List[pd.Timestamp]:
    out: List[pd.Timestamp] = []
    for path in zip_paths:
        ts = parse_dt_from_zip_name(path.name)
        if ts is None:
            continue
        out.append(pd.Timestamp(ts).floor("5min"))
    return sorted(set(out))


def convert_eumetsat_zips_to_frames(zip_paths: Iterable[Path], frames_dir: Path) -> List[Path]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    zip_list = list(zip_paths)
    created: List[Path] = []
    if not zip_list:
        return created

    cached_pngs: List[Path] = []
    pending_zips: List[Path] = []
    for zip_path in zip_list:
        t_hint = slot_dt_from_zip_name(zip_path.name)
        if t_hint is not None:
            out_hint = frames_dir / f"airmass_rgb_{t_hint.strftime('%Y%m%d_%H%M')}.png"
            if out_hint.exists() and out_hint.stat().st_size > 0:
                cached_pngs.append(out_hint)
                continue
        pending_zips.append(zip_path)

    LOG.info(
        "PNG Airmass RGB gia presenti: %d | ZIP da convertire: %d",
        len(cached_pngs),
        len(pending_zips),
    )
    if not pending_zips:
        return cached_pngs

    for i, zip_path in enumerate(pending_zips, start=1):
        LOG.info("[%d/%d] Convert %s", i, len(pending_zips), zip_path.name)
        if not is_valid_eumetsat_zip(zip_path):
            if zip_path.exists():
                zip_path.unlink()
            raise RuntimeError(
                f"ZIP non valido rilevato prima della conversione: {zip_path}. "
                "Rilancia la run: i file PNG gia prodotti verranno riusati e il file corrotto verra riscaricato."
            )
        out_path = process_one_zip(zip_path, output_dir=frames_dir)
        created.append(out_path)
        LOG.info("  -> PNG: %s", out_path.name)
    return cached_pngs + created


def run_inference_pipeline(
    python_exec: str,
    frames_dir: Path,
    run_dir: Path,
    firstpass_model_path: Path,
    tracking_model_path: Path,
    manos_file: Path,
    video_name: str,
) -> None:
    cmd = [
        python_exec,
        str(INFERENCE_SCRIPT),
        "--input_dir",
        str(frames_dir),
        "--output_dir",
        str(run_dir),
        "--firstpass_model_path",
        str(firstpass_model_path),
        "--tracking_model_path",
        str(tracking_model_path),
        "--manos_file",
        str(manos_file),
        "--make_video",
        "--video_name",
        video_name,
    ]
    LOG.info("Lancio inferenza: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def main() -> int:
    args = parse_args()

    requested_start = parse_user_datetime(args.start, is_end=False)
    requested_end = parse_user_datetime(args.end, is_end=True)
    if requested_end < requested_start:
        raise ValueError("`end` deve essere >= `start`.")

    output_root = Path(args.output_root).expanduser().resolve()
    run_dir, frames_dir = build_run_paths(output_root, requested_start, requested_end)
    setup_logging(run_dir)

    LOG.info("Repo root: %s", REPO_ROOT)
    LOG.info("Range richiesto: %s -> %s", requested_start, requested_end)
    LOG.info("Cartella run: %s", run_dir)

    firstpass_model_path = Path(args.firstpass_model_path).expanduser().resolve()
    tracking_model_path = Path(args.tracking_model_path).expanduser().resolve()
    manos_file = Path(args.manos_file).expanduser().resolve()

    for path in (firstpass_model_path, tracking_model_path, INFERENCE_SCRIPT):
        if not path.exists():
            raise FileNotFoundError(f"Path richiesto non trovato: {path}")
    if not manos_file.exists():
        LOG.warning("manos_file non trovato: %s", manos_file)

    year_map = list_available_nonhrv_years()
    latest_year, latest_ts = latest_public_timestamp()
    LOG.info(
        "Dataset annuali disponibili: %s-%s | ultimo timestamp pubblico: %s UTC",
        min(year_map),
        max(year_map),
        latest_ts,
    )

    segments = requested_segments(requested_start, requested_end)
    source = choose_download_source(
        mode=args.download_source,
        requested_start=requested_start,
        requested_end=requested_end,
        segments=segments,
    )
    LOG.info("Sorgente download selezionata: %s", source)

    expected_times: List[pd.Timestamp]
    if source == "public":
        if not segments:
            report = write_availability_report(
                run_dir=run_dir,
                requested_start=requested_start,
                requested_end=requested_end,
                latest_available=latest_ts,
                latest_year=latest_year,
            )
            LOG.info(
                "Nessun frame disponibile nel range pubblico richiesto. Ultimo timestamp pubblico: %s UTC. "
                "Report: %s",
                latest_ts,
                report,
            )
            return 2

        effective_start = segments[0][1]
        effective_end = segments[-1][2]
        if effective_start != requested_start or effective_end != requested_end:
            LOG.info(
                "Range effettivo disponibile nel dataset pubblico: %s -> %s (clippato rispetto alla richiesta)",
                effective_start,
                effective_end,
            )

        expected_times = expected_times_for_segments(segments)
        existing_map = collect_existing_frame_map(frames_dir)
        missing_before = [ts for ts in expected_times if ts not in existing_map]
        LOG.info(
            "Frame attesi nel range disponibile: %d | gia presenti: %d | da generare: %d",
            len(expected_times),
            len(expected_times) - len(missing_before),
            len(missing_before),
        )

        with ExitStack() as stack:
            if missing_before:
                workers = max(1, int(args.dask_workers))
                cluster = LocalCluster(n_workers=workers, threads_per_worker=1)
                client = Client(cluster)
                stack.callback(client.close)
                stack.callback(cluster.close)
                LOG.info("Dask cluster attivo: %s", client)
                download_frames_for_segments(segments, frames_dir)
    else:
        raw_dir = run_dir / "raw_eumetsat"
        zip_paths = download_products_from_eumetsat(
            requested_start=requested_start,
            requested_end=requested_end,
            raw_dir=raw_dir,
            collection_id=args.eumetsat_collection,
            download_workers=args.eumetsat_download_workers,
            max_retries=args.eumetsat_download_retries,
            read_timeout=args.eumetsat_read_timeout,
        )
        if not zip_paths:
            raise RuntimeError(
                f"Nessun prodotto EUMETSAT trovato in {args.eumetsat_collection} per "
                f"{requested_start} -> {requested_end}"
            )

        expected_times = expected_times_for_zip_files(zip_paths)
        existing_map = collect_existing_frame_map(frames_dir)
        missing_before = [ts for ts in expected_times if ts not in existing_map]
        LOG.info(
            "Frame attesi da ZIP EUMETSAT: %d | gia presenti: %d | da generare: %d",
            len(expected_times),
            len(expected_times) - len(missing_before),
            len(missing_before),
        )
        if missing_before:
            convert_eumetsat_zips_to_frames(zip_paths, frames_dir)

    missing_after = verify_downloaded_frames(frames_dir, expected_times)
    if missing_after:
        missing_preview = ", ".join(p.name for p in missing_after[:5])
        raise RuntimeError(
            f"Download incompleto: mancano {len(missing_after)} frame. Esempi: {missing_preview}"
        )

    frame_map = collect_existing_frame_map(frames_dir)
    frame_times = sorted(frame_map)
    if not frame_times:
        raise RuntimeError(f"Nessun frame valido scaricato in {frames_dir}")
    LOG.info(
        "Frame pronti per inferenza: %d (%s -> %s)",
        len(frame_times),
        frame_times[0],
        frame_times[-1],
    )

    if args.skip_inference:
        LOG.info("skip_inference attivo: fermo dopo il download.")
        return 0

    video_name = run_dir.name
    run_inference_pipeline(
        python_exec=args.python_exec,
        frames_dir=frames_dir,
        run_dir=run_dir,
        firstpass_model_path=firstpass_model_path,
        tracking_model_path=tracking_model_path,
        manos_file=manos_file,
        video_name=video_name,
    )

    final_csv = run_dir / "tracking_inference_predictions.csv"
    final_mp4 = run_dir / f"{video_name}.mp4"
    LOG.info("Output CSV: %s", final_csv)
    LOG.info("Output video: %s", final_mp4)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
