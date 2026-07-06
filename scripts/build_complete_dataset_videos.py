#!/home/isac/miniconda3/envs/videomae/bin/python
from __future__ import annotations

import argparse
import csv
import html
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from predict_firstpass_and_track_from_folder import _overlay_coastlines


FRAME_RE = re.compile(r"airmass_rgb_(\d{8}_\d{4})\.png$")
RENDER_MODE = "gt_marker_green_multi_cl3_coast_date_v3"
TIMESTAMP_FONT = Path("/media/isacDisk1/Demetra/moduli/videomae/digital-7 (italic).ttf")

SITE_CSS = """
:root {
  --bg: #f4efe7;
  --panel: #fffaf2;
  --panel-border: #d7c8b6;
  --text: #1f2933;
  --muted: #6b7280;
  --accent: #b25b35;
  --accent-dark: #7f3d22;
  --success-bg: #dff3dd;
  --success-text: #1d4f2a;
  --warn-bg: #fbe4b8;
  --warn-text: #7e4f00;
  --danger-bg: #f6d3cf;
  --danger-text: #7c261f;
  --shadow: 0 14px 32px rgba(73, 51, 33, 0.12);
}

* { box-sizing: border-box; }

body {
  margin: 0;
  color: var(--text);
  background:
    radial-gradient(circle at top left, rgba(178, 91, 53, 0.15), transparent 30%),
    linear-gradient(180deg, #f7f1e8 0%, #efe5d7 100%);
  font-family: "Fira Sans", "Gill Sans", "Trebuchet MS", sans-serif;
}

a { color: var(--accent-dark); text-decoration: none; }
a:hover { text-decoration: underline; }
code, pre { font-family: "IBM Plex Mono", "DejaVu Sans Mono", Consolas, monospace; }

.shell {
  width: min(1380px, calc(100% - 2rem));
  margin: 0 auto;
}

.hero {
  padding: 2.5rem 0 1.75rem;
  border-bottom: 1px solid rgba(127, 61, 34, 0.12);
  background: linear-gradient(180deg, rgba(255, 251, 245, 0.82), rgba(255, 247, 237, 0.55));
}

.hero h1 {
  margin: 0.15rem 0 0.35rem;
  font-size: clamp(2rem, 3vw, 3rem);
  letter-spacing: -0.04em;
}

.eyebrow,
.kicker {
  margin: 0;
  color: var(--accent-dark);
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-size: 0.75rem;
  font-weight: 700;
}

.subtitle,
.panel-note,
.empty-state,
.small,
.meta-list {
  color: var(--muted);
}

main.shell {
  padding: 1.5rem 0 2rem;
}

.panel {
  margin-bottom: 1.25rem;
  padding: 1.25rem;
  border: 1px solid var(--panel-border);
  border-radius: 20px;
  background: rgba(255, 250, 242, 0.92);
  box-shadow: var(--shadow);
}

.panel-head {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  align-items: flex-start;
  margin-bottom: 1rem;
}

.panel-head h2 {
  margin: 0.15rem 0 0;
  font-size: 1.35rem;
}

.stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 1rem;
}

.stat {
  padding: 1rem;
  border-radius: 16px;
  background: rgba(255, 255, 255, 0.62);
  border: 1px solid rgba(125, 99, 74, 0.15);
}

.stat .value {
  font-size: 2rem;
  font-weight: 700;
  color: var(--accent-dark);
}

.badge {
  display: inline-flex;
  align-items: center;
  border-radius: 999px;
  padding: 0.28rem 0.7rem;
  font-size: 0.78rem;
  font-weight: 700;
  border: 1px solid transparent;
}

.badge-complete,
.badge-generated_video,
.badge-existing_video,
.badge-track_overlay {
  background: var(--success-bg);
  color: var(--success-text);
  border-color: rgba(29, 79, 42, 0.2);
}

.badge-partial,
.badge-partial_track_overlay,
.badge-no_manos_id {
  background: var(--warn-bg);
  color: var(--warn-text);
  border-color: rgba(126, 79, 0, 0.18);
}

.badge-missing,
.badge-no_frames,
.badge-no_track_points {
  background: var(--danger-bg);
  color: var(--danger-text);
  border-color: rgba(124, 38, 31, 0.18);
}

.table-wrap {
  overflow-x: auto;
  border-radius: 16px;
  border: 1px solid rgba(125, 99, 74, 0.15);
}

.listing {
  width: 100%;
  border-collapse: collapse;
}

.listing th,
.listing td {
  padding: 0.8rem 0.7rem;
  border-bottom: 1px solid rgba(125, 99, 74, 0.15);
  text-align: left;
  vertical-align: top;
}

.listing th {
  color: var(--accent-dark);
  font-size: 0.9rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  background: #fffaf2;
}

.back-link {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  padding: 0.9rem 1.15rem;
  border-radius: 999px;
  background: linear-gradient(135deg, var(--accent), #d17a50);
  color: #fff7f0;
  font-weight: 700;
  box-shadow: 0 10px 24px rgba(178, 91, 53, 0.24);
}

.video-shell {
  background: rgba(34, 26, 21, 0.9);
  border-radius: 18px;
  padding: 0.9rem;
}

video {
  width: 100%;
  border-radius: 12px;
  display: block;
}

.meta-list {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 0.75rem 1rem;
}

.meta-item {
  padding: 0.85rem 0.9rem;
  background: rgba(255, 255, 255, 0.55);
  border-radius: 14px;
  border: 1px solid rgba(125, 99, 74, 0.15);
}

.meta-item strong {
  display: block;
  color: var(--text);
  margin-bottom: 0.2rem;
}

.note-box {
  padding: 1rem;
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.58);
  border: 1px solid rgba(125, 99, 74, 0.15);
  line-height: 1.5;
}

@media (max-width: 860px) {
  .panel-head {
    display: block;
  }
}
""".strip()


@dataclass(frozen=True)
class EventRow:
    raw_interval: str
    name: str
    id_manos_cl7: str
    presente_totale: str
    presente_parziale: str
    note: str
    start: datetime
    end: datetime


@dataclass(frozen=True)
class EventFrame:
    timestamp: datetime
    path: Path


@dataclass(frozen=True)
class TrackPoint:
    timestamp: datetime
    x: float
    y: float
    cyclone_id: str
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Genera un MP4 per ogni evento del CSV full_list_medicanes_dataset_presence, "
            "usando i frame gia presenti in source_dataset, con overlay opzionale della "
            "ground truth Manos da manos_CL7_pixel.csv. Genera anche un sito statico."
        )
    )
    parser.add_argument(
        "--presence_csv",
        default="/media/isacDisk1/Demetra/notebooks/full_list_medicanes_dataset_presence.csv",
        help="CSV degli eventi con presenza dati.",
    )
    parser.add_argument(
        "--dataset_root",
        default="/media/isacDisk2/source_dataset",
        help="Root del dataset immagini YYYY/MM/airmass_rgb_*.png.",
    )
    parser.add_argument(
        "--output_root",
        default="/media/isacDisk2/complete_dataset",
        help="Cartella dove salvare i video evento-per-evento.",
    )
    parser.add_argument(
        "--manos_pixel_csv",
        default="/media/isacDisk1/Demetra/notebooks/manos_CL7_pixel.csv",
        help="CSV Manos con x_pix/y_pix del centro ciclone.",
    )
    parser.add_argument(
        "--context_manos_pixel_csv",
        default="/media/isacDisk1/Demetra/moduli/videomae/medicane_data_input/all_manos_CL_pixel.csv",
        help="CSV Manos cumulativo da cui leggere tracce di contesto, ad esempio CL3.",
    )
    parser.add_argument(
        "--context_sources",
        default="CL3",
        help="Sorgenti Manos da aggiungere come contesto, separate da virgola. Default: CL3.",
    )
    parser.add_argument(
        "--site_dir",
        default="",
        help="Cartella del sito statico. Default: <output_root>/site",
    )
    parser.add_argument(
        "--ffmpeg_path",
        default="",
        help="Path esplicito a ffmpeg o alla sua cartella.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="FPS del video finale.",
    )
    parser.add_argument(
        "--min_year",
        type=int,
        default=2008,
        help="Anno minimo degli eventi da processare.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rigenera video, frame overlay e sito anche se gia presenti.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limita il numero di eventi processati.",
    )
    parser.add_argument(
        "--event_regex",
        default="",
        help="Processa solo eventi il cui nome, intervallo o slug matcha questa regex.",
    )
    return parser.parse_args()


def parse_interval(raw: str) -> tuple[datetime, datetime]:
    raw = raw.strip()
    year = int(raw[:4])
    tail = raw[4:]
    if "-" in tail:
        start_part, end_part = tail.split("-", 1)
    else:
        start_part = end_part = tail
    start_month = int(start_part[:2])
    start_day = int(start_part[2:4])
    if len(end_part) == 2:
        end_month = start_month
        end_day = int(end_part)
    else:
        end_month = int(end_part[:2])
        end_day = int(end_part[2:4])
    start = datetime(year, start_month, start_day, 0, 0)
    end = datetime(year, end_month, end_day, 23, 55)
    return start, end


def load_events(csv_path: Path, min_year: int) -> list[EventRow]:
    events: list[EventRow] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = (row.get("yyyymmdd") or "").strip()
            if not raw:
                continue
            start, end = parse_interval(raw)
            if start.year < min_year:
                continue
            events.append(
                EventRow(
                    raw_interval=raw,
                    name=(row.get("NAME") or "").strip(),
                    id_manos_cl7=(row.get("id_manos_cl7") or "").strip(),
                    presente_totale=(row.get("presente_totale") or "").strip(),
                    presente_parziale=(row.get("presente_parziale") or "").strip(),
                    note=(row.get("motivo_assenza_strategy") or "").strip(),
                    start=start,
                    end=end,
                )
            )
    return events


def load_manos_tracks(csv_path: Path) -> dict[str, list[TrackPoint]]:
    tracks: dict[str, list[TrackPoint]] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cyclone_id = (row.get("id_cyc_unico") or "").strip()
            ts_raw = (row.get("time") or "").strip()
            x_raw = (row.get("x_pix") or "").strip()
            y_raw = (row.get("y_pix") or "").strip()
            if not cyclone_id or not ts_raw or not x_raw or not y_raw:
                continue
            try:
                point = TrackPoint(
                    timestamp=datetime.fromisoformat(ts_raw),
                    x=float(x_raw),
                    y=float(y_raw),
                    cyclone_id=cyclone_id,
                    start_time=datetime.fromisoformat(row["start_time"]) if row.get("start_time") else None,
                    end_time=datetime.fromisoformat(row["end_time"]) if row.get("end_time") else None,
                    source=(row.get("source") or "").strip() or "unknown",
                )
            except ValueError:
                continue
            tracks.setdefault(cyclone_id, []).append(point)
    for cyclone_id, pts in tracks.items():
        pts.sort(key=lambda item: item.timestamp)
        deduped: list[TrackPoint] = []
        seen: set[tuple[datetime, float, float]] = set()
        for point in pts:
            key = (point.timestamp, point.x, point.y)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(point)
        tracks[cyclone_id] = deduped
    return tracks


def load_context_tracks(csv_path: Path, sources: set[str]) -> list[TrackPoint]:
    if not sources:
        return []
    selected: list[TrackPoint] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            source = (row.get("source") or "").strip()
            if source not in sources:
                continue
            cyclone_id = (row.get("id_cyc_unico") or "").strip()
            ts_raw = (row.get("time") or "").strip()
            x_raw = (row.get("x_pix") or "").strip()
            y_raw = (row.get("y_pix") or "").strip()
            if not cyclone_id or not ts_raw or not x_raw or not y_raw:
                continue
            try:
                selected.append(
                    TrackPoint(
                        timestamp=datetime.fromisoformat(ts_raw),
                        x=float(x_raw),
                        y=float(y_raw),
                        cyclone_id=cyclone_id,
                        start_time=datetime.fromisoformat(row["start_time"]) if row.get("start_time") else None,
                        end_time=datetime.fromisoformat(row["end_time"]) if row.get("end_time") else None,
                        source=source,
                    )
                )
            except ValueError:
                continue
    selected.sort(key=lambda item: (item.timestamp, item.source, item.cyclone_id, item.x, item.y))
    return selected


def sanitize_slug(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = value.strip("_")
    return value or "unnamed"


def iter_month_dirs(dataset_root: Path, start: datetime, end: datetime) -> Iterable[Path]:
    year = start.year
    month = start.month
    while (year, month) <= (end.year, end.month):
        yield dataset_root / f"{year:04d}" / f"{month:02d}"
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1


def parse_frame_timestamp(path: Path) -> Optional[datetime]:
    match = FRAME_RE.match(path.name)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y%m%d_%H%M")


def collect_event_frames(dataset_root: Path, start: datetime, end: datetime) -> list[EventFrame]:
    found: list[EventFrame] = []
    for month_dir in iter_month_dirs(dataset_root, start, end):
        if not month_dir.exists():
            continue
        for path in sorted(month_dir.glob("airmass_rgb_*.png")):
            ts = parse_frame_timestamp(path)
            if ts is None:
                continue
            if start <= ts <= end:
                found.append(EventFrame(timestamp=ts, path=path))
    found.sort(key=lambda item: (item.timestamp, item.path.name))
    return found


def resolve_ffmpeg_executable(ffmpeg_path: Optional[str] = None) -> Optional[str]:
    if ffmpeg_path:
        cand = Path(ffmpeg_path).expanduser().resolve()
        if cand.is_file():
            return str(cand)
        if cand.is_dir():
            exe = cand / "ffmpeg"
            if exe.exists():
                return str(exe)
    which = shutil.which("ffmpeg")
    if which:
        return which
    local_ffmpeg = Path("/mnt/share/Demetra_files/VideoMAEv2/ffmpeg-7.0.2-amd64-static/ffmpeg")
    if local_ffmpeg.exists():
        return str(local_ffmpeg)
    return None


def recreate_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_frames(source_frames: list[EventFrame], linked_dir: Path) -> None:
    recreate_dir(linked_dir)
    for idx, frame in enumerate(source_frames, start=1):
        target = linked_dir / f"frame_{idx:05d}.png"
        try:
            target.symlink_to(frame.path)
        except OSError:
            os.link(frame.path, target)


def encode_video_from_frames(
    frames_dir: Path,
    output_mp4: Path,
    ffmpeg_exec: str,
    fps: int,
) -> None:
    cmd = [
        ffmpeg_exec,
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
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


def expected_slots(start: datetime, end: datetime) -> int:
    return int(((end - start).total_seconds() / 300.0) + 1)


def select_event_track_points(event: EventRow, tracks_by_id: dict[str, list[TrackPoint]]) -> list[TrackPoint]:
    selected: list[TrackPoint] = []
    ids = [token.strip() for token in event.id_manos_cl7.split(";") if token.strip()]
    for cyclone_id in ids:
        selected.extend(tracks_by_id.get(cyclone_id, []))
    selected = [point for point in selected if event.start <= point.timestamp <= event.end]
    selected.sort(key=lambda item: (item.timestamp, item.cyclone_id, item.x, item.y))
    deduped: list[TrackPoint] = []
    seen: set[tuple[datetime, float, float, str, str]] = set()
    for point in selected:
        key = (point.timestamp, point.x, point.y, point.cyclone_id, point.source)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(point)
    return deduped


def select_context_track_points(event: EventRow, context_tracks: list[TrackPoint]) -> list[TrackPoint]:
    selected = [point for point in context_tracks if event.start <= point.timestamp <= event.end]
    selected.sort(key=lambda item: (item.timestamp, item.source, item.cyclone_id, item.x, item.y))
    return selected


def merge_track_points(*groups: list[TrackPoint]) -> list[TrackPoint]:
    merged: list[TrackPoint] = []
    seen: set[tuple[datetime, float, float, str, str]] = set()
    for group in groups:
        for point in group:
            key = (point.timestamp, point.x, point.y, point.cyclone_id, point.source)
            if key in seen:
                continue
            seen.add(key)
            merged.append(point)
    merged.sort(key=lambda item: (item.timestamp, item.source, item.cyclone_id, item.x, item.y))
    return merged


def build_gt_frame_lookup(
    frames: list[EventFrame],
    track_points: list[TrackPoint],
) -> dict[datetime, list[tuple[float, float, str]]]:
    if not frames or not track_points:
        return {}
    timeline = pd.DataFrame({"datetime": [frame.timestamp for frame in frames]}).sort_values("datetime")
    lookup: dict[datetime, list[tuple[float, float, str]]] = {}
    track_keys = sorted({(point.source, point.cyclone_id) for point in track_points})
    for source, cyclone_id in track_keys:
        cyclone_points = [
            point for point in track_points if point.source == source and point.cyclone_id == cyclone_id
        ]
        gt_src = pd.DataFrame(
            {
                "gt_time": [point.timestamp for point in cyclone_points],
                "x_pix": [point.x for point in cyclone_points],
                "y_pix": [point.y for point in cyclone_points],
                "start_time": [point.start_time for point in cyclone_points],
                "end_time": [point.end_time for point in cyclone_points],
            }
        ).sort_values("gt_time")
        gt_hold = pd.merge_asof(
            timeline,
            gt_src,
            left_on="datetime",
            right_on="gt_time",
            direction="backward",
        )
        gt_x = pd.to_numeric(gt_hold["x_pix"], errors="coerce")
        gt_y = pd.to_numeric(gt_hold["y_pix"], errors="coerce")
        gt_start = pd.to_datetime(gt_hold.get("start_time"), errors="coerce")
        gt_end = pd.to_datetime(gt_hold.get("end_time"), errors="coerce")
        valid_mask = np.isfinite(gt_x) & np.isfinite(gt_y)
        has_start = gt_start.notna()
        has_end = gt_end.notna()
        valid_mask = valid_mask & (~has_start | (gt_hold["datetime"] >= gt_start))
        valid_mask = valid_mask & (~has_end | (gt_hold["datetime"] <= gt_end))
        for ts, x, y, ok in zip(gt_hold["datetime"], gt_x, gt_y, valid_mask):
            if bool(ok):
                frame_ts = pd.Timestamp(ts).to_pydatetime()
                lookup.setdefault(frame_ts, []).append((float(x), float(y), f"{source}:{cyclone_id}"))
    return lookup


def draw_timestamp(img: np.ndarray, timestamp: datetime) -> None:
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype(str(TIMESTAMP_FONT), 30)
    except OSError:
        font = ImageFont.load_default()
    label = pd.Timestamp(timestamp).strftime(" %H:%M %d-%m-%Y")
    bbox = draw.textbbox((0, 0), label, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = pil_img.size[0] - text_width - 15
    y = pil_img.size[1] - text_height - 15
    draw.text((x, y), label, font=font, fill=(255, 80, 80))
    img[:] = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)


def render_overlay_frames(
    frames: list[EventFrame],
    track_points: list[TrackPoint],
    output_dir: Path,
) -> None:
    recreate_dir(output_dir)
    gt_lookup = build_gt_frame_lookup(frames, track_points)
    for idx, frame in enumerate(frames, start=1):
        target = output_dir / f"frame_{idx:05d}.png"
        img = cv2.imread(str(frame.path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        _overlay_coastlines(img)
        for gt_x, gt_y, _cyclone_id in gt_lookup.get(frame.timestamp, []):
            cv2.circle(img, (int(round(gt_x)), int(round(gt_y))), 4, (0, 255, 0), -1)
            cv2.circle(img, (int(round(gt_x)), int(round(gt_y))), 6, (255, 255, 255), 1)
        draw_timestamp(img, frame.timestamp)
        cv2.imwrite(str(target), img)


def write_track_points_csv(event_dir: Path, track_points: list[TrackPoint]) -> Path:
    track_csv = event_dir / "manos_track_points.csv"
    with track_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["time", "x_pix", "y_pix", "id_cyc_unico", "source"])
        writer.writeheader()
        for point in track_points:
            writer.writerow(
                {
                    "time": point.timestamp.isoformat(sep=" "),
                    "x_pix": f"{point.x:.3f}",
                    "y_pix": f"{point.y:.3f}",
                    "id_cyc_unico": point.cyclone_id,
                    "source": point.source,
                }
            )
    return track_csv


def coverage_label(event: EventRow) -> str:
    if event.presente_totale == "1":
        return "complete"
    if event.presente_parziale == "1":
        return "partial"
    return "missing"


def overlay_status_label(track_points: list[TrackPoint], event: EventRow) -> str:
    if track_points:
        return "track_overlay"
    if event.id_manos_cl7:
        return "no_track_points"
    return "no_manos_id"


def write_event_metadata(
    event_dir: Path,
    event: EventRow,
    frame_count: int,
    video_name: str,
    overlay_status: str,
    track_points: list[TrackPoint],
) -> None:
    metadata_path = event_dir / "event_metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "yyyymmdd",
                "name",
                "id_manos_cl7",
                "start",
                "end",
                "expected_slots",
                "frame_count",
                "presente_totale",
                "presente_parziale",
                "motivo_assenza_strategy",
                "video_name",
                "overlay_status",
                "manos_track_points",
                "manos_sources",
                "render_mode",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "yyyymmdd": event.raw_interval,
                "name": event.name,
                "id_manos_cl7": event.id_manos_cl7,
                "start": event.start.isoformat(sep=" "),
                "end": event.end.isoformat(sep=" "),
                "expected_slots": expected_slots(event.start, event.end),
                "frame_count": frame_count,
                "presente_totale": event.presente_totale,
                "presente_parziale": event.presente_parziale,
                "motivo_assenza_strategy": event.note,
                "video_name": video_name,
                "overlay_status": overlay_status,
                "manos_track_points": len(track_points),
                "manos_sources": ";".join(sorted({point.source for point in track_points})),
                "render_mode": RENDER_MODE,
            }
        )


def read_event_metadata(metadata_path: Path) -> dict[str, str]:
    if not metadata_path.exists():
        return {}
    with metadata_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            return {str(k): str(v) for k, v in row.items()}
    return {}


def event_output_ready(
    event_dir: Path,
    output_mp4: Path,
    overlay_status: str,
    frame_count: int,
    require_track_csv: bool,
) -> bool:
    if not output_mp4.exists():
        return False
    metadata = read_event_metadata(event_dir / "event_metadata.csv")
    if not metadata:
        return False
    if metadata.get("render_mode", "") != RENDER_MODE:
        return False
    if metadata.get("overlay_status", "") != overlay_status:
        return False
    if metadata.get("frame_count", "") != str(frame_count):
        return False
    track_csv = event_dir / "manos_track_points.csv"
    if require_track_csv and not track_csv.exists():
        return False
    return True


def relpath_str(target: Path, start: Path) -> str:
    return os.path.relpath(target, start).replace(os.sep, "/")


def event_display_name(row: dict[str, str | int]) -> str:
    name = str(row.get("name") or "").strip()
    interval = str(row.get("yyyymmdd") or "").strip()
    return f"{interval} | {name}" if name else interval


def badge(label: str, text: str) -> str:
    return f'<span class="badge badge-{html.escape(label)}">{html.escape(text)}</span>'


def create_event_page(site_events_dir: Path, row: dict[str, str | int], site_dir: Path) -> Path:
    event_slug = str(row["event_slug"])
    page_path = site_events_dir / f"{event_slug}.html"
    event_dir = Path(str(row["output_dir"]))
    video_path = Path(str(row["video_path"])) if row.get("video_path") else None
    metadata_path = event_dir / "event_metadata.csv"
    track_csv_path = event_dir / "manos_track_points.csv"

    video_html = "<p class=\"empty-state\">Nessun video disponibile per questo evento.</p>"
    if video_path and video_path.exists():
        video_rel = relpath_str(video_path, page_path.parent)
        video_html = (
            f'<div class="video-shell"><video controls preload="metadata" src="{html.escape(video_rel)}"></video></div>'
        )

    metadata_rel = relpath_str(metadata_path, page_path.parent) if metadata_path.exists() else ""
    track_rel = relpath_str(track_csv_path, page_path.parent) if track_csv_path.exists() else ""

    coverage = (
        "Copertura completa" if str(row["presente_totale"]) == "1"
        else "Copertura parziale" if str(row["presente_parziale"]) == "1"
        else "Non disponibile"
    )
    overlay_text = {
        "track_overlay": "Overlay Manos presente",
        "no_track_points": "ID Manos presente ma nessun punto nel range",
        "no_manos_id": "Nessun ID Manos per l'evento",
    }.get(str(row["overlay_status"]), str(row["overlay_status"]))

    html_text = f"""<!doctype html>
<html lang="it">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{html.escape(event_display_name(row))}</title>
    <link rel="stylesheet" href="../style.css">
  </head>
  <body>
    <header class="hero">
      <div class="shell">
        <p class="eyebrow">Complete Dataset</p>
        <h1>{html.escape(event_display_name(row))}</h1>
        <p class="subtitle">Video evento con frame RSS locali e marker ground truth Manos quando disponibile.</p>
      </div>
    </header>
    <main class="shell">
      <section class="panel">
        <div class="panel-head">
          <div>
            <p class="kicker">Evento</p>
            <h2>Dettaglio video</h2>
          </div>
          <a class="back-link" href="../index.html">Torna all'indice</a>
        </div>
        {video_html}
      </section>
      <section class="panel">
        <div class="panel-head">
          <div>
            <p class="kicker">Metadati</p>
            <h2>Stato e copertura</h2>
          </div>
        </div>
        <div class="meta-list">
          <div class="meta-item"><strong>Intervallo</strong>{html.escape(str(row["yyyymmdd"]))}</div>
          <div class="meta-item"><strong>Nome</strong>{html.escape(str(row["name"])) or "-"}</div>
          <div class="meta-item"><strong>ID Manos CL7</strong>{html.escape(str(row["id_manos_cl7"])) or "-"}</div>
          <div class="meta-item"><strong>Status video</strong>{badge(str(row["status"]), str(row["status"]))}</div>
          <div class="meta-item"><strong>Copertura</strong>{coverage}</div>
          <div class="meta-item"><strong>Overlay</strong>{overlay_text}</div>
          <div class="meta-item"><strong>Frame presenti</strong>{html.escape(str(row["frame_count"]))} / {html.escape(str(row["expected_slots"]))}</div>
          <div class="meta-item"><strong>Slot mancanti</strong>{html.escape(str(row["missing_slots_vs_interval"]))}</div>
        </div>
      </section>
      <section class="panel">
        <div class="panel-head">
          <div>
            <p class="kicker">Note</p>
            <h2>Strategia dataset</h2>
          </div>
        </div>
        <div class="note-box">{html.escape(str(row["motivo_assenza_strategy"]))}</div>
      </section>
      <section class="panel">
        <div class="panel-head">
          <div>
            <p class="kicker">File</p>
            <h2>Link diretti</h2>
          </div>
        </div>
        <div class="meta-list">
          <div class="meta-item"><strong>Video</strong>{f'<a href="{html.escape(video_rel)}">apri mp4</a>' if video_path and video_path.exists() else '-'}</div>
          <div class="meta-item"><strong>Event metadata</strong>{f'<a href="{html.escape(metadata_rel)}">apri csv</a>' if metadata_rel else '-'}</div>
          <div class="meta-item"><strong>Track Manos</strong>{f'<a href="{html.escape(track_rel)}">apri csv</a>' if track_rel else '-'}</div>
          <div class="meta-item"><strong>Output dir</strong><code>{html.escape(str(row["output_dir"]))}</code></div>
        </div>
      </section>
    </main>
  </body>
</html>
"""
    page_path.write_text(html_text, encoding="utf-8")
    return page_path


def write_static_site(site_dir: Path, summary_rows: list[dict[str, str | int]]) -> Path:
    site_events_dir = site_dir / "events"
    site_dir.mkdir(parents=True, exist_ok=True)
    site_events_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / "style.css").write_text(SITE_CSS + "\n", encoding="utf-8")

    for row in summary_rows:
        page_path = create_event_page(site_events_dir, row, site_dir)
        row["site_page"] = relpath_str(page_path, site_dir)

    generated = sum(1 for row in summary_rows if row["status"] == "generated_video")
    existing = sum(1 for row in summary_rows if row["status"] == "existing_video")
    no_frames = sum(1 for row in summary_rows if row["status"] == "no_frames")
    complete = sum(1 for row in summary_rows if row["presente_totale"] == "1")
    partial = sum(1 for row in summary_rows if row["presente_parziale"] == "1")
    missing = len(summary_rows) - complete - partial
    with_overlay = sum(1 for row in summary_rows if row["overlay_status"] == "track_overlay")

    table_rows = []
    for row in summary_rows:
        page_rel = str(row["site_page"])
        video_link = "-"
        if row["video_path"]:
            video_rel = relpath_str(Path(str(row["video_path"])), site_dir)
            video_link = f'<a href="{html.escape(video_rel)}">video</a>'
        track_link = "-"
        track_csv = Path(str(row["output_dir"])) / "manos_track_points.csv"
        if track_csv.exists():
            track_link = f'<a href="{html.escape(relpath_str(track_csv, site_dir))}">track csv</a>'
        coverage_badge = (
            badge("complete", "complete") if row["presente_totale"] == "1"
            else badge("partial", "partial") if row["presente_parziale"] == "1"
            else badge("missing", "missing")
        )
        overlay_badge = {
            "track_overlay": badge("track_overlay", "track overlay"),
            "no_track_points": badge("no_track_points", "id without points"),
            "no_manos_id": badge("no_manos_id", "no manos id"),
        }.get(str(row["overlay_status"]), badge("missing", str(row["overlay_status"])))
        table_rows.append(
            "<tr>"
            f"<td><a href=\"{html.escape(page_rel)}\">{html.escape(event_display_name(row))}</a></td>"
            f"<td>{html.escape(str(row['id_manos_cl7']) or '-')}</td>"
            f"<td>{coverage_badge}</td>"
            f"<td>{overlay_badge}</td>"
            f"<td>{html.escape(str(row['frame_count']))} / {html.escape(str(row['expected_slots']))}</td>"
            f"<td>{badge(str(row['status']), str(row['status']))}</td>"
            f"<td class=\"small\">{video_link} | {track_link}</td>"
            "</tr>"
        )

    index_html = f"""<!doctype html>
<html lang="it">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Demetra Complete Dataset</title>
    <link rel="stylesheet" href="style.css">
  </head>
  <body>
    <header class="hero">
      <div class="shell">
        <p class="eyebrow">Static Video Catalog</p>
        <h1>Demetra Complete Dataset</h1>
        <p class="subtitle">Catalogo dei video evento-per-evento costruiti dal dataset RSS locale, con marker ground truth Manos quando disponibile.</p>
      </div>
    </header>
    <main class="shell">
      <section class="panel">
        <div class="panel-head">
          <div>
            <p class="kicker">Panoramica</p>
            <h2>Stato del catalogo</h2>
          </div>
          <p class="panel-note">Root output: <code>{html.escape(str(site_dir.parent))}</code></p>
        </div>
        <div class="stats">
          <div class="stat"><div class="value">{len(summary_rows)}</div><div>Eventi 2008+</div></div>
          <div class="stat"><div class="value">{generated + existing}</div><div>Video disponibili</div></div>
          <div class="stat"><div class="value">{no_frames}</div><div>Eventi senza frame</div></div>
          <div class="stat"><div class="value">{with_overlay}</div><div>Video con overlay Manos</div></div>
          <div class="stat"><div class="value">{complete}</div><div>Copertura completa</div></div>
          <div class="stat"><div class="value">{partial}</div><div>Copertura parziale</div></div>
          <div class="stat"><div class="value">{missing}</div><div>Intervalli assenti</div></div>
        </div>
      </section>
      <section class="panel">
        <div class="panel-head">
          <div>
            <p class="kicker">Eventi</p>
            <h2>Indice completo</h2>
          </div>
          <p class="panel-note">Apri la pagina evento per vedere il video, i metadati e i CSV collegati.</p>
        </div>
        <div class="table-wrap">
          <table class="listing">
            <thead>
              <tr>
                <th>Evento</th>
                <th>ID Manos</th>
                <th>Copertura</th>
                <th>Overlay</th>
                <th>Frame</th>
                <th>Status</th>
                <th>Link</th>
              </tr>
            </thead>
            <tbody>
              {''.join(table_rows)}
            </tbody>
          </table>
        </div>
      </section>
    </main>
  </body>
</html>
"""
    index_path = site_dir / "index.html"
    index_path.write_text(index_html, encoding="utf-8")
    return index_path


def main() -> int:
    args = parse_args()
    presence_csv = Path(args.presence_csv).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    manos_pixel_csv = Path(args.manos_pixel_csv).expanduser().resolve()
    context_manos_pixel_csv = Path(args.context_manos_pixel_csv).expanduser().resolve()
    site_dir = (
        Path(args.site_dir).expanduser().resolve()
        if args.site_dir
        else (output_root / "site").resolve()
    )

    if not presence_csv.exists():
        raise FileNotFoundError(f"CSV eventi non trovato: {presence_csv}")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root non trovato: {dataset_root}")
    if not manos_pixel_csv.exists():
        raise FileNotFoundError(f"CSV Manos pixel non trovato: {manos_pixel_csv}")
    if args.context_sources and not context_manos_pixel_csv.exists():
        raise FileNotFoundError(f"CSV Manos context non trovato: {context_manos_pixel_csv}")

    ffmpeg_exec = resolve_ffmpeg_executable(args.ffmpeg_path or None)
    if ffmpeg_exec is None:
        raise RuntimeError("ffmpeg non trovato. Specifica --ffmpeg_path.")

    events = load_events(presence_csv, min_year=args.min_year)
    if args.event_regex:
        event_pattern = re.compile(args.event_regex, flags=re.IGNORECASE)
        events = [
            event
            for event in events
            if event_pattern.search(
                "__".join(
                    part
                    for part in (
                        event.raw_interval,
                        event.name,
                        event.id_manos_cl7,
                        sanitize_slug(event.name) if event.name else "",
                    )
                    if part
                )
            )
        ]
    if args.limit is not None:
        events = events[: args.limit]
    tracks_by_id = load_manos_tracks(manos_pixel_csv)
    context_sources = {
        source.strip()
        for source in str(args.context_sources).split(",")
        if source.strip()
    }
    context_tracks = load_context_tracks(context_manos_pixel_csv, context_sources)

    output_root.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, str | int]] = []

    batch_t0 = time.perf_counter()

    for idx, event in enumerate(events, start=1):
        event_t0 = time.perf_counter()
        slug_parts = [event.raw_interval]
        if event.name:
            slug_parts.append(sanitize_slug(event.name))
        event_slug = "__".join(slug_parts)
        event_dir = output_root / event_slug
        frames_dir = event_dir / f"anim_frames_{event_slug}"
        video_name = event_slug
        output_mp4 = event_dir / f"{video_name}.mp4"
        frame_items = collect_event_frames(dataset_root, event.start, event.end)
        exp_slots = expected_slots(event.start, event.end)
        missing_slots = max(exp_slots - len(frame_items), 0)
        event_track_points = select_event_track_points(event, tracks_by_id)
        context_track_points = select_context_track_points(event, context_tracks)
        track_points = merge_track_points(event_track_points, context_track_points)
        overlay_status = overlay_status_label(track_points, event)
        status = "no_frames"
        ready = event_output_ready(
            event_dir=event_dir,
            output_mp4=output_mp4,
            overlay_status=overlay_status,
            frame_count=len(frame_items),
            require_track_csv=bool(track_points),
        )

        print(
            f"[{idx}/{len(events)}] {event_slug} | frames={len(frame_items)}/{exp_slots} "
            f"| overlay={overlay_status} | ready={int(ready)}",
            flush=True,
        )

        if ready:
            status = "existing_video"
        elif output_mp4.exists() and not args.overwrite:
            status = "existing_video"
        elif frame_items:
            event_dir.mkdir(parents=True, exist_ok=True)
            render_overlay_frames(frame_items, track_points, frames_dir)
            encode_video_from_frames(
                frames_dir=frames_dir,
                output_mp4=output_mp4,
                ffmpeg_exec=ffmpeg_exec,
                fps=args.fps,
            )
            status = "generated_video"
        else:
            event_dir.mkdir(parents=True, exist_ok=True)
            if frames_dir.exists():
                shutil.rmtree(frames_dir)

        if track_points:
            write_track_points_csv(event_dir, track_points)
        else:
            track_csv = event_dir / "manos_track_points.csv"
            if track_csv.exists():
                track_csv.unlink()

        write_event_metadata(
            event_dir=event_dir,
            event=event,
            frame_count=len(frame_items),
            video_name=video_name,
            overlay_status=overlay_status,
            track_points=track_points,
        )
        summary_rows.append(
            {
                "yyyymmdd": event.raw_interval,
                "name": event.name,
                "id_manos_cl7": event.id_manos_cl7,
                "status": status,
                "frame_count": len(frame_items),
                "expected_slots": exp_slots,
                "missing_slots_vs_interval": missing_slots,
                "presente_totale": event.presente_totale,
                "presente_parziale": event.presente_parziale,
                "motivo_assenza_strategy": event.note,
                "output_dir": str(event_dir),
                "video_path": str(output_mp4 if output_mp4.exists() else ""),
                "event_slug": event_slug,
                "overlay_status": overlay_status,
            }
        )
        print(
            f"[{idx}/{len(events)}] {event_slug} | status={status} | elapsed_s={time.perf_counter() - event_t0:.1f}",
            flush=True,
        )

    index_path = write_static_site(site_dir=site_dir, summary_rows=summary_rows)

    summary_path = output_root / "complete_dataset_video_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "yyyymmdd",
                "name",
                "id_manos_cl7",
                "status",
                "frame_count",
                "expected_slots",
                "missing_slots_vs_interval",
                "presente_totale",
                "presente_parziale",
                "motivo_assenza_strategy",
                "overlay_status",
                "event_slug",
                "output_dir",
                "video_path",
                "site_page",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Output root: {output_root}")
    print(f"Summary: {summary_path}")
    print(f"Site index: {index_path}")
    print(f"Eventi processati: {len(events)}")
    print(f"Video generati: {sum(1 for row in summary_rows if row['status'] == 'generated_video')}")
    print(f"Video gia esistenti: {sum(1 for row in summary_rows if row['status'] == 'existing_video')}")
    print(f"Eventi senza frame: {sum(1 for row in summary_rows if row['status'] == 'no_frames')}")
    print(f"Overlay Manos attivi: {sum(1 for row in summary_rows if row['overlay_status'] == 'track_overlay')}")
    print(f"Tempo totale_s: {time.perf_counter() - batch_t0:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
