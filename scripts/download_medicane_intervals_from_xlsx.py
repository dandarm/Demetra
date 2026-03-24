#!/home/isac/miniconda3/envs/videomae/bin/python
"""Parse medicane intervals from XLSX and download only missing frames."""
from __future__ import annotations

import argparse
import csv
import logging
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from download_and_track_range import (  # noqa: E402
    AIRMAss_OUTPUT_ROOT_DEFAULT,
    FRAME_RE,
    PYTHON_EXEC_DEFAULT,
    build_run_paths,
)


LOG = logging.getLogger("download_medicane_xlsx")
FIVE_MINUTES = pd.Timedelta(minutes=5)
XLSX_DEFAULT = REPO_ROOT / "notebooks" / "Full_List_Medicanes_new.xlsx"
DATASET_ROOT_DEFAULT = Path("/media/isacDisk2/source_dataset")
DOWNLOAD_SCRIPT = REPO_ROOT / "scripts" / "download_and_track_range.py"
FIRST_COL_HEADER = "yyyymmdd"
XLSX_NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}
SAME_MONTH_RE = re.compile(r"^\d{8}-\d{2}$")
CROSS_MONTH_RE = re.compile(r"^\d{8}-\d{4}$")
FULL_RANGE_RE = re.compile(r"^\d{8}-\d{8}$")
SINGLE_DAY_RE = re.compile(r"^\d{8}$")


@dataclass(frozen=True)
class MedicaneInterval:
    row_index: int
    raw_value: str
    start: pd.Timestamp
    end: pd.Timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Legge gli intervalli temporali dei medicanes da un file XLSX e "
            "scarica solo i frame mancanti rispetto a source_dataset."
        )
    )
    parser.add_argument(
        "--xlsx_path",
        default=str(XLSX_DEFAULT),
        help="File XLSX con la lista dei medicanes.",
    )
    parser.add_argument(
        "--sheet_name",
        default=None,
        help="Nome del foglio Excel da usare. Default: primo foglio.",
    )
    parser.add_argument(
        "--dataset_root",
        default=str(DATASET_ROOT_DEFAULT),
        help="Root del dataset locale gia scaricato, organizzato in anno/mese.",
    )
    parser.add_argument(
        "--output_root",
        default=str(AIRMAss_OUTPUT_ROOT_DEFAULT),
        help="Root cache per le run di download_and_track_range.py.",
    )
    parser.add_argument(
        "--python_exec",
        default=str(PYTHON_EXEC_DEFAULT),
        help="Interprete Python con cui lanciare download_and_track_range.py.",
    )
    parser.add_argument(
        "--download_source",
        choices=["auto", "public", "eumetsat"],
        default="eumetsat",
        help="Sorgente da passare al downloader esistente.",
    )
    parser.add_argument(
        "--eumetsat_download_workers",
        type=int,
        default=4,
        help="Numero di worker concorrenti per il downloader EUMETSAT.",
    )
    parser.add_argument(
        "--eumetsat_download_retries",
        type=int,
        default=3,
        help="Numero di retry per download EUMETSAT.",
    )
    parser.add_argument(
        "--eumetsat_read_timeout",
        type=int,
        default=180,
        help="Read timeout in secondi per stream EUMETSAT.",
    )
    parser.add_argument(
        "--max_intervals",
        type=int,
        default=None,
        help="Limita il numero di intervalli elaborati dall'xlsx.",
    )
    parser.add_argument(
        "--row_index",
        type=int,
        action="append",
        default=None,
        help="Elabora solo le righe Excel specificate. Opzione ripetibile.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Non scarica nulla; produce solo il piano dei range mancanti.",
    )
    parser.add_argument(
        "--summary_csv",
        default=None,
        help="CSV di riepilogo. Default: <output_root>/medicane_interval_download_summary.csv",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


def _xlsx_shared_strings(zf: ZipFile) -> List[str]:
    shared_path = "xl/sharedStrings.xml"
    if shared_path not in zf.namelist():
        return []
    root = ET.fromstring(zf.read(shared_path))
    out: List[str] = []
    for si in root:
        out.append("".join(t.text or "" for t in si.iter(f"{{{XLSX_NS['a']}}}t")))
    return out


def _resolve_sheet_target(zf: ZipFile, sheet_name: str | None) -> Tuple[str, str]:
    wb = ET.fromstring(zf.read("xl/workbook.xml"))
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}

    sheets = list(wb.find("a:sheets", XLSX_NS))
    if not sheets:
        raise RuntimeError("Nessun foglio trovato nell'xlsx.")

    selected = sheets[0]
    if sheet_name is not None:
        for sheet in sheets:
            if sheet.attrib.get("name") == sheet_name:
                selected = sheet
                break
        else:
            available = ", ".join(sheet.attrib.get("name", "?") for sheet in sheets)
            raise ValueError(f"Foglio {sheet_name!r} non trovato. Disponibili: {available}")

    resolved_name = selected.attrib.get("name", "Sheet1")
    rid = selected.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
    return resolved_name, "xl/" + rel_map[rid]


def load_first_column_values(xlsx_path: Path, sheet_name: str | None = None) -> List[Tuple[int, str]]:
    with ZipFile(xlsx_path) as zf:
        shared = _xlsx_shared_strings(zf)
        resolved_sheet_name, target = _resolve_sheet_target(zf, sheet_name)
        LOG.info("Foglio XLSX selezionato: %s", resolved_sheet_name)
        root = ET.fromstring(zf.read(target))
        rows = root.find("a:sheetData", XLSX_NS)
        if rows is None:
            return []

        values: List[Tuple[int, str]] = []
        for row in rows:
            row_index = int(row.attrib["r"])
            cell = row.find(f"a:c[@r='A{row_index}']", XLSX_NS)
            if cell is None:
                continue
            cell_type = cell.attrib.get("t")
            value_node = cell.find("a:v", XLSX_NS)
            value = value_node.text if value_node is not None else ""
            if cell_type == "s" and value:
                value = shared[int(value)]
            value = (value or "").strip()
            if not value or value == FIRST_COL_HEADER:
                continue
            values.append((row_index, value))
        return values


def parse_compact_interval(value: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    raw = value.strip()
    if SINGLE_DAY_RE.fullmatch(raw):
        start = pd.Timestamp(datetime.strptime(raw, "%Y%m%d"))
        end = start
    elif SAME_MONTH_RE.fullmatch(raw):
        start = pd.Timestamp(datetime.strptime(raw[:8], "%Y%m%d"))
        end = pd.Timestamp(datetime.strptime(raw[:6] + raw[-2:], "%Y%m%d"))
    elif CROSS_MONTH_RE.fullmatch(raw):
        start = pd.Timestamp(datetime.strptime(raw[:8], "%Y%m%d"))
        mmdd = raw[-4:]
        end = pd.Timestamp(datetime.strptime(f"{start.year}{mmdd}", "%Y%m%d"))
        if end < start:
            end = pd.Timestamp(datetime.strptime(f"{start.year + 1}{mmdd}", "%Y%m%d"))
    elif FULL_RANGE_RE.fullmatch(raw):
        start = pd.Timestamp(datetime.strptime(raw[:8], "%Y%m%d"))
        end = pd.Timestamp(datetime.strptime(raw[-8:], "%Y%m%d"))
    else:
        raise ValueError(f"Formato intervallo non riconosciuto: {value}")

    start = pd.Timestamp(start).replace(hour=0, minute=0)
    end = pd.Timestamp(end).replace(hour=23, minute=55)
    if end < start:
        raise ValueError(f"Intervallo invalido: {value}")
    return start, end


def load_medicane_intervals(xlsx_path: Path, sheet_name: str | None = None) -> List[MedicaneInterval]:
    intervals: List[MedicaneInterval] = []
    for row_index, raw_value in load_first_column_values(xlsx_path, sheet_name):
        start, end = parse_compact_interval(raw_value)
        intervals.append(
            MedicaneInterval(
                row_index=row_index,
                raw_value=raw_value,
                start=start,
                end=end,
            )
        )
    return intervals


def collect_existing_recursive(dataset_root: Path) -> Dict[pd.Timestamp, Path]:
    frame_map: Dict[pd.Timestamp, Path] = {}
    if not dataset_root.exists():
        return frame_map
    for path in sorted(dataset_root.rglob("airmass_rgb_*.png")):
        match = FRAME_RE.match(path.name)
        if not match:
            continue
        try:
            ts = pd.Timestamp(datetime.strptime(match.group(1), "%Y%m%d_%H%M"))
        except ValueError:
            continue
        frame_map[ts] = path
    return frame_map


def expected_timestamps(start: pd.Timestamp, end: pd.Timestamp) -> List[pd.Timestamp]:
    return [pd.Timestamp(ts) for ts in pd.date_range(start=start, end=end, freq="5min")]


def missing_timestamps_for_interval(
    interval: MedicaneInterval,
    existing_map: Dict[pd.Timestamp, Path],
) -> List[pd.Timestamp]:
    return [ts for ts in expected_timestamps(interval.start, interval.end) if ts not in existing_map]


def condense_missing_segments(missing_times: Sequence[pd.Timestamp]) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    if not missing_times:
        return []
    ordered = sorted(pd.Timestamp(ts) for ts in missing_times)
    segments: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    seg_start = ordered[0]
    seg_end = ordered[0]
    for ts in ordered[1:]:
        if ts - seg_end == FIVE_MINUTES:
            seg_end = ts
            continue
        segments.append((seg_start, seg_end))
        seg_start = ts
        seg_end = ts
    segments.append((seg_start, seg_end))
    return segments


def sync_frames_into_dataset(frames_dir: Path, dataset_root: Path, existing_map: Dict[pd.Timestamp, Path]) -> int:
    copied = 0
    for path in sorted(frames_dir.glob("airmass_rgb_*.png")):
        match = FRAME_RE.match(path.name)
        if not match:
            continue
        ts = pd.Timestamp(datetime.strptime(match.group(1), "%Y%m%d_%H%M"))
        if ts in existing_map:
            continue
        target_dir = dataset_root / ts.strftime("%Y") / ts.strftime("%m")
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / path.name
        if not target.exists():
            shutil.copy2(path, target)
        existing_map[ts] = target
        copied += 1
    return copied


def format_cli_datetime(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M")


def run_download_segment(
    python_exec: Path,
    output_root: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    download_source: str,
    eumetsat_download_workers: int,
    eumetsat_download_retries: int,
    eumetsat_read_timeout: int,
) -> Path:
    modes_to_try = [download_source]
    if download_source != "eumetsat":
        modes_to_try.append("eumetsat")

    last_error: str | None = None
    run_dir, frames_dir = build_run_paths(output_root, start, end)
    for mode in modes_to_try:
        cmd = [
            str(python_exec),
            str(DOWNLOAD_SCRIPT),
            "--start",
            format_cli_datetime(start),
            "--end",
            format_cli_datetime(end),
            "--output_root",
            str(output_root),
            "--download_source",
            mode,
            "--skip_inference",
            "--eumetsat_download_workers",
            str(eumetsat_download_workers),
            "--eumetsat_download_retries",
            str(eumetsat_download_retries),
            "--eumetsat_read_timeout",
            str(eumetsat_read_timeout),
        ]
        LOG.info("Lancio download segmento (%s): %s", mode, " ".join(cmd))
        completed = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
        if completed.returncode == 0 and frames_dir.exists():
            return frames_dir
        last_error = f"download_source={mode} returncode={completed.returncode}"
        LOG.warning(
            "Tentativo fallito per %s -> %s con sorgente %s",
            start,
            end,
            mode,
        )

    raise RuntimeError(
        f"Download fallito per il segmento {start} -> {end}. Ultimo errore: {last_error}"
    )


def write_summary_csv(summary_path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    fieldnames = [
        "row_index",
        "raw_interval",
        "start",
        "end",
        "expected_slots",
        "missing_slots",
        "missing_segments",
        "status",
        "message",
    ]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    setup_logging()

    xlsx_path = Path(args.xlsx_path).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    python_exec = Path(args.python_exec).expanduser().resolve()
    summary_csv = (
        Path(args.summary_csv).expanduser().resolve()
        if args.summary_csv
        else output_root / "medicane_interval_download_summary.csv"
    )

    if not xlsx_path.exists():
        raise FileNotFoundError(f"XLSX non trovato: {xlsx_path}")
    if not python_exec.exists():
        raise FileNotFoundError(f"Python env non trovato: {python_exec}")
    if not DOWNLOAD_SCRIPT.exists():
        raise FileNotFoundError(f"Script downloader non trovato: {DOWNLOAD_SCRIPT}")

    intervals = load_medicane_intervals(xlsx_path, args.sheet_name)
    if args.row_index:
        wanted = set(args.row_index)
        intervals = [interval for interval in intervals if interval.row_index in wanted]
    if args.max_intervals is not None:
        intervals = intervals[: args.max_intervals]
    LOG.info("Intervalli letti dall'xlsx: %d", len(intervals))

    existing_map = collect_existing_recursive(dataset_root)
    LOG.info("Frame gia presenti in %s: %d", dataset_root, len(existing_map))

    summary_rows: List[dict] = []
    downloaded_segments = 0
    synced_frames = 0
    failures = 0

    for interval in intervals:
        missing_times = missing_timestamps_for_interval(interval, existing_map)
        missing_segments = condense_missing_segments(missing_times)
        row_summary = {
            "row_index": interval.row_index,
            "raw_interval": interval.raw_value,
            "start": interval.start.isoformat(),
            "end": interval.end.isoformat(),
            "expected_slots": len(expected_timestamps(interval.start, interval.end)),
            "missing_slots": len(missing_times),
            "missing_segments": len(missing_segments),
            "status": "covered",
            "message": "",
        }

        if not missing_segments:
            LOG.info(
                "[row %s] %s gia coperto: %s -> %s",
                interval.row_index,
                interval.raw_value,
                interval.start,
                interval.end,
            )
            summary_rows.append(row_summary)
            continue

        if args.dry_run:
            row_summary["status"] = "dry_run_missing"
            row_summary["message"] = "; ".join(
                f"{seg_start.strftime('%Y-%m-%d %H:%M')} -> {seg_end.strftime('%Y-%m-%d %H:%M')}"
                for seg_start, seg_end in missing_segments
            )
            LOG.info(
                "[row %s] %s missing slots=%d segments=%d",
                interval.row_index,
                interval.raw_value,
                len(missing_times),
                len(missing_segments),
            )
            summary_rows.append(row_summary)
            continue

        row_messages: List[str] = []
        row_failed = False
        for seg_start, seg_end in missing_segments:
            try:
                frames_dir = run_download_segment(
                    python_exec=python_exec,
                    output_root=output_root,
                    start=seg_start,
                    end=seg_end,
                    download_source=args.download_source,
                    eumetsat_download_workers=args.eumetsat_download_workers,
                    eumetsat_download_retries=args.eumetsat_download_retries,
                    eumetsat_read_timeout=args.eumetsat_read_timeout,
                )
                downloaded_segments += 1
                copied = sync_frames_into_dataset(frames_dir, dataset_root, existing_map)
                synced_frames += copied
                row_messages.append(
                    f"{format_cli_datetime(seg_start)} -> {format_cli_datetime(seg_end)} copied={copied}"
                )
            except Exception as exc:
                failures += 1
                row_failed = True
                row_messages.append(
                    f"{format_cli_datetime(seg_start)} -> {format_cli_datetime(seg_end)} failed={exc}"
                )
                LOG.exception(
                    "Download fallito per il segmento %s -> %s (row %s, %s)",
                    seg_start,
                    seg_end,
                    interval.row_index,
                    interval.raw_value,
                )

        remaining_missing = missing_timestamps_for_interval(interval, existing_map)
        row_summary["missing_slots"] = len(remaining_missing)
        row_summary["missing_segments"] = len(condense_missing_segments(remaining_missing))
        row_summary["status"] = "partial_failure" if row_failed else "downloaded"
        row_summary["message"] = " | ".join(row_messages)
        summary_rows.append(row_summary)

    write_summary_csv(summary_csv, summary_rows)
    LOG.info("Summary scritto in: %s", summary_csv)
    LOG.info(
        "Completato. dry_run=%s | segmenti lanciati=%d | frame sincronizzati=%d | failure=%d",
        args.dry_run,
        downloaded_segments,
        synced_frames,
        failures,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
