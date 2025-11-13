#!/usr/bin/env python3
"""
Copy frames whose timestamp (parsed from filename) falls within cyclone time windows
specified in a CSV (medicanes_new_windows.csv), optionally extended by a configurable
number of days before/after each window. Produces an optional manifest CSV of what was
copied and how it was labeled (core=positive, buffer=negative).

OS: Linux/Windows/macOS
Deps: stdlib only

Usage example:
  python scripts/copy_frames_by_windows.py \
    --windows-csv data/medicanes_new_windows.csv \
    --src /data/frames_src \
    --dst /data/frames_dst \
    --pre-days 2 --post-days 2 \
    --recurse \
    --preserve-structure \
    --write-manifest /data/frames_dst/copied_manifest.csv \
    --time-regex "\d{4}[-_]?(\d{2})[-_]?(\d{2})[T _-]?(\d{2})[-_:]?(\d{2})" \
    --strptime yyyy-MM-ddTHH-mm yyyyMMdd_HHmm yyyyMMddHHmm

Notes:
- Timestamps are treated as naive (no timezone conversion). Ensure CSV and filenames
  are in the same temporal convention.
- If multiple windows overlap, label = max(labels) and event_ids are pipe-joined.
- If a filename doesn't contain a recognizable timestamp, it's skipped with a message.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Dict
import shutil

# -----------------------------
# Utilities
# -----------------------------

CSV_ID_CANDIDATES = ("id", "id_final", "event_id", "cyclone_id")
DEFAULT_TIME_REGEX = r"\d{4}[-_]?(\d{2})[-_]?(\d{2})[T _-]?(\d{2})[-_:]?(\d{2})"
DEFAULT_FILENAME_FORMATS = (
    "yyyy-MM-ddTHH-mm",
    "yyyy-MM-dd_HH-mm",
    "yyyy-MM-dd HH-mm",
    "yyyyMMdd_HHmm",
    "yyyyMMddHHmm",
)
DEFAULT_CSV_TIME_FORMATS = (
    # Try ISO first via fromisoformat, then fallback to these explicit formats
    "yyyy-MM-dd HH:mm",
    "yyyy-MM-ddTHH:mm",
    "yyyy/MM/dd HH:mm",
)
DEFAULT_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

PY_STRPTIME_MAP = {
    "yyyy": "%Y",
    "MM": "%m",
    "dd": "%d",
    "HH": "%H",
    "mm": "%M",
}


def to_strptime(fmt: str) -> str:
    """Convert our short tokens (yyyy,MM,dd,HH,mm) to Python strptime string."""
    out = fmt
    for k, v in PY_STRPTIME_MAP.items():
        out = out.replace(k, v)
    return out


# -----------------------------
# Time parsing
# -----------------------------

def parse_dt_flex(text: str, formats: Sequence[str]) -> Optional[datetime]:
    """Try ISO first, then given formats (our token style)."""
    text = text.strip()
    # ISO
    try:
        # fromisoformat supports 'YYYY-MM-DD HH:MM' and 'YYYY-MM-DDTHH:MM'
        return datetime.fromisoformat(text)
    except Exception:
        pass
    # explicit
    for f in formats:
        try:
            return datetime.strptime(text, to_strptime(f))
        except Exception:
            continue
    return None


def parse_dt_from_filename(name: str, time_regex: re.Pattern, fmts: Sequence[str]) -> Optional[datetime]:
    m = time_regex.search(name)
    if not m:
        return None
    raw = m.group(0)
    dt = parse_dt_flex(raw, fmts)
    if dt is not None:
        return dt
    # fallback: digits only -> try yyyyMMddHHmm
    digits = re.sub(r"\D", "", raw)
    if len(digits) == 12:
        try:
            return datetime.strptime(digits, "%Y%m%d%H%M")
        except Exception:
            return None
    return None


# -----------------------------
# Windows structures
# -----------------------------

@dataclass
class Window:
    start_core: datetime
    end_core: datetime
    start_ext: datetime
    end_ext: datetime
    event_id: str


def load_windows(csv_path: Path, pre_days: int, post_days: int, 
                 csv_time_formats: Sequence[str], id_cols: Sequence[str]) -> List[Window]:
    rows: List[Window] = []
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        rdr = csv.DictReader(fh)
        if not rdr.fieldnames:
            raise RuntimeError("CSV appears empty or has no header")
        # detect id column
        id_col = None
        for cand in id_cols:
            if cand in rdr.fieldnames:
                id_col = cand
                break
        # optional id
        for i, row in enumerate(rdr):
            st = parse_dt_flex(row.get("start_time", ""), csv_time_formats)
            en = parse_dt_flex(row.get("end_time", ""), csv_time_formats)
            if st is None or en is None:
                raise ValueError(f"Unparsable start/end at CSV line {i+2}: {row}")
            if en < st:
                st, en = en, st
            ev_id = row.get(id_col) if id_col else str(i)
            pre = timedelta(days=pre_days)
            post = timedelta(days=post_days)
            rows.append(Window(
                start_core=st,
                end_core=en,
                start_ext=st - pre,
                end_ext=en + post,
                event_id=str(ev_id) if ev_id is not None else str(i)
            ))
    return rows


# -----------------------------
# Copy helpers
# -----------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def do_copy(src: Path, dst: Path, symlink: bool, dry_run: bool) -> Tuple[str, str]:
    """Return (status, msg). status in {copied, exists, skipped, error}."""
    if dst.exists():
        return ("exists", "already exists")
    if dry_run:
        return ("copied", "dry-run")
    try:
        if symlink:
            try:
                # On Windows this may require privileges; fall back to copy on failure
                os.symlink(src, dst)
            except Exception:
                shutil.copy2(src, dst)
        else:
            shutil.copy2(src, dst)
        return ("copied", "ok")
    except Exception as e:
        return ("error", str(e))


# -----------------------------
# Main
# -----------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Copy frames whose timestamp falls within cyclone windows (with pre/post margins), and write a labeled manifest.")
    p.add_argument("--windows-csv", required=True, type=Path)
    p.add_argument("--src", required=True, type=Path)
    p.add_argument("--dst", required=True, type=Path)
    p.add_argument("--pre-days", type=int, default=0, help="Days before start_time to include as buffer (label=0)")
    p.add_argument("--post-days", type=int, default=0, help="Days after end_time to include as buffer (label=0)")
    p.add_argument("--recurse", action="store_true", help="Scan source recursively")
    p.add_argument("--preserve-structure", action="store_true", help="Preserve directory tree under destination")
    p.add_argument("--symlink", action="store_true", help="Create symlinks instead of copying when possible")
    p.add_argument("--write-manifest", type=Path, default=None)
    p.add_argument("--time-regex", type=str, default=DEFAULT_TIME_REGEX, help="Regex to extract a datetime-like token from filename")
    p.add_argument("--strptime", nargs="*", default=list(DEFAULT_FILENAME_FORMATS), help="Filename datetime formats e.g. yyyy-MM-ddTHH-mm yyyyMMdd_HHmm")
    p.add_argument("--csv-strptime", nargs="*", default=list(DEFAULT_CSV_TIME_FORMATS), help="CSV datetime formats if not ISO")
    p.add_argument("--id-cols", nargs="*", default=list(CSV_ID_CANDIDATES), help="Candidate id columns in CSV")
    p.add_argument("--ext", nargs="*", default=list(DEFAULT_EXTS), help="Extensions to include (lowercase)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--workers", type=int, default=1, help="Parallel copy workers (I/O-bound)")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    time_re = re.compile(args.time_regex)
    fname_formats = list(args.strptime)
    csv_formats = list(args.csv_strptime)
    exts = set(e.lower() for e in args.ext)

    # Load windows
    windows = load_windows(args.windows_csv, args.pre_days, args.post_days, csv_formats, args.id_cols)
    if not windows:
        print("No windows found in CSV.")
        return 1

    # Destination
    ensure_dir(args.dst)

    # Scan files
    files: Iterable[Path]
    if args.recurse:
        files = (p for p in args.src.rglob("*") if p.is_file())
    else:
        files = (p for p in args.src.iterdir() if p.is_file())

    # Prepare manifest
    manifest_rows: List[Dict[str, str]] = []

    # Copy executor
    def plan_one(p: Path) -> Optional[Tuple[Path, Path, Dict[str, str]]]:
        if p.suffix.lower() not in exts:
            return None
        dt = parse_dt_from_filename(p.name, time_re, fname_formats)
        if dt is None:
            # Unparsed timestamp; skip
            return None
        # Determine membership
        in_ext_any = False
        in_core_any = False
        ids_ext: List[str] = []
        ids_core: List[str] = []
        for w in windows:
            if w.start_ext <= dt <= w.end_ext:
                in_ext_any = True
                ids_ext.append(w.event_id)
                if w.start_core <= dt <= w.end_core:
                    in_core_any = True
                    ids_core.append(w.event_id)
        if not in_ext_any:
            return None
        label = 1 if in_core_any else 0
        reason = "core" if label == 1 else "buffer"
        event_ids = "|".join(sorted(set(ids_core if label == 1 else ids_ext)))

        # Destination path
        if args.preserve_structure:
            rel = p.relative_to(args.src)
            dst_path = args.dst / rel
            ensure_dir(dst_path.parent)
        else:
            # Flat mode: reuse filename in destination and let do_copy skip if it already exists
            dst_path = args.dst / p.name
            ensure_dir(args.dst)

        row = {
            "src_path": str(p),
            "dst_path": str(dst_path),
            "datetime_iso": dt.isoformat(timespec="minutes"),
            "label": str(label),
            "in_core_window": "1" if in_core_any else "0",
            "in_extended_window": "1",
            "event_ids": event_ids,
            "reason": reason,
            # status filled after copy
        }
        return (p, dst_path, row)

    plans: List[Tuple[Path, Path, Dict[str, str]]] = []
    for p in files:
        res = plan_one(p)
        if res is not None:
            plans.append(res)

    # Execute copy
    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(do_copy, src, dst, args.symlink, args.dry_run): (src, dst, row)
                    for src, dst, row in plans}
            for fut in as_completed(futs):
                src, dst, row = futs[fut]
                status, msg = fut.result()
                row["status"] = status
                row["status_msg"] = msg
                manifest_rows.append(row)
    else:
        for src, dst, row in plans:
            status, msg = do_copy(src, dst, args.symlink, args.dry_run)
            row["status"] = status
            row["status_msg"] = msg
            manifest_rows.append(row)

    # Write manifest
    if args.write_manifest:
        ensure_dir(args.write_manifest.parent)
        fieldnames = [
            "src_path", "dst_path", "datetime_iso", "label",
            "in_core_window", "in_extended_window", "event_ids",
            "reason", "status", "status_msg"
        ]
        with args.write_manifest.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(manifest_rows)

    # Summary
    total = len(manifest_rows)
    copied = sum(1 for r in manifest_rows if r.get("status") == "copied")
    existed = sum(1 for r in manifest_rows if r.get("status") == "exists")
    errors = sum(1 for r in manifest_rows if r.get("status") == "error")
    pos = sum(1 for r in manifest_rows if r.get("label") == "1")
    neg = total - pos
    print(f"Planned {len(plans)}; wrote {total} manifest rows.")
    print(f"Pos {pos} / Neg {neg}; copied {copied}, exists {existed}, errors {errors}.")
    if args.dry_run:
        print("[dry-run] No files were written.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
