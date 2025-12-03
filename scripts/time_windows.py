#!/usr/bin/env python3
"""Shared utilities for selecting frames by temporal windows."""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Sequence

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
    """Extract datetime from filename via regex + formats."""
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


@dataclass
class Window:
    start_core: datetime
    end_core: datetime
    start_ext: datetime
    end_ext: datetime
    event_id: str


def load_windows(
    csv_path: Path,
    pre: timedelta,
    post: timedelta,
    csv_time_formats: Sequence[str],
    id_cols: Sequence[str],
) -> List[Window]:
    """Load windows from CSV and expand them with pre/post margins."""
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
        for i, row in enumerate(rdr):
            st = parse_dt_flex(row.get("start_time", ""), csv_time_formats)
            en = parse_dt_flex(row.get("end_time", ""), csv_time_formats)
            if st is None or en is None:
                raise ValueError(f"Unparsable start/end at CSV line {i+2}: {row}")
            if en < st:
                st, en = en, st
            ev_id = row.get(id_col) if id_col else str(i)
            rows.append(
                Window(
                    start_core=st,
                    end_core=en,
                    start_ext=st - pre,
                    end_ext=en + post,
                    event_id=str(ev_id) if ev_id is not None else str(i),
                )
            )
    return rows
