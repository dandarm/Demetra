#!/usr/bin/env python3
"""
Organize airmass frames into per-cyclone folders using event windows from CSV.

For each cyclone:
- positives: frames with timestamp inside [start_time, end_time]
- negatives: frames outside all cyclone windows, selected around the event
  to target N_neg ~= N_pos (symmetric before/after when possible)

Selected frames are moved (or copied) into:
  <fromgcloud>/<cyclone_id>/

Frames not selected remain in <fromgcloud> root.
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


ID_COL_CANDIDATES: Tuple[str, ...] = ("id_final", "id_cyc_unico", "id", "cyclone_id")
CSV_TIME_FORMATS: Tuple[str, ...] = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M",
)


@dataclass(frozen=True)
class Event:
    event_id: str
    start: datetime
    end: datetime


@dataclass(frozen=True)
class Frame:
    idx: int
    path: Path
    dt: datetime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Move/copy frames from fromgcloud root into per-cyclone folders, "
            "balancing positive/negative frames."
        )
    )
    parser.add_argument(
        "--windows-csv",
        type=Path,
        default=Path("./Demetra/moduli/videomae/medicane_data_input/medicanes_new_windows.csv"),
        help="CSV with columns start_time/end_time and id column (default: %(default)s).",
    )
    parser.add_argument(
        "--fromgcloud",
        type=Path,
        default=Path("./fromgcloud"),
        help="Root folder containing frame images (default: %(default)s).",
    )
    parser.add_argument(
        "--action",
        choices=("move", "copy"),
        default="move",
        help="File operation to perform (default: %(default)s).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan files recursively (default: only files in fromgcloud root).",
    )
    parser.add_argument(
        "--frame-regex",
        type=str,
        default=r"airmass_rgb_(\d{8})_(\d{4})\.png$",
        help="Regex with two groups: YYYYMMDD and HHMM (default: %(default)s).",
    )
    parser.add_argument(
        "--ext",
        nargs="*",
        default=[".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"],
        help="Allowed extensions (default: %(default)s).",
    )
    parser.add_argument(
        "--create-empty-dirs",
        action="store_true",
        help="Create cyclone directories even if no files are assigned.",
    )
    parser.add_argument(
        "--report-csv",
        type=Path,
        default=None,
        help="Optional summary CSV report path.",
    )
    parser.add_argument(
        "--selection-csv",
        type=Path,
        default=None,
        help="Optional per-file selection CSV report path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan only; do not move/copy files.",
    )
    return parser.parse_args()


def parse_dt(text: str) -> Optional[datetime]:
    raw = str(text).strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        pass
    for fmt in CSV_TIME_FORMATS:
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def normalize_event_id(raw: object) -> Optional[str]:
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return None
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except ValueError:
        pass
    return s


def load_events(windows_csv: Path) -> List[Event]:
    if not windows_csv.exists():
        raise FileNotFoundError(f"CSV not found: {windows_csv}")

    with windows_csv.open("r", newline="", encoding="utf-8") as fh:
        rdr = csv.DictReader(fh)
        if not rdr.fieldnames:
            raise ValueError(f"CSV appears empty or has no header: {windows_csv}")
        id_col = next((c for c in ID_COL_CANDIDATES if c in rdr.fieldnames), None)
        if id_col is None:
            raise ValueError(
                f"Missing id column. Expected one of: {', '.join(ID_COL_CANDIDATES)}"
            )

        agg: Dict[str, Tuple[datetime, datetime]] = {}
        for line_no, row in enumerate(rdr, start=2):
            start = parse_dt(row.get("start_time", ""))
            end = parse_dt(row.get("end_time", ""))
            ev_id = normalize_event_id(row.get(id_col))
            if ev_id is None:
                raise ValueError(f"Missing/invalid event id at CSV line {line_no}")
            if start is None or end is None:
                raise ValueError(f"Unparsable start/end at CSV line {line_no}")
            if end < start:
                start, end = end, start
            cur = agg.get(ev_id)
            if cur is None:
                agg[ev_id] = (start, end)
            else:
                agg[ev_id] = (min(cur[0], start), max(cur[1], end))

    events = [Event(event_id=ev_id, start=st, end=en) for ev_id, (st, en) in agg.items()]
    events.sort(key=lambda x: (x.start, x.end, x.event_id))
    return events


def scan_frames(
    root: Path, frame_re: re.Pattern[str], exts: Sequence[str], recursive: bool
) -> List[Frame]:
    if not root.exists():
        raise FileNotFoundError(f"fromgcloud folder not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"fromgcloud path is not a directory: {root}")

    exts_lc = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts}
    iterator: Iterable[Path]
    if recursive:
        iterator = (p for p in root.rglob("*") if p.is_file())
    else:
        iterator = (p for p in root.iterdir() if p.is_file())

    frames: List[Frame] = []
    bad_count = 0
    for p in iterator:
        if p.suffix.lower() not in exts_lc:
            continue
        m = frame_re.search(p.name)
        if not m:
            bad_count += 1
            continue
        try:
            dt = datetime.strptime(f"{m.group(1)}{m.group(2)}", "%Y%m%d%H%M")
        except ValueError:
            bad_count += 1
            continue
        frames.append(Frame(idx=-1, path=p, dt=dt))

    frames.sort(key=lambda x: (x.dt, x.path.name))
    return [Frame(idx=i, path=f.path, dt=f.dt) for i, f in enumerate(frames)]


def assign_positive_owner(
    frames: Sequence[Frame], events: Sequence[Event]
) -> Tuple[Dict[str, List[int]], List[bool], int]:
    positives_by_event: Dict[str, List[int]] = defaultdict(list)
    is_core_any = [False] * len(frames)
    overlap_count = 0

    for fr in frames:
        candidates = [ev for ev in events if ev.start <= fr.dt <= ev.end]
        if not candidates:
            continue
        is_core_any[fr.idx] = True
        if len(candidates) == 1:
            owner = candidates[0]
        else:
            overlap_count += 1
            owner = min(
                candidates,
                key=lambda ev: abs(
                    fr.dt - (ev.start + (ev.end - ev.start) / 2)
                ),
            )
        positives_by_event[owner.event_id].append(fr.idx)
    return positives_by_event, is_core_any, overlap_count


def choose_balanced_negatives(
    before: List[int],
    after: List[int],
    target_neg: int,
    frames: Sequence[Frame],
    start: datetime,
    end: datetime,
) -> List[int]:
    if target_neg <= 0:
        return []

    pre_target = target_neg // 2
    post_target = target_neg - pre_target
    chosen: List[int] = []

    chosen.extend(before[:pre_target])
    chosen.extend(after[:post_target])

    rem_before = before[pre_target:]
    rem_after = after[post_target:]

    ib = 0
    ia = 0
    while len(chosen) < target_neg and (ib < len(rem_before) or ia < len(rem_after)):
        b_idx = rem_before[ib] if ib < len(rem_before) else None
        a_idx = rem_after[ia] if ia < len(rem_after) else None

        if b_idx is None:
            chosen.append(a_idx)  # type: ignore[arg-type]
            ia += 1
            continue
        if a_idx is None:
            chosen.append(b_idx)
            ib += 1
            continue

        dist_b = start - frames[b_idx].dt
        dist_a = frames[a_idx].dt - end
        if dist_b <= dist_a:
            chosen.append(b_idx)
            ib += 1
        else:
            chosen.append(a_idx)
            ia += 1

    return chosen


def write_summary_report(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "event_id",
        "event_start",
        "event_end",
        "n_pos",
        "target_neg",
        "picked_neg",
        "picked_total",
        "moved_or_copied",
        "already_present",
        "errors",
        "neg_before_count",
        "neg_after_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_selection_report(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["event_id", "label", "timestamp", "src_path", "dst_path", "status", "message"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> int:
    args = parse_args()
    frame_re = re.compile(args.frame_regex)
    root = args.fromgcloud.resolve()
    windows_csv = args.windows_csv.resolve()
    report_csv = args.report_csv.resolve() if args.report_csv else None
    selection_csv = args.selection_csv.resolve() if args.selection_csv else None

    events = load_events(windows_csv)
    if not events:
        raise RuntimeError("No cyclone events found in CSV.")

    frames = scan_frames(root, frame_re, args.ext, args.recursive)
    if not frames:
        raise RuntimeError("No valid frames found in fromgcloud.")

    positives_by_event, is_core_any, overlap_count = assign_positive_owner(frames, events)

    available_neg = {fr.idx for fr in frames if not is_core_any[fr.idx]}
    event_to_selected: Dict[str, List[int]] = {}
    selection_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    for ev in events:
        pos_idxs = sorted(positives_by_event.get(ev.event_id, []), key=lambda i: frames[i].dt)
        n_pos = len(pos_idxs)
        target_neg = n_pos

        neg_sorted = sorted(available_neg, key=lambda i: frames[i].dt)
        before = [i for i in neg_sorted if frames[i].dt < ev.start]
        after = [i for i in neg_sorted if frames[i].dt > ev.end]

        # Closest to the event boundary first.
        before.reverse()

        neg_idxs = choose_balanced_negatives(before, after, target_neg, frames, ev.start, ev.end)
        neg_set = set(neg_idxs)
        available_neg.difference_update(neg_set)

        all_idxs = sorted(pos_idxs + neg_idxs, key=lambda i: frames[i].dt)
        event_to_selected[ev.event_id] = all_idxs

        before_count = sum(1 for i in neg_idxs if frames[i].dt < ev.start)
        after_count = len(neg_idxs) - before_count

        summary_rows.append(
            {
                "event_id": ev.event_id,
                "event_start": ev.start.isoformat(sep=" ", timespec="minutes"),
                "event_end": ev.end.isoformat(sep=" ", timespec="minutes"),
                "n_pos": n_pos,
                "target_neg": target_neg,
                "picked_neg": len(neg_idxs),
                "picked_total": len(all_idxs),
                "moved_or_copied": 0,
                "already_present": 0,
                "errors": 0,
                "neg_before_count": before_count,
                "neg_after_count": after_count,
            }
        )

    if args.create_empty_dirs or not args.dry_run:
        for ev in events:
            ev_dir = root / ev.event_id
            if not args.dry_run:
                ev_dir.mkdir(parents=True, exist_ok=True)

    summary_by_id = {str(r["event_id"]): r for r in summary_rows}

    for ev in events:
        ev_dir = root / ev.event_id
        selected = event_to_selected.get(ev.event_id, [])
        pos_set = set(positives_by_event.get(ev.event_id, []))
        for idx in selected:
            fr = frames[idx]
            dst = ev_dir / fr.path.name
            label = 1 if idx in pos_set else 0

            if dst.exists():
                status = "exists"
                msg = "destination already exists"
                summary_by_id[ev.event_id]["already_present"] = int(
                    summary_by_id[ev.event_id]["already_present"]
                ) + 1
            elif args.dry_run:
                status = "planned"
                msg = "dry-run"
                summary_by_id[ev.event_id]["moved_or_copied"] = int(
                    summary_by_id[ev.event_id]["moved_or_copied"]
                ) + 1
            else:
                try:
                    if args.action == "move":
                        shutil.move(str(fr.path), str(dst))
                    else:
                        shutil.copy2(fr.path, dst)
                    status = args.action
                    msg = "ok"
                    summary_by_id[ev.event_id]["moved_or_copied"] = int(
                        summary_by_id[ev.event_id]["moved_or_copied"]
                    ) + 1
                except Exception as exc:  # pragma: no cover
                    status = "error"
                    msg = str(exc)
                    summary_by_id[ev.event_id]["errors"] = int(
                        summary_by_id[ev.event_id]["errors"]
                    ) + 1

            if selection_csv:
                selection_rows.append(
                    {
                        "event_id": ev.event_id,
                        "label": label,
                        "timestamp": fr.dt.isoformat(sep=" ", timespec="minutes"),
                        "src_path": str(fr.path),
                        "dst_path": str(dst),
                        "status": status,
                        "message": msg,
                    }
                )

    if report_csv:
        write_summary_report(report_csv, summary_rows)
    if selection_csv:
        write_selection_report(selection_csv, selection_rows)

    total_pos = sum(int(r["n_pos"]) for r in summary_rows)
    total_neg = sum(int(r["picked_neg"]) for r in summary_rows)
    total_sel = sum(int(r["picked_total"]) for r in summary_rows)
    total_done = sum(int(r["moved_or_copied"]) for r in summary_rows)
    total_exists = sum(int(r["already_present"]) for r in summary_rows)
    total_err = sum(int(r["errors"]) for r in summary_rows)

    print(f"Events: {len(events)} | Frames scanned: {len(frames)} | Overlap positives: {overlap_count}")
    print(f"Selected total: {total_sel} | Positives: {total_pos} | Negatives picked: {total_neg}")
    print(
        f"Action: {args.action} | {'dry-run' if args.dry_run else 'live'} | "
        f"done: {total_done}, exists: {total_exists}, errors: {total_err}"
    )
    if report_csv:
        print(f"Summary report: {report_csv}")
    if selection_csv:
        print(f"Selection report: {selection_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
