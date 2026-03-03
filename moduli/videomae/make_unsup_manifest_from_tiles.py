#!/usr/bin/env python3
"""Create an unsupervised manifest CSV from pre-saved videotile folders.

Output format is compatible with specialization/HybridVideoMAE:
path,start,end,x_off,y_off

For pre-cropped tile folders, offsets should be 0,0 (default). If requested,
offsets can be parsed from folder name suffix "..._<x>_<y>".
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pandas as pd


FRAME_RE = re.compile(r"^img_(\d{5})\.(png|jpg|jpeg)$", re.IGNORECASE)
OFFSET_RE = re.compile(r"_(?P<x>-?\d+)_(?P<y>-?\d+)(?:_v\d+)?$")


def _collect_tile_folders(root: Path, num_frames: int):
    folders = []
    skipped_short = 0
    scanned_dirs = 0
    required = set(range(1, int(num_frames) + 1))

    for dirpath, dirnames, filenames in os.walk(root):
        scanned_dirs += 1
        frame_indices = []
        for fname in filenames:
            match = FRAME_RE.match(fname)
            if match:
                frame_indices.append(int(match.group(1)))

        if frame_indices:
            if required.issubset(set(frame_indices)):
                folders.append(Path(dirpath).resolve())
            else:
                skipped_short += 1
            # If this folder already contains frame files, do not descend further.
            dirnames[:] = []

    return sorted(folders), skipped_short, scanned_dirs


def _parse_offsets_from_name(folder_name: str):
    match = OFFSET_RE.search(folder_name)
    if match is None:
        return None
    return int(match.group("x")), int(match.group("y"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create unsupervised manifest CSV from tile folders."
    )

    # Support both positional and named args.
    parser.add_argument("input_dir", nargs="?", help="Root folder containing tile subfolders.")
    parser.add_argument("output_csv", nargs="?", help="Output CSV path.")
    parser.add_argument("--input_dir", dest="input_dir_opt", help="Root folder containing tile subfolders.")
    parser.add_argument("--output_csv", dest="output_csv_opt", help="Output CSV path.")

    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
        help="Number of required frames per tile folder (default: 16).",
    )
    parser.add_argument(
        "--use_offsets_from_name",
        action="store_true",
        help="Parse x_off/y_off from folder suffix ..._<x>_<y>; default writes x_off=y_off=0.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir_opt if args.input_dir_opt is not None else args.input_dir
    output_csv = args.output_csv_opt if args.output_csv_opt is not None else args.output_csv
    if input_dir is None or output_csv is None:
        raise SystemExit(
            "Devi specificare input_dir e output_csv (posizionali o con --input_dir/--output_csv)."
        )
    if args.num_frames <= 0:
        raise ValueError("--num_frames must be > 0")

    in_root = Path(input_dir).expanduser().resolve()
    out_csv = Path(output_csv).expanduser().resolve()

    if not in_root.exists():
        raise FileNotFoundError(f"Input dir non trovata: {in_root}")

    folders, skipped_short, scanned_dirs = _collect_tile_folders(in_root, num_frames=args.num_frames)
    if not folders:
        raise RuntimeError(
            f"Nessuna tile folder valida trovata in {in_root} con almeno i frame 1..{args.num_frames}."
        )

    rows = []
    missing_offsets = 0
    for folder in folders:
        if args.use_offsets_from_name:
            parsed = _parse_offsets_from_name(folder.name)
            if parsed is None:
                x_off, y_off = 0, 0
                missing_offsets += 1
            else:
                x_off, y_off = parsed
        else:
            x_off, y_off = 0, 0

        rows.append(
            {
                "path": str(folder),
                "start": 1,
                "end": int(args.num_frames),
                "x_off": int(x_off),
                "y_off": int(y_off),
            }
        )

    df = pd.DataFrame(rows, columns=["path", "start", "end", "x_off", "y_off"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print("Unsupervised manifest creato.")
    print(f"input_dir             : {in_root}")
    print(f"output_csv            : {out_csv}")
    print(f"scanned_dirs          : {scanned_dirs}")
    print(f"valid_tile_folders    : {len(df)}")
    print(f"skipped_short_folders : {skipped_short}")
    if args.use_offsets_from_name:
        print(f"offset_parse_fallback : {missing_offsets} (usato x_off=y_off=0)")


if __name__ == "__main__":
    main()
