#!/usr/bin/env python3
"""Build random unsupervised videotiles from terminal.

CLI arguments are intentionally minimal:
1) input_dir
2) output_dir

All other dataset-generation parameters are fixed to the defaults validated in
`notebooks/random_tiles.ipynb`.
"""

import argparse
import os
from pathlib import Path

from dataset.build_dataset import make_unsup_dataset


# Defaults aligned with notebooks/random_tiles.ipynb
NUM_FRAMES = 16
RANDOM_SEED = None
MAX_GAP_MINUTES = 60
NUM_RANDOM_OFFSETS_PER_WINDOW = 10
MAX_VIDEOS = None
SAVE_ONE_EXAMPLE_TILE = False

ASYNC_TEMPORAL_SAMPLING = True
TEMPORAL_START_JITTER_WINDOWS = 16
MIN_TEMPORAL_GAP_WINDOWS = 6
MAX_TEMPORAL_GAP_WINDOWS = 6

APPLY_OVERLAP_CONSTRAINT = True
MAX_IOU_ACTIVE = 0.20
APPLY_COVERAGE_BIAS = True
COVERAGE_WEIGHT = 1.0
BORDER_BOOST = 0.25
SAMPLING_ATTEMPTS = 64

# Keep the parallelism modest: this pipeline can quickly become I/O-bound.
SAVE_NUM_WORKERS = max(1, min(4, os.cpu_count() or 1))
SAVE_NUM_WORKERS = max(1, min(4, os.cpu_count() or 1))

DRY_RUN = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build random unsupervised videotiles with overlap+coverage controls."
    )
    # Support both positional args and --input_dir/--output_dir flags.
    parser.add_argument("input_dir", nargs="?", type=str, help="Folder with source AirmassRGB .png frames")
    parser.add_argument("output_dir", nargs="?", type=str, help="Destination folder for videotile folders")
    parser.add_argument("--input_dir", dest="input_dir_opt", type=str, help="Folder with source AirmassRGB .png frames")
    parser.add_argument("--output_dir", dest="output_dir_opt", type=str, help="Destination folder for videotile folders")
    args = parser.parse_args()

    input_dir = args.input_dir_opt if args.input_dir_opt is not None else args.input_dir
    output_dir = args.output_dir_opt if args.output_dir_opt is not None else args.output_dir
    if input_dir is None or output_dir is None:
        parser.error("Devi specificare input_dir e output_dir (posizionali o con --input_dir/--output_dir).")
    args.input_dir = input_dir
    args.output_dir = output_dir
    return args


def main():
    args = parse_args()
    input_dir = str(Path(args.input_dir).expanduser())
    output_dir = str(Path(args.output_dir).expanduser())

    print("Random unsupervised dataset builder")
    print(f"input_dir: {input_dir}")
    print(f"output_dir: {output_dir}")
    print(f"num_frames: {NUM_FRAMES}")
    print(f"num_random_offsets_per_window: {NUM_RANDOM_OFFSETS_PER_WINDOW}")
    print(f"overlap_constraint: {APPLY_OVERLAP_CONSTRAINT} (max_iou_active={MAX_IOU_ACTIVE})")
    print(
        "coverage_bias: "
        f"{APPLY_COVERAGE_BIAS} (coverage_weight={COVERAGE_WEIGHT}, border_boost={BORDER_BOOST})"
    )
    print(f"save_num_workers: {SAVE_NUM_WORKERS}")
    print(f"dry_run: {DRY_RUN}")

    df = make_unsup_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        random_offsets=True,
        dry_run=DRY_RUN,
        save_example_tile=SAVE_ONE_EXAMPLE_TILE,
        random_seed=RANDOM_SEED,
        num_frames=NUM_FRAMES,
        max_gap_minutes=MAX_GAP_MINUTES,
        num_random_offsets_per_window=NUM_RANDOM_OFFSETS_PER_WINDOW,
        asynchronous_temporal_sampling=ASYNC_TEMPORAL_SAMPLING,
        temporal_start_jitter_windows=TEMPORAL_START_JITTER_WINDOWS,
        min_temporal_gap_windows=MIN_TEMPORAL_GAP_WINDOWS,
        max_temporal_gap_windows=MAX_TEMPORAL_GAP_WINDOWS,
        apply_overlap_constraint=APPLY_OVERLAP_CONSTRAINT,
        max_iou_active=MAX_IOU_ACTIVE,
        apply_coverage_bias=APPLY_COVERAGE_BIAS,
        coverage_weight=COVERAGE_WEIGHT,
        border_boost=BORDER_BOOST,
        sampling_attempts=SAMPLING_ATTEMPTS,
        save_num_workers=SAVE_NUM_WORKERS,
        max_videos=MAX_VIDEOS,
    )
    print(f"Completed. videos: {df.shape[0]}")


if __name__ == "__main__":
    main()
