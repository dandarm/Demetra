#!/usr/bin/env python3
"""Plot comparable tracking-error histograms for selected train/val/test splits."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


DEFAULT_SPLITS = {
    "train": "tracking_inference_train_sel_ckpt1.csv",
    "val": "tracking_inference_val_selezionati_ckpt1.csv",
    "test": "tracking_inference_test_selezionati_ckpt1.csv",
}


@dataclass(frozen=True)
class SplitErrors:
    values: pd.Series
    total_rows: int
    nan_rows: int
    negative_rows: int


def _load_split_errors(csv_path: Path) -> SplitErrors:
    df = pd.read_csv(csv_path)
    if "err_km" not in df.columns:
        raise ValueError(f"{csv_path} does not contain an err_km column")
    values = pd.to_numeric(df["err_km"], errors="coerce")
    valid_values = values.dropna()
    non_negative_values = valid_values[valid_values >= 0]
    return SplitErrors(
        values=non_negative_values,
        total_rows=len(df),
        nan_rows=int(values.isna().sum()),
        negative_rows=int((valid_values < 0).sum()),
    )


def _zero_epsilon(split_errors: dict[str, SplitErrors], requested_epsilon: float | None) -> float:
    if requested_epsilon is not None:
        if requested_epsilon <= 0:
            raise ValueError("--zero-epsilon must be > 0")
        return requested_epsilon

    positive_values = pd.concat(
        [split.values[split.values > 0] for split in split_errors.values()],
        ignore_index=True,
    )
    if positive_values.empty:
        return 1e-6
    return float(positive_values.min()) * 0.5


def _shared_edges(
    split_errors: dict[str, SplitErrors],
    bins: int,
    scale: str,
    zero_epsilon: float,
    xmax_km: float | None,
) -> np.ndarray | int:
    all_values = pd.concat([split.values for split in split_errors.values()], ignore_index=True)
    if all_values.empty:
        raise ValueError("No non-negative err_km values found in the selected CSV files")

    plot_values = all_values.mask(all_values == 0, zero_epsilon)
    xmin = float(plot_values.min())
    xmax = float(xmax_km) if xmax_km is not None else float(all_values.max())
    if xmax <= xmin:
        raise ValueError(f"xmax_km must be greater than the minimum plotted value ({xmin:.6f})")
    if xmax <= 0:
        xmax = zero_epsilon * 10.0
    if scale == "log":
        return np.logspace(np.log10(xmin), np.log10(xmax), bins + 1)
    return np.linspace(0.0, xmax, bins + 1)


def _plot_hist(
    ax,
    split_errors: SplitErrors,
    *,
    title: str,
    edges: np.ndarray | int,
    scale: str,
    zero_epsilon: float,
    ymax: float | None,
    color: str = "indianred",
) -> None:
    values = split_errors.values
    plot_values = values.mask(values == 0, zero_epsilon) if scale == "log" else values
    ax.hist(plot_values, bins=edges, color=color, alpha=0.8, edgecolor="black", linewidth=0.45)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Kilometers", fontsize=11)
    ax.set_ylabel("Number of clips", fontsize=11)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    if ymax is not None:
        ax.set_ylim(0, ymax)
    ax.tick_params(axis="x", which="major", labelsize=11, length=7, width=1.2)
    ax.tick_params(axis="x", which="minor", length=4, width=0.9)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(alpha=0.2, which="both")
    if isinstance(edges, np.ndarray):
        ax.set_xlim(float(edges[0]), float(edges[-1]))
    if scale == "log":
        ax.set_xscale("log")

    median = float(values.median()) if not values.empty else float("nan")
    mean = float(values.mean()) if not values.empty else float("nan")
    ax.text(
        0.98,
        0.92,
        f"n={len(values)}\nmedian={median:.1f} km\nmean={mean:.1f} km",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
    )


def render_histograms(
    csv_dir: Path,
    output_dir: Path,
    *,
    csv_paths: dict[str, Path],
    bins: int,
    scale: str,
    zero_epsilon: float | None,
    xmax_km: float | None,
    ymax: float | None,
    dpi: int,
) -> None:
    split_errors = {
        split: _load_split_errors(path if path.is_absolute() else csv_dir / path)
        for split, path in csv_paths.items()
    }
    resolved_zero_epsilon = _zero_epsilon(split_errors, zero_epsilon)
    edges = _shared_edges(
        split_errors,
        bins=bins,
        scale=scale,
        zero_epsilon=resolved_zero_epsilon,
        xmax_km=xmax_km,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, errors in split_errors.items():
        fig, ax = plt.subplots(figsize=(7, 4), tight_layout=True)
        _plot_hist(
            ax,
            errors,
            title=f"Distribution of Tracking Error (km) - {split}",
            edges=edges,
            scale=scale,
            zero_epsilon=resolved_zero_epsilon,
            ymax=ymax,
        )
        out_png = output_dir / f"tracking_error_selected_{split}.png"
        fig.savefig(out_png, dpi=dpi)
        plt.close(fig)
        print(f"Saved {out_png}")

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(8, 9),
        sharex=True,
        tight_layout=True,
    )
    for ax, (split, errors) in zip(axes, split_errors.items()):
        _plot_hist(
            ax,
            errors,
            title=f"Distribution of Tracking Error (km) - {split}",
            edges=edges,
            scale=scale,
            zero_epsilon=resolved_zero_epsilon,
            ymax=ymax,
        )
    combined_png = output_dir / "tracking_error_selected_train_val_test.png"
    fig.savefig(combined_png, dpi=dpi)
    plt.close(fig)
    print(f"Saved {combined_png}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", type=Path, default=Path("output"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--train-csv", type=Path, default=Path(DEFAULT_SPLITS["train"]))
    parser.add_argument("--val-csv", type=Path, default=Path(DEFAULT_SPLITS["val"]))
    parser.add_argument("--test-csv", type=Path, default=Path(DEFAULT_SPLITS["test"]))
    parser.add_argument("--bins", type=int, default=40)
    parser.add_argument("--scale", choices=("log", "linear"), default="log")
    parser.add_argument(
        "--zero-epsilon",
        type=float,
        default=None,
        help="Value used to plot err_km=0 on a log x-axis. Defaults to half the minimum positive err_km.",
    )
    parser.add_argument("--xmax-km", type=float, default=400.0)
    parser.add_argument("--ymax", type=float, default=100.0)
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_histograms(
        csv_dir=args.csv_dir,
        output_dir=args.output_dir,
        csv_paths={
            "train": args.train_csv,
            "val": args.val_csv,
            "test": args.test_csv,
        },
        bins=args.bins,
        scale=args.scale,
        zero_epsilon=args.zero_epsilon,
        xmax_km=args.xmax_km,
        ymax=args.ymax,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
