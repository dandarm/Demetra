#!/usr/bin/env python3
"""Compare DeMeTrA and MeRCAD cyclone centres against Manos.

For every numeric cyclone directory in ``demetra_output``, MeRCAD points are
selected from ``medicanes_new_windows_with_mercad.csv`` using the cyclone ID
(including the 700-prefix alias). Each MeRCAD time is paired with the nearest
DeMeTrA prediction and the nearest Manos/CL7 point. Manos is consistently
used as ground truth: the two model errors are d(DeMeTrA, Manos) and
d(MeRCAD, Manos).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import ConnectionPatch


DEFAULT_INPUT = Path(__file__).with_name("medicanes_new_windows_with_mercad.csv")
DEFAULT_DEMETRA_ROOT = Path("/media/isacDisk2/demetra_output")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("batch_outputs")
EARTH_RADIUS_KM = 6371.0088


def normalize_id(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


def id_aliases(cyclone_id: str) -> set[str]:
    """Return both the original ID and its with/without-700 counterpart."""
    cyclone_id = normalize_id(cyclone_id)
    aliases = {cyclone_id}
    if cyclone_id.startswith("700") and len(cyclone_id) > 3:
        aliases.add(cyclone_id[3:])
    elif cyclone_id.isdigit():
        aliases.add("700" + cyclone_id)
    return aliases


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return float(2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a)))


def numeric_cyclone_dirs(root: Path) -> list[Path]:
    return sorted(
        (path for path in root.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: path.name,
    )


def load_reference(input_csv: Path) -> pd.DataFrame:
    frame = pd.read_csv(input_csv, dtype={"id_cyc_unico": "string"})
    required = {"id_cyc_unico", "time", "lat", "lon", "mercad_lat", "mercad_lon"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Colonne mancanti in {input_csv}: {sorted(missing)}")

    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    for column in ("lat", "lon", "mercad_lat", "mercad_lon"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ("id_cyc_unico", "id_final", "idorig"):
        if column not in frame:
            frame[column] = ""
        frame[column + "_norm"] = frame[column].map(normalize_id)
    return frame


def select_cyclone(frame: pd.DataFrame, cyclone_id: str) -> pd.DataFrame:
    aliases = id_aliases(cyclone_id)
    mask = pd.Series(False, index=frame.index)
    for column in ("id_cyc_unico_norm", "id_final_norm", "idorig_norm"):
        mask |= frame[column].isin(aliases)
    return frame.loc[mask].copy()


def load_demetra(prediction_csv: Path) -> pd.DataFrame:
    frame = pd.read_csv(prediction_csv)
    required = {"datetime", "has_cyclone", "pred_lat", "pred_lon"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Colonne mancanti in {prediction_csv}: {sorted(missing)}")

    frame["demetra_time"] = pd.to_datetime(frame["datetime"], errors="coerce")
    frame["has_cyclone"] = pd.to_numeric(frame["has_cyclone"], errors="coerce").fillna(0).astype(int)
    frame["demetra_lat"] = pd.to_numeric(frame["pred_lat"], errors="coerce")
    frame["demetra_lon"] = pd.to_numeric(frame["pred_lon"], errors="coerce")
    frame = frame.loc[frame["has_cyclone"].eq(1)].dropna(
        subset=["demetra_time", "demetra_lat", "demetra_lon"]
    )
    return frame.sort_values("demetra_time").drop_duplicates("demetra_time", keep="first")


def nearest_row(frame: pd.DataFrame, time_column: str, target: pd.Timestamp) -> tuple[pd.Series | None, float]:
    if frame.empty:
        return None, np.nan
    delta_minutes = (frame[time_column] - target).abs().dt.total_seconds() / 60.0
    index = delta_minutes.idxmin()
    return frame.loc[index], float(delta_minutes.loc[index])


def build_comparison(
    reference: pd.DataFrame,
    demetra_root: Path,
    demetra_tolerance_minutes: float,
    manos_tolerance_minutes: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for cyclone_dir in numeric_cyclone_dirs(demetra_root):
        cyclone_id = cyclone_dir.name
        selected = select_cyclone(reference, cyclone_id)
        mercad = selected.dropna(subset=["time", "mercad_lat", "mercad_lon"]).sort_values("time")
        manos = (
            selected.dropna(subset=["time", "lat", "lon"])
            .sort_values("time")
            .drop_duplicates("time", keep="first")
        )

        prediction_csv = cyclone_dir / "tracking_inference_predictions.csv"
        demetra = load_demetra(prediction_csv) if prediction_csv.exists() else pd.DataFrame()
        cyclone_details: list[dict[str, object]] = []

        for observation_order, (_, reference_row) in enumerate(mercad.iterrows(), start=1):
            ref_time = reference_row["time"]
            demetra_row, demetra_delta = nearest_row(demetra, "demetra_time", ref_time)
            manos_row, manos_delta = nearest_row(manos, "time", ref_time)
            demetra_ok = demetra_row is not None and demetra_delta <= demetra_tolerance_minutes
            manos_ok = manos_row is not None and manos_delta <= manos_tolerance_minutes
            common_match = bool(demetra_ok and manos_ok)

            output: dict[str, object] = {
                "cyclone_id": cyclone_id,
                "observation_order": observation_order,
                "mercad_time": ref_time,
                "mercad_lat": reference_row["mercad_lat"],
                "mercad_lon": reference_row["mercad_lon"],
                "demetra_time": demetra_row["demetra_time"] if demetra_row is not None else pd.NaT,
                "demetra_lat": demetra_row["demetra_lat"] if demetra_row is not None else np.nan,
                "demetra_lon": demetra_row["demetra_lon"] if demetra_row is not None else np.nan,
                "demetra_delta_minutes": demetra_delta,
                "manos_time": manos_row["time"] if manos_row is not None else pd.NaT,
                "manos_lat": manos_row["lat"] if manos_row is not None else np.nan,
                "manos_lon": manos_row["lon"] if manos_row is not None else np.nan,
                "manos_delta_minutes": manos_delta,
                "common_match": common_match,
                "demetra_to_mercad_km": np.nan,
                "manos_to_mercad_km": np.nan,
                "demetra_error_km": np.nan,
                "mercad_error_km": np.nan,
                "error_delta_km": np.nan,
                "error_pair_mean_km": np.nan,
            }
            if common_match:
                output["demetra_to_mercad_km"] = haversine_km(
                    demetra_row["demetra_lat"], demetra_row["demetra_lon"],
                    reference_row["mercad_lat"], reference_row["mercad_lon"],
                )
                output["manos_to_mercad_km"] = haversine_km(
                    manos_row["lat"], manos_row["lon"],
                    reference_row["mercad_lat"], reference_row["mercad_lon"],
                )
                output["pair_mean_km"] = (
                    output["demetra_to_mercad_km"] + output["manos_to_mercad_km"]
                ) / 2.0
                output["pair_signed_delta_km"] = (
                    output["demetra_to_mercad_km"] - output["manos_to_mercad_km"]
                )
                output["demetra_to_manos_km"] = haversine_km(
                    demetra_row["demetra_lat"], demetra_row["demetra_lon"],
                    manos_row["lat"], manos_row["lon"],
                )
                output["mercad_to_manos_km"] = haversine_km(
                    reference_row["mercad_lat"], reference_row["mercad_lon"],
                    manos_row["lat"], manos_row["lon"],
                )
                output["demetra_manos_minus_mercad_manos_km"] = (
                    output["demetra_to_manos_km"] - output["mercad_to_manos_km"]
                )
                output["demetra_manos_mercad_manos_mean_km"] = (
                    output["demetra_to_manos_km"] + output["mercad_to_manos_km"]
                ) / 2.0
                # Canonical model errors: Manos is the ground-truth reference.
                output["demetra_error_km"] = output["demetra_to_manos_km"]
                output["mercad_error_km"] = output["mercad_to_manos_km"]
                output["error_delta_km"] = (
                    output["demetra_error_km"] - output["mercad_error_km"]
                )
                output["error_pair_mean_km"] = (
                    output["demetra_error_km"] + output["mercad_error_km"]
                ) / 2.0
            else:
                output["pair_mean_km"] = np.nan
                output["pair_signed_delta_km"] = np.nan
                output["demetra_to_manos_km"] = np.nan
                output["mercad_to_manos_km"] = np.nan
                output["demetra_manos_minus_mercad_manos_km"] = np.nan
                output["demetra_manos_mercad_manos_mean_km"] = np.nan
            cyclone_details.append(output)
            detail_rows.append(output)

        common = pd.DataFrame(cyclone_details)
        if not common.empty:
            common = common.loc[common["common_match"]]
        if common.empty:
            demetra_errors = pd.Series(dtype=float)
            mercad_errors = pd.Series(dtype=float)
        else:
            demetra_errors = pd.to_numeric(common["demetra_error_km"], errors="coerce").dropna()
            mercad_errors = pd.to_numeric(common["mercad_error_km"], errors="coerce").dropna()
        summary_rows.append(
            {
                "cyclone_id": cyclone_id,
                "n_mercad": len(mercad),
                "n_common_matches": len(common),
                "demetra_mean_km": demetra_errors.mean(),
                "demetra_median_km": demetra_errors.median(),
                "mercad_mean_km": mercad_errors.mean(),
                "mercad_median_km": mercad_errors.median(),
            }
        )

    return pd.DataFrame(detail_rows), pd.DataFrame(summary_rows)


def render_plot(
    detail: pd.DataFrame,
    summary: pd.DataFrame,
    output_png: Path,
    series_specs: tuple[tuple[str, str, str, str], ...] | None = None,
    title: str = "Cyclone-centre errors relative to Manos",
    ylabel: str = "Error relative to Manos (km)",
) -> None:
    cyclone_ids = summary["cyclone_id"].tolist()
    matched = detail.loc[detail["common_match"]].copy() if not detail.empty else detail.copy()
    cyclone_order = pd.Categorical(matched["cyclone_id"], categories=cyclone_ids, ordered=True)
    matched = (
        matched.assign(_cyclone_order=cyclone_order)
        .sort_values(["_cyclone_order", "mercad_time"], kind="stable")
        .drop(columns="_cyclone_order")
    )

    # Give every MeRCAD observation its own x coordinate.  A block is reserved
    # even for IDs without valid matches so all 18 numeric directories remain visible.
    group_gap = 0.75
    cursor = 0.0
    group_layout: dict[str, dict[str, float]] = {}
    matched["observation_x"] = np.nan
    for cyclone_id in cyclone_ids:
        group_index = matched.index[matched["cyclone_id"].eq(cyclone_id)]
        width = float(max(len(group_index), 1))
        left = cursor
        right = cursor + width
        center = (left + right) / 2.0
        if len(group_index):
            matched.loc[group_index, "observation_x"] = left + np.arange(len(group_index)) + 0.5
        group_layout[cyclone_id] = {"left": left, "right": right, "center": center}
        cursor = right + group_gap

    if series_specs is None:
        series_specs = (
            ("demetra_error_km", "#0072B2", "o", "DeMeTrA error"),
            ("mercad_error_km", "#D55E00", "^", "MeRCAD error"),
        )

    fig, ax = plt.subplots(figsize=(18, 8.2))
    for column, color, marker, label in series_specs:
        x = matched["observation_x"]
        ax.scatter(x, matched[column], s=48, alpha=0.58, color=color, marker=marker,
                   edgecolors="white", linewidths=0.65, label=label, zorder=3)

    ax.set_xticks([])
    ax.set_xlim(-0.25, cursor - group_gap + 0.25)
    ax.set_ylim(bottom=0)
    for previous_id, next_id in zip(cyclone_ids[:-1], cyclone_ids[1:]):
        boundary = (
            group_layout[previous_id]["right"] + group_layout[next_id]["left"]
        ) / 2.0
        ax.axvline(boundary, color="0.55", linestyle="--", linewidth=0.85,
                   alpha=0.7, zorder=1)
    axis_transform = ax.get_xaxis_transform()
    for cyclone_id in cyclone_ids:
        layout = group_layout[cyclone_id]
        left = layout["left"] + 0.08
        right = layout["right"] - 0.08
        center = layout["center"]
        ax.plot([left, left, right, right], [-0.065, -0.095, -0.095, -0.065],
                transform=axis_transform, clip_on=False, color="0.3", linewidth=0.9)
        ax.text(center, -0.115, cyclone_id, transform=axis_transform, rotation=45,
                ha="right", va="top", fontsize=9, color="0.2", clip_on=False)
        if summary.loc[summary["cyclone_id"].eq(cyclone_id), "n_common_matches"].iloc[0] == 0:
            ax.text(center, 0.018, "n.d.", transform=axis_transform, ha="center",
                    va="bottom", color="0.45", fontsize=8)
    ax.set_xlabel("Osservazioni raggruppate per ID ciclone", labelpad=82)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axhline(28.0, color="black", linewidth=1.1, zorder=2)
    ax.grid(axis="y", color="0.85", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", ncols=2, frameon=True)
    fig.tight_layout()
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_sorted_pairs_plot(
    detail: pd.DataFrame,
    output_png: Path,
    sort_column: str = "error_pair_mean_km",
    title: str = "DeMeTrA and MeRCAD errors relative to Manos, sorted by pair",
    xlabel: str = "Pairs sorted by decreasing mean error",
    ylabel: str = "Error relative to Manos (km)",
    series_specs: tuple[tuple[str, str, str, str], ...] | None = None,
) -> None:
    """Plot all matched pairs sorted by a decreasing pair-level metric."""
    matched = detail.loc[detail["common_match"]].copy()
    if matched.empty:
        raise ValueError("Nessuna coppia comune disponibile per il grafico ordinato.")
    matched = matched.sort_values(
        [sort_column, "mercad_time"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    matched["pair_x"] = np.arange(len(matched), dtype=float)

    if series_specs is None:
        series_specs = (
            ("demetra_error_km", "#0072B2", "o", "DeMeTrA error"),
            ("mercad_error_km", "#D55E00", "^", "MeRCAD error"),
        )

    fig, ax = plt.subplots(figsize=(18, 8.2))
    for column, color, marker, label in series_specs:
        ax.scatter(
            matched["pair_x"], matched[column], s=48, alpha=0.58, color=color,
            marker=marker, edgecolors="white", linewidths=0.65, label=label, zorder=3,
        )

    # Separators emphasize that each x position contains one DeMeTrA/Manos pair.
    for boundary in np.arange(0.5, len(matched), 1.0):
        ax.axvline(boundary, color="0.55", linestyle="--", linewidth=0.7,
                   alpha=0.65, zorder=1)
    ax.axhline(28.0, color="black", linewidth=1.1, zorder=2)
    ax.set_xlim(-0.75, len(matched) - 0.25)
    ax.set_ylim(bottom=0)
    ax.set_xticks([])
    ax.set_xlabel(xlabel, labelpad=18)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", color="0.85", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", ncols=2, frameon=True)
    fig.tight_layout()
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_signed_difference_plot(detail: pd.DataFrame, output_png: Path) -> None:
    """Plot Δ = error(DeMeTrA) - error(MeRCAD), with Manos as ground truth."""
    matched = detail.loc[detail["common_match"]].copy()
    if matched.empty:
        raise ValueError("Nessuna coppia comune disponibile per il grafico delle differenze.")
    matched = matched.sort_values(
        ["error_delta_km", "mercad_time"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    matched["pair_x"] = np.arange(len(matched), dtype=float)
    positive = matched["error_delta_km"].ge(0.0)

    fig, ax = plt.subplots(figsize=(18, 8.2))
    ax.scatter(
        matched.loc[positive, "pair_x"], matched.loc[positive, "error_delta_km"],
        s=50, color="#0072B2", alpha=0.7, edgecolors="white", linewidths=0.65,
        label="Δ ≥ 0: DeMeTrA error larger", zorder=3,
    )
    ax.scatter(
        matched.loc[~positive, "pair_x"], matched.loc[~positive, "error_delta_km"],
        s=50, color="#D55E00", alpha=0.7, edgecolors="white", linewidths=0.65,
        label="Δ < 0: MeRCAD error larger", zorder=3,
    )
    for boundary in np.arange(0.5, len(matched), 1.0):
        ax.axvline(boundary, color="0.55", linestyle="--", linewidth=0.7,
                   alpha=0.65, zorder=1)
    ax.axhline(0.0, color="black", linewidth=1.1, zorder=2)
    ax.set_xlim(-0.75, len(matched) - 0.25)
    ax.set_xticks([])
    ax.set_xlabel("Pairs sorted by decreasing Δ", labelpad=18)
    ax.set_ylabel("Error difference (km): DeMeTrA − MeRCAD")
    ax.set_title("Paired difference in errors relative to Manos")
    ax.grid(axis="y", color="0.85", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_demetra_manos_vs_mercad_manos_plot(detail: pd.DataFrame, output_png: Path) -> None:
    """Compare DeMeTrA–Manos and MeRCAD–Manos distances for each matched pair."""
    matched = detail.loc[detail["common_match"]].copy()
    if matched.empty:
        raise ValueError("Nessuna coppia comune disponibile per il grafico Manos.")
    matched = matched.sort_values(
        ["demetra_to_manos_km", "mercad_time"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    matched["pair_x"] = np.arange(len(matched), dtype=float)

    fig, ax = plt.subplots(figsize=(18, 8.2))
    for column, color, marker, label in (
        ("demetra_error_km", "#0072B2", "o", "DeMeTrA error"),
        ("mercad_error_km", "#D55E00", "^", "MeRCAD error"),
    ):
        ax.scatter(
            matched["pair_x"], matched[column], s=48, alpha=0.58, color=color,
            marker=marker, edgecolors="white", linewidths=0.65, label=label, zorder=3,
        )
    for boundary in np.arange(0.5, len(matched), 1.0):
        ax.axvline(boundary, color="0.55", linestyle="--", linewidth=0.7,
                   alpha=0.65, zorder=1)
    ax.axhline(28.0, color="black", linewidth=1.1, zorder=2)
    ax.set_xlim(-0.75, len(matched) - 0.25)
    ax.set_ylim(bottom=0)
    ax.set_xticks([])
    ax.set_xlabel("Pairs sorted by decreasing DeMeTrA error", labelpad=18)
    ax.set_ylabel("Error relative to Manos (km)")
    ax.set_title("DeMeTrA and MeRCAD errors relative to Manos")
    ax.grid(axis="y", color="0.85", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", ncols=2, frameon=True)
    fig.tight_layout()
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_demetra_manos_difference_plot(detail: pd.DataFrame, output_png: Path) -> None:
    """Plot Δ = DeMeTrA error - MeRCAD error, with Manos as ground truth."""
    matched = detail.loc[detail["common_match"]].copy()
    if matched.empty:
        raise ValueError("Nessuna coppia comune disponibile per il grafico delle differenze Manos.")
    delta_column = "error_delta_km"
    matched = matched.sort_values(
        [delta_column, "mercad_time"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    matched["pair_x"] = np.arange(len(matched), dtype=float)
    positive = matched[delta_column].ge(0.0)
    matched["plot_delta_km"] = matched[delta_column].abs()

    fig, ax = plt.subplots(figsize=(18, 8.2))
    ax.scatter(
        matched.loc[positive, "pair_x"], matched.loc[positive, "plot_delta_km"],
        s=50, color="#0072B2", alpha=0.7, edgecolors="white", linewidths=0.65,
        label="Positive Δ: DeMeTrA error larger", zorder=3,
    )
    ax.scatter(
        matched.loc[~positive, "pair_x"], matched.loc[~positive, "plot_delta_km"],
        s=50, color="#D55E00", alpha=0.7, edgecolors="white", linewidths=0.65,
        label="Negative Δ: MeRCAD error larger", zorder=3,
    )
    for boundary in np.arange(0.5, len(matched), 1.0):
        ax.axvline(boundary, color="0.55", linestyle="--", linewidth=0.7,
                   alpha=0.65, zorder=1)
    ax.axhline(0.0, color="black", linewidth=1.1, zorder=2)
    ax.set_xlim(-0.75, len(matched) - 0.25)
    ax.set_xticks([])
    ax.set_xlabel("Pairs sorted by decreasing signed Δ", labelpad=18)
    ax.set_ylabel("Difference magnitude |Δ| (km)")
    ax.set_title("Difference between DeMeTrA and MeRCAD errors relative to Manos")
    ax.grid(axis="y", color="0.85", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", frameon=True)
    positive_mean = matched.loc[positive, delta_column].mean()
    negative_mean = matched.loc[~positive, delta_column].mean()
    ax.text(
        0.98, 0.98,
        f"Mean positive Δ: {positive_mean:+.2f} km\n"
        f"Mean negative Δ: {negative_mean:+.2f} km",
        transform=ax.transAxes, ha="right", va="top", fontsize=10,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "0.5", "alpha": 0.92},
    )
    fig.tight_layout()
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_paired_error_scatter(detail: pd.DataFrame, output_png: Path, output_pdf: Path) -> None:
    """Paired scatter of MeRCAD and DeMeTrA errors relative to Manos."""
    paired = detail.loc[
        detail["common_match"], ["mercad_error_km", "demetra_error_km"]
    ].dropna().copy()
    if paired.empty:
        raise ValueError("Nessuna coppia valida disponibile per il confronto paired.")

    mercad_error_km = paired["mercad_error_km"].to_numpy(dtype=float)
    demetra_error_km = paired["demetra_error_km"].to_numpy(dtype=float)
    delta_km = demetra_error_km - mercad_error_km
    above_diagonal = delta_km > 0.0
    # Display one decimal place by truncating (rather than rounding) the
    # conditional mean absolute differences shown in the legend.
    mean_abs_delta_demetra = np.trunc(np.abs(delta_km[above_diagonal]).mean() * 10.0) / 10.0
    mean_abs_delta_mercad = np.trunc(np.abs(delta_km[~above_diagonal]).mean() * 10.0) / 10.0
    upper = max(float(mercad_error_km.max()), float(demetra_error_km.max()))
    upper = max(1.0, upper * 1.05)

    # Use a larger canvas and explicit typography so the exported figure is
    # legible at publication scale and when enlarged.
    fig, ax = plt.subplots(figsize=(10.0, 9.5))
    ax.scatter(
        mercad_error_km[above_diagonal], demetra_error_km[above_diagonal],
        s=48, alpha=0.72, color="#D55E00", edgecolors="white", linewidths=0.65,
        label=(
            f"DeMeTrA error higher (n={above_diagonal.sum()}; "
            f"mean |Δ|={mean_abs_delta_demetra:.1f} km)"
        ), zorder=3,
    )
    ax.scatter(
        mercad_error_km[~above_diagonal], demetra_error_km[~above_diagonal],
        s=48, alpha=0.72, color="#0072B2", edgecolors="white", linewidths=0.65,
        label=(
            f"MeRCAD error higher (n={(~above_diagonal).sum()}; "
            f"mean |Δ|={mean_abs_delta_mercad:.1f} km)"
        ), zorder=3,
    )
    ax.plot([0.0, upper], [0.0, upper], color="black", linestyle="--",
            linewidth=1.15, label="y = x", zorder=2)
    ax.set_xlim(0.0, upper)
    ax.set_ylim(0.0, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("MeRCAD error (km)", fontsize=17)
    ax.set_ylabel("DeMeTrA error (km)", fontsize=17)
    ax.set_title(
        "Paired comparison of DeMeTrA and MeRCAD cyclone-centre errors",
        fontsize=19,
        pad=14,
    )
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(color="0.86", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=True, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_png, dpi=1200, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def build_long_error_table(detail: pd.DataFrame) -> pd.DataFrame:
    """Return paired errors in long format with model and error_km columns."""
    paired = detail.loc[
        detail["common_match"], ["mercad_error_km", "demetra_error_km"]
    ].dropna()
    return pd.DataFrame(
        {
            "model": ["mercad"] * len(paired) + ["demetra"] * len(paired),
            "error_km": pd.concat(
                [paired["mercad_error_km"], paired["demetra_error_km"]],
                ignore_index=True,
            ).to_numpy(dtype=float),
        }
    )


def render_error_distribution_plot(
    error_long: pd.DataFrame, output_png: Path, output_pdf: Path
) -> None:
    """Plot model error distributions as boxplots with all observations overlaid."""
    order = ["mercad", "demetra"]
    values = [
        error_long.loc[error_long["model"].eq(model), "error_km"].to_numpy(dtype=float)
        for model in order
    ]
    if any(len(values_for_model) == 0 for values_for_model in values):
        raise ValueError("Distribuzione errori incompleta per il box/strip plot.")

    fig, ax = plt.subplots(figsize=(7.8, 7.2))
    box = ax.boxplot(
        values, positions=[1, 2], widths=0.48, patch_artist=True, showfliers=False,
        medianprops={"color": "black", "linewidth": 1.4},
        whiskerprops={"color": "0.3", "linewidth": 1.0},
        capprops={"color": "0.3", "linewidth": 1.0},
    )
    for patch, color in zip(box["boxes"], ["#D55E00", "#0072B2"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.34)
        patch.set_edgecolor(color)

    rng = np.random.default_rng(20260719)
    for position, model, color in zip([1, 2], order, ["#D55E00", "#0072B2"]):
        y = error_long.loc[error_long["model"].eq(model), "error_km"].to_numpy(dtype=float)
        jitter = rng.uniform(-0.12, 0.12, size=len(y))
        ax.scatter(
            position + jitter, y, s=42, alpha=0.70, color=color,
            edgecolors="white", linewidths=0.55, zorder=3,
        )

    ax.set_xticks([1, 2], labels=["mercad", "demetra"])
    ax.set_ylabel("Tracking error (km)")
    ax.set_xlabel("Model")
    ax.set_title("Distribution of cyclone-centre tracking errors relative to Manos")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="0.86", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_png, dpi=600, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def render_sorted_scatter_consistency_plot(
    detail: pd.DataFrame, output_png: Path, output_pdf: Path
) -> None:
    """Link each sorted error-difference point to its paired-scatter counterpart."""
    delta_column = "error_delta_km"
    paired = detail.loc[
        detail["common_match"], ["mercad_error_km", "demetra_error_km", delta_column, "mercad_time"]
    ].dropna().sort_values([delta_column, "mercad_time"], ascending=[False, True], kind="stable")
    if paired.empty:
        raise ValueError("Nessuna coppia valida disponibile per il grafico di coerenza.")
    paired = paired.reset_index(drop=True)
    paired["rank"] = np.arange(1, len(paired) + 1, dtype=float)
    paired["difference_magnitude_km"] = paired[delta_column].abs()
    positive = paired[delta_column].ge(0.0)
    colors = np.where(positive, "#0072B2", "#D55E00")
    upper = max(
        float(paired["mercad_error_km"].max()),
        float(paired["demetra_error_km"].max()),
    ) * 1.05

    fig, (ax_sorted, ax_scatter) = plt.subplots(
        2, 1, figsize=(12.5, 15.5), gridspec_kw={"hspace": 0.34}
    )

    for is_positive, color, label in (
        (True, "#0072B2", "Positive Δ: DeMeTrA error larger"),
        (False, "#D55E00", "Negative Δ: MeRCAD error larger"),
    ):
        subset = paired.loc[positive.eq(is_positive)]
        ax_sorted.scatter(
            subset["rank"], subset["difference_magnitude_km"], s=48, color=color,
            alpha=0.72, edgecolors="white", linewidths=0.65, label=label, zorder=3,
        )
        ax_scatter.scatter(
            subset["mercad_error_km"], subset["demetra_error_km"], s=48, color=color,
            alpha=0.72, edgecolors="white", linewidths=0.65, zorder=3,
        )

    for boundary in np.arange(1.5, len(paired), 1.0):
        ax_sorted.axvline(boundary, color="0.65", linestyle="--", linewidth=0.55,
                          alpha=0.5, zorder=1)
    ax_sorted.axhline(0.0, color="black", linewidth=1.0, zorder=2)
    ax_sorted.set_xlim(0.25, len(paired) + 0.75)
    ax_sorted.set_ylim(bottom=0)
    ax_sorted.set_xlabel("Pair rank (decreasing signed Δ)")
    ax_sorted.set_ylabel("Difference magnitude |Δ| (km)")
    ax_sorted.set_title("Sorted paired error differences")
    ax_sorted.grid(axis="y", color="0.86", linewidth=0.8)
    ax_sorted.set_axisbelow(True)
    ax_sorted.legend(loc="upper left", frameon=True)

    ax_scatter.plot([0.0, upper], [0.0, upper], color="black", linestyle="--",
                    linewidth=1.15, label="y = x", zorder=2)
    ax_scatter.set_xlim(0.0, upper)
    ax_scatter.set_ylim(0.0, upper)
    ax_scatter.set_aspect("equal", adjustable="box")
    ax_scatter.set_xlabel("MeRCAD error relative to Manos (km)")
    ax_scatter.set_ylabel("DeMeTrA error relative to Manos (km)")
    ax_scatter.set_title("Paired error comparison")
    ax_scatter.grid(color="0.86", linewidth=0.8)
    ax_scatter.set_axisbelow(True)
    ax_scatter.legend(loc="upper left", frameon=True)

    fig.suptitle("One-to-one consistency between sorted differences and paired errors", y=0.985)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.06, top=0.92, hspace=0.34)

    # Draw links last so they remain visible over the white axes backgrounds.
    # Blue links are rendered first, then orange links, as requested.
    for is_positive, color in ((True, "#0072B2"), (False, "#D55E00")):
        for _, row in paired.loc[positive.eq(is_positive)].iterrows():
            link = ConnectionPatch(
                xyA=(row["rank"], row["difference_magnitude_km"]), coordsA=ax_sorted.transData,
                xyB=(row["mercad_error_km"], row["demetra_error_km"]), coordsB=ax_scatter.transData,
                color=color, alpha=0.42, linewidth=1.05, zorder=20, clip_on=False,
            )
            fig.add_artist(link)
    fig.savefig(output_png, dpi=600, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--demetra-root", type=Path, default=DEFAULT_DEMETRA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--demetra-tolerance-minutes", type=float, default=5.0)
    parser.add_argument("--manos-tolerance-minutes", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail, summary = build_comparison(
        load_reference(args.input_csv), args.demetra_root,
        args.demetra_tolerance_minutes, args.manos_tolerance_minutes,
    )
    detail_csv = args.output_dir / "demetra_manos_vs_mercad_distances.csv"
    summary_csv = args.output_dir / "demetra_manos_vs_mercad_summary.csv"
    output_png = args.output_dir / "demetra_manos_vs_mercad_distances.png"
    sorted_output_png = args.output_dir / "demetra_manos_vs_mercad_pairs_sorted.png"
    signed_sorted_output_png = args.output_dir / "demetra_manos_vs_mercad_pairs_sorted_signed_delta.png"
    difference_output_png = args.output_dir / "demetra_minus_manos_distance_sorted.png"
    manos_distance_output_png = args.output_dir / "demetra_manos_vs_mercad_manos_distances.png"
    manos_difference_output_png = args.output_dir / "demetra_manos_minus_mercad_manos_sorted.png"
    manos_by_cyclone_output_png = args.output_dir / "demetra_manos_vs_mercad_manos_by_cyclone.png"
    manos_pairs_sorted_output_png = args.output_dir / "demetra_manos_vs_mercad_manos_pairs_sorted.png"
    paired_error_png = args.output_dir / "paired_demetra_vs_manos_mercad_error.png"
    paired_error_pdf = args.output_dir / "paired_demetra_vs_manos_mercad_error.pdf"
    long_error_csv = args.output_dir / "paired_model_errors_long.csv"
    error_distribution_png = args.output_dir / "paired_model_error_distribution.png"
    error_distribution_pdf = args.output_dir / "paired_model_error_distribution.pdf"
    consistency_png = args.output_dir / "sorted_vs_paired_error_consistency.png"
    consistency_pdf = args.output_dir / "sorted_vs_paired_error_consistency.pdf"
    detail.to_csv(detail_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    render_plot(detail, summary, output_png)
    render_sorted_pairs_plot(detail, sorted_output_png)
    render_sorted_pairs_plot(
        detail,
        signed_sorted_output_png,
        sort_column="error_delta_km",
        title="DeMeTrA and MeRCAD errors sorted by signed difference",
        xlabel="Pairs sorted by decreasing (DeMeTrA error − MeRCAD error)",
    )
    render_signed_difference_plot(detail, difference_output_png)
    render_demetra_manos_vs_mercad_manos_plot(detail, manos_distance_output_png)
    render_demetra_manos_difference_plot(detail, manos_difference_output_png)
    manos_reference_specs = (
        ("demetra_error_km", "#0072B2", "o", "DeMeTrA error"),
        ("mercad_error_km", "#D55E00", "^", "MeRCAD error"),
    )
    render_plot(
        detail,
        summary,
        manos_by_cyclone_output_png,
        series_specs=manos_reference_specs,
        title="DeMeTrA and MeRCAD errors relative to Manos by cyclone ID",
        ylabel="Error relative to Manos (km)",
    )
    render_sorted_pairs_plot(
        detail,
        manos_pairs_sorted_output_png,
        sort_column="error_pair_mean_km",
        title="DeMeTrA and MeRCAD errors relative to Manos, sorted by pair",
        xlabel="Pairs sorted by decreasing mean error",
        ylabel="Error relative to Manos (km)",
        series_specs=manos_reference_specs,
    )
    render_paired_error_scatter(detail, paired_error_png, paired_error_pdf)
    error_long = build_long_error_table(detail)
    error_long.to_csv(long_error_csv, index=False)
    render_error_distribution_plot(error_long, error_distribution_png, error_distribution_pdf)
    render_sorted_scatter_consistency_plot(detail, consistency_png, consistency_pdf)

    print(f"Cartelle numeriche: {len(summary)}")
    print(f"Punti MeRCAD abbinati a entrambi i metodi: {int(summary['n_common_matches'].sum())}")
    print(summary.to_string(index=False))
    print(
        f"Salvati: {detail_csv}, {summary_csv}, {output_png}, "
        f"{sorted_output_png}, {signed_sorted_output_png}, {difference_output_png}, "
        f"{manos_distance_output_png}, {manos_difference_output_png}, "
        f"{manos_by_cyclone_output_png}, {manos_pairs_sorted_output_png}, "
        f"{paired_error_png}, {paired_error_pdf}, {long_error_csv}, "
        f"{error_distribution_png}, {error_distribution_pdf}, {consistency_png}, {consistency_pdf}"
    )


if __name__ == "__main__":
    main()
