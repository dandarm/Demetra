import ast
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from mpl_toolkits.basemap import Basemap


REPO_ROOT = Path(__file__).resolve().parents[1]
VIDEOMAE_DIR = REPO_ROOT / "moduli" / "videomae"
if str(VIDEOMAE_DIR) not in sys.path:
    sys.path.insert(0, str(VIDEOMAE_DIR))

from medicane_utils.geo_const import create_basemap_obj, latcorners, loncorners


def normalize_cyclone_id(v) -> str:
    if pd.isna(v):
        return ""
    s = str(v).strip()
    if not s:
        return ""
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
        else:
            return str(f.upper())
    except Exception:
        pass
    return s


def _nearest_to_hour(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    work = df.copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    work = work[work[time_col].notna()].copy()
    if work.empty:
        return work
    work["hour_key"] = work[time_col].dt.round("h")
    work["dist_hour_sec"] = (work[time_col] - work["hour_key"]).abs().dt.total_seconds()
    work = (
        work.sort_values(["hour_key", "dist_hour_sec", time_col])
        .drop_duplicates(subset=["hour_key"], keep="first")
        .sort_values(time_col)
        .drop(columns=["hour_key", "dist_hour_sec"])
        .reset_index(drop=True)
    )
    return work


def _load_tracking_predictions_csv(pred_csv: Path) -> pd.DataFrame:
    expected_cols = ["datetime", "has_cyclone", "pred_lat", "pred_lon"]
    pred_df = pd.read_csv(pred_csv)
    if not set(expected_cols).issubset(pred_df.columns):
        pred_df = pd.read_csv(pred_csv, header=None, names=expected_cols)

    pred_df["datetime"] = pd.to_datetime(pred_df.get("datetime"), errors="coerce")
    pred_df["pred_lat"] = pd.to_numeric(pred_df.get("pred_lat"), errors="coerce")
    pred_df["pred_lon"] = pd.to_numeric(pred_df.get("pred_lon"), errors="coerce")
    if "has_cyclone" in pred_df.columns:
        pred_df["has_cyclone"] = pd.to_numeric(pred_df["has_cyclone"], errors="coerce").fillna(0).astype(int)
    else:
        pred_df["has_cyclone"] = 1
    return pred_df


def _build_id_aliases(cyclone_id: str) -> set:
    cid = normalize_cyclone_id(cyclone_id)
    aliases = {cid} if cid else set()
    if cid and cid.isdigit():
        if cid.startswith("700") and len(cid) > 3:
            aliases.add(cid[3:])
        else:
            aliases.add("700" + cid)
    return {a for a in aliases if a}


def _select_rows_by_aliases(df: pd.DataFrame, aliases: set, id_cols: list) -> pd.DataFrame:
    if df is None or df.empty or not aliases:
        return df.iloc[0:0].copy() if isinstance(df, pd.DataFrame) else pd.DataFrame([])
    mask = pd.Series(False, index=df.index)
    for c in id_cols:
        if c in df.columns:
            mask = mask | df[c].astype(str).isin(aliases)
    return df[mask].copy()


def _infer_gt_interval(gt_df: pd.DataFrame) -> tuple:
    if gt_df is None or gt_df.empty:
        return (pd.NaT, pd.NaT)

    t = pd.to_datetime(gt_df.get("time"), errors="coerce")
    t = t[t.notna()]
    start = t.min() if not t.empty else pd.NaT
    end = t.max() if not t.empty else pd.NaT

    if "start_time" in gt_df.columns:
        st = pd.to_datetime(gt_df.get("start_time"), errors="coerce")
        st = st[st.notna()]
        if not st.empty:
            start = st.min()

    if "end_time" in gt_df.columns:
        en = pd.to_datetime(gt_df.get("end_time"), errors="coerce")
        en = en[en.notna()]
        if not en.empty:
            end = en.max()

    return (start, end)


def _build_track_title(cyclone_id: str, gt_df: pd.DataFrame = None) -> str:
    prefix = "Deep-learning Medicane Tracking Algorithm (DeMeTrA)"
    cid = normalize_cyclone_id(cyclone_id)
    if cid.startswith("700") and len(cid) > 3:
        return f"{prefix} {cid[3:]}"

    if gt_df is not None and not gt_df.empty and "name" in gt_df.columns:
        names = gt_df["name"].dropna().astype(str).str.strip()
        names = names[names != ""]
        if not names.empty:
            return names.iloc[0]
    cid = str(cid.title())
    return f"{prefix} - {cid}" if cid else f"{prefix}"

def build_hist_cyclone_label(cyclone_id: str, id_to_name: dict = None) -> str:
    cid = normalize_cyclone_id(cyclone_id)
    if cid.startswith("700") and len(cid) > 3:
        return cid[3:]
    if id_to_name:
        return id_to_name.get(cid, cid)
    return cid


def _clean_track_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["time", "lat", "lon"])
    w = df.copy()
    w = w.loc[:, ~w.columns.duplicated()].copy()
    w["time"] = pd.to_datetime(w.get("time"), errors="coerce")
    w["lat"] = pd.to_numeric(w.get("lat"), errors="coerce")
    w["lon"] = pd.to_numeric(w.get("lon"), errors="coerce")
    w = w[w["time"].notna() & w["lat"].notna() & w["lon"].notna()].copy().sort_values("time")
    return w


def _window_slice(df: pd.DataFrame, window: tuple) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["time", "lat", "lon"])
    if window is None:
        return df.copy()
    w_start, w_end = window
    if pd.isna(w_start) or pd.isna(w_end):
        return pd.DataFrame(columns=df.columns)
    return df[(df["time"] >= pd.Timestamp(w_start)) & (df["time"] <= pd.Timestamp(w_end))].copy()


def _temporal_gradient_colors(base_rgb: tuple, n: int, start_mix: float = 0.35, end_mix: float = 1.0) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, 3), dtype=float)
    base = np.array(base_rgb, dtype=float).reshape(1, 3)
    white = np.ones((1, 3), dtype=float)
    mix = np.linspace(float(start_mix), float(end_mix), int(n)).reshape(-1, 1)
    colors = white * (1.0 - mix) + base * mix
    return np.clip(colors, 0.0, 1.0)


def _window_mask(df: pd.DataFrame, window: tuple) -> np.ndarray:
    if df is None or df.empty or window is None:
        return np.zeros(0, dtype=bool)
    w_start, w_end = window
    if pd.isna(w_start) or pd.isna(w_end):
        return np.zeros(len(df), dtype=bool)
    ts = pd.to_datetime(df["time"], errors="coerce")
    return (ts >= pd.Timestamp(w_start)) & (ts <= pd.Timestamp(w_end))


def _compute_zoom_bbox(gt_df: pd.DataFrame, window: tuple) -> tuple:
    glat_min, glat_max = float(latcorners[0]), float(latcorners[1])
    glon_min, glon_max = float(loncorners[0]), float(loncorners[1])

    min_lon_span = 12.0
    min_lat_span = 12.0
    extra_border_deg = 1.0

    if gt_df is None or gt_df.empty:
        return glon_min, glat_min, glon_max, glat_max

    pts = _window_slice(gt_df, window)
    if pts.empty:
        pts = gt_df.copy()
    if pts.empty:
        return glon_min, glat_min, glon_max, glat_max

    lat_lo = float(pts["lat"].min())
    lat_hi = float(pts["lat"].max())
    lon_lo = float(pts["lon"].min())
    lon_hi = float(pts["lon"].max())

    lat_c = 0.5 * (lat_lo + lat_hi)
    lon_c = 0.5 * (lon_lo + lon_hi)
    lat_span = max(min_lat_span, lat_hi - lat_lo)
    lon_span = max(min_lon_span, lon_hi - lon_lo)

    ll_lat = max(glat_min, lat_c - 0.5 * lat_span - extra_border_deg)
    ur_lat = min(glat_max, lat_c + 0.5 * lat_span + extra_border_deg)
    ll_lon = max(glon_min, lon_c - 0.5 * lon_span - extra_border_deg)
    ur_lon = min(glon_max, lon_c + 0.5 * lon_span + extra_border_deg)

    if ur_lat <= ll_lat or ur_lon <= ll_lon:
        return glon_min, glat_min, glon_max, glat_max

    return ll_lon, ll_lat, ur_lon, ur_lat


def _choose_bbox_df(gt_df: pd.DataFrame, pred_df: pd.DataFrame, mercad_df: pd.DataFrame) -> pd.DataFrame:
    if gt_df is not None and not gt_df.empty:
        return gt_df
    if pred_df is not None and not pred_df.empty:
        return pred_df
    if mercad_df is not None and not mercad_df.empty:
        return mercad_df
    return gt_df


def _create_zoom_basemap(ax, bbox: tuple):
    ll_lon, ll_lat, ur_lon, ur_lat = bbox
    try:
        return Basemap(
            projection="geos",
            ax=ax,
            rsphere=(6378137.0, 6356752.3142),
            resolution="i",
            area_thresh=10000.0,
            lon_0=9.5,
            satellite_height=3.5785831e7,
            llcrnrlon=float(ll_lon),
            llcrnrlat=float(ll_lat),
            urcrnrlon=float(ur_lon),
            urcrnrlat=float(ur_lat),
        )
    except Exception:
        return create_basemap_obj(ax=ax)


def _annotate_track_window_triplet(
    ax,
    x_vals,
    y_vals,
    t_vals,
    color: str,
    window: tuple = None,
    target_times: list = None,
    every_hours: int = None,
) -> None:
    if len(x_vals) == 0:
        return

    t_ser = pd.to_datetime(pd.Series(t_vals), errors="coerce")
    ann = pd.DataFrame({"x": x_vals, "y": y_vals, "t": t_ser}).dropna().sort_values("t").reset_index(drop=True)
    if ann.empty:
        return

    targets = []
    if target_times:
        for t in target_times:
            if pd.notna(t):
                targets.append(pd.Timestamp(t))

    w_start = pd.NaT
    w_end = pd.NaT
    if window is not None and len(window) == 2:
        w_start, w_end = window
        w_start = pd.Timestamp(w_start) if pd.notna(w_start) else pd.NaT
        w_end = pd.Timestamp(w_end) if pd.notna(w_end) else pd.NaT
    if pd.isna(w_start) or pd.isna(w_end) or w_end < w_start:
        w_start = pd.Timestamp(ann["t"].iloc[0])
        w_end = pd.Timestamp(ann["t"].iloc[-1])

    if not targets:
        targets = [w_start, w_end]

    if every_hours is not None:
        try:
            every_hours = int(every_hours)
        except Exception:
            every_hours = None
        if every_hours is not None and every_hours > 0:
            p_start = w_start
            p_end = w_end
            if pd.notna(p_start) and pd.notna(p_end) and p_end >= p_start:
                periodic = pd.date_range(start=p_start, end=p_end, freq=f"{every_hours}h")
                targets.extend(list(periodic))

    targets = sorted({pd.Timestamp(t) for t in targets if pd.notna(t)})

    used = set()
    label_idx = 0
    for tgt in targets:
        diffs = (ann["t"] - pd.Timestamp(tgt)).abs()
        idx = int(diffs.idxmin())
        if idx in used:
            continue
        used.add(idx)

        row = ann.iloc[idx]
        x0 = float(row["x"])
        y0 = float(row["y"])
        label = pd.Timestamp(row["t"]).strftime("%d-%m %H:%M")
        dy = 8 if (label_idx % 2 == 0) else -8
        va = "bottom" if dy > 0 else "top"
        label_idx += 1

        ax.scatter(
            [x0],
            [y0],
            s=32,
            facecolors="none",
            edgecolors="black",
            linewidths=0.8,
            zorder=9,
            clip_on=True,
        )
        ax.annotate(
            label,
            (x0, y0),
            xytext=(0, dy),
            textcoords="offset points",
            fontsize=5,
            color=color,
            alpha=0.7,
            ha="center",
            va=va,
            zorder=10,
            clip_on=True,
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor="white",
                edgecolor="black",
                linewidth=0.5,
                alpha=0.5,
            ),
            arrowprops=dict(
                arrowstyle="-",
                color="black",
                linewidth=0.7,
                alpha=0.7,
                shrinkA=0,
                shrinkB=3,
            ),
        )


def _plot_med_tracks_map(
    out_png: Path,
    title: str,
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    mercad_df: pd.DataFrame = None,
    gt_label: str = "Ground Truth",
    pred_label: str = "Prediction",
    mercad_label: str = "MERCAD",
    pred_connect_window: tuple = None,
    annotation_window: tuple = None,
    annotation_every_hours: int = None,
    dpi: int = 150,
) -> None:
    g = _clean_track_df(gt_df)
    r = _clean_track_df(pred_df)
    b = _clean_track_df(mercad_df)

    b_line = _window_slice(b, pred_connect_window)
    ann_window = annotation_window if annotation_window is not None else pred_connect_window

    bbox = _compute_zoom_bbox(_choose_bbox_df(g, r, b), ann_window)
    ll_lon, ll_lat, ur_lon, ur_lat = bbox
    lat_span = max(1e-6, ur_lat - ll_lat)
    lon_span = max(1e-6, ur_lon - ll_lon)
    bbox_ratio = lon_span / lat_span

    fig_h = 4.8
    fig_w = float(np.clip(fig_h * bbox_ratio * 1.05, 6.8, 14.0))
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_axes([0.09, 0.065, 0.84, 0.86])
    m = _create_zoom_basemap(ax=ax, bbox=bbox)
    m.drawcoastlines(linewidth=0.8, color="black", zorder=2)
    lat_step = 1.0
    lon_step = 1.0
    parallels = np.arange(np.floor(ll_lat), np.ceil(ur_lat) + 0.1, lat_step)
    meridians = np.arange(np.floor(ll_lon), np.ceil(ur_lon) + 0.1, lon_step)
    m.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=8, color="0.5")
    m.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=8, color="0.5", rotation=25)

    if not g.empty:
        xg, yg = m(g["lon"].to_numpy(), g["lat"].to_numpy())
        g_colors = _temporal_gradient_colors(base_rgb=(0.0, 0.55, 0.0), n=len(g), start_mix=0.28, end_mix=1.0)
        g_in_mask = _window_mask(g, pred_connect_window)
        if len(g_in_mask) == len(g) and g_in_mask.any():
            ax.scatter(
                np.array(xg)[~g_in_mask],
                np.array(yg)[~g_in_mask],
                s=9,
                marker="o",
                facecolors="none",
                edgecolors=g_colors[~g_in_mask],
                linewidths=0.9,
                zorder=4,
                label=gt_label,
            )
            ax.scatter(
                np.array(xg)[g_in_mask],
                np.array(yg)[g_in_mask],
                c=g_colors[g_in_mask],
                s=25,
                marker="o",
                edgecolors="none",
                zorder=4,
            )
        else:
            ax.scatter(
                np.array(xg),
                np.array(yg),
                s=9,
                marker="o",
                facecolors="none",
                edgecolors=g_colors,
                linewidths=0.9,
                zorder=4,
                label=gt_label,
            )

        g_targets = []
        g_t = pd.to_datetime(g["time"], errors="coerce").dropna()
        if not g_t.empty:
            g_targets.extend([g_t.min(), g_t.max()])
        if pred_connect_window is not None and len(pred_connect_window) == 2:
            g_targets.extend([pred_connect_window[0], pred_connect_window[1]])
        _annotate_track_window_triplet(
            ax,
            xg,
            yg,
            g["time"].to_numpy(),
            color="green",
            window=pred_connect_window,
            target_times=g_targets,
            every_hours=annotation_every_hours,
        )

    if not b.empty:
        xb, yb = m(b["lon"].to_numpy(), b["lat"].to_numpy())
        m.plot(
            xb,
            yb,
            color="royalblue",
            linestyle="None",
            marker="x",
            markersize=2.5,
            markeredgewidth=0.9,
            label=mercad_label,
            zorder=5,
        )
        if not b_line.empty:
            xb_in, yb_in = m(b_line["lon"].to_numpy(), b_line["lat"].to_numpy())
            m.plot(
                xb_in,
                yb_in,
                color="royalblue",
                linestyle="None",
                marker="x",
                markersize=5.0,
                markeredgewidth=1.2,
                zorder=5,
            )

    if not r.empty:
        xr_all, yr_all = m(r["lon"].to_numpy(), r["lat"].to_numpy())
        r_colors = plt.cm.jet(np.linspace(0.08, 0.92, len(r)))[:, :3]
        r_in_mask = _window_mask(r, pred_connect_window)
        if g.empty:
            ax.scatter(
                np.array(xr_all),
                np.array(yr_all),
                c=r_colors,
                s=11,
                marker="o",
                edgecolors="none",
                zorder=6,
                label=pred_label,
            )
        elif len(r_in_mask) == len(r) and r_in_mask.any():
            ax.scatter(
                np.array(xr_all)[~r_in_mask],
                np.array(yr_all)[~r_in_mask],
                s=9,
                marker="o",
                facecolors="none",
                edgecolors=r_colors[~r_in_mask],
                linewidths=0.9,
                zorder=6,
                label=pred_label,
            )
            ax.scatter(
                np.array(xr_all)[r_in_mask],
                np.array(yr_all)[r_in_mask],
                c=r_colors[r_in_mask],
                s=25,
                marker="o",
                edgecolors="none",
                zorder=6,
            )
        else:
            ax.scatter(
                np.array(xr_all),
                np.array(yr_all),
                s=9,
                marker="o",
                facecolors="none",
                edgecolors=r_colors,
                linewidths=0.9,
                zorder=6,
                label=pred_label,
            )

        r_targets = []
        if pred_connect_window is not None and len(pred_connect_window) == 2:
            r_targets = [pred_connect_window[0], pred_connect_window[1]]
        _annotate_track_window_triplet(
            ax,
            xr_all,
            yr_all,
            r["time"].to_numpy(),
            color="red",
            window=pred_connect_window,
            target_times=r_targets,
            every_hours=annotation_every_hours,
        )

    ax.set_title(title)
    legend_handles = []
    if not g.empty:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=6,
                markerfacecolor=(0.0, 0.55, 0.0),
                markeredgecolor=(0.0, 0.55, 0.0),
                label=gt_label,
            )
        )
    if not b.empty:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="x",
                linestyle="None",
                markersize=6,
                color="royalblue",
                markeredgewidth=1.0,
                label=mercad_label,
            )
        )
    if not r.empty:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=6,
                markerfacecolor=(0.80, 0.0, 0.0),
                markeredgecolor=(0.80, 0.0, 0.0),
                label=pred_label,
            )
        )
    if legend_handles:
        #ax.legend(handles=legend_handles, loc="lower left", fontsize=8)
        pass

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def save_error_histogram(
    values,
    out_png: Path,
    title: str,
    bins: int = 30,
    xmax: float = 500.0,
    color: str = "indianred",
    dpi: int = 150,
) -> None:
    vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    vals = vals[vals > 0]
    fig, ax = plt.subplots(figsize=(7, 4), tight_layout=True)
    sns.histplot(vals, bins=bins, binrange=(0, xmax), ax=ax, color=color)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Kilometers", fontsize=12)
    ax.set_ylabel("Number of clips", fontsize=12)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(alpha=0.2)
    ax.set_xlim(0, xmax)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def save_split_error_histogram(
    values_in,
    values_out,
    out_png: Path,
    title: str,
    bins: int = 30,
    xmax: float = 500.0,
    dpi: int = 150,
    inside_label: str = "Inside mature phase",
    outside_label: str = "Outside mature phase",
    inside_color: str = "indianred",
    outside_color: str = "mediumpurple",
    inside_alpha: float = 0.55,
    outside_alpha: float = 0.45,
    title_size: int = 13,
    label_size: int = 11,
    legend_size: int = 8,
) -> None:
    vals_in = pd.to_numeric(pd.Series(values_in), errors="coerce").dropna()
    vals_out = pd.to_numeric(pd.Series(values_out), errors="coerce").dropna()
    vals_in = vals_in[vals_in > 0]
    vals_out = vals_out[vals_out > 0]

    fig, ax = plt.subplots(figsize=(7, 4), tight_layout=True)
    if not vals_out.empty:
        sns.histplot(
            vals_out,
            bins=bins,
            binrange=(0, xmax),
            ax=ax,
            color=outside_color,
            alpha=outside_alpha,
            label=f"{outside_label} (n={len(vals_out)})",
        )
    if not vals_in.empty:
        sns.histplot(
            vals_in,
            bins=bins,
            binrange=(0, xmax),
            ax=ax,
            color=inside_color,
            alpha=inside_alpha,
            label=f"{inside_label} (n={len(vals_in)})",
        )

    ax.set_title(title, fontsize=title_size)
    ax.set_xlabel("Kilometers", fontsize=label_size)
    ax.set_ylabel("Number of clips", fontsize=label_size)
    ax.grid(alpha=0.2)
    ax.set_xlim(0, xmax)
    if (not vals_out.empty) and (not vals_in.empty):
        ax.legend(fontsize=legend_size)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def extract_thresholds_from_last_cell(nb_path: Path) -> dict:
    nb = json.loads(nb_path.read_text())
    code_cells = [
        "".join(c.get("source", []))
        for c in nb.get("cells", [])
        if c.get("cell_type") == "code" and "".join(c.get("source", [])).strip()
    ]
    if not code_cells:
        raise RuntimeError(f"Nessuna cella code non vuota in {nb_path}")

    tree = ast.parse(code_cells[-1])
    dict_obj = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            try:
                candidate = ast.literal_eval(node.value)
            except Exception:
                continue
            if isinstance(candidate, dict) and candidate:
                dict_obj = candidate

    if not isinstance(dict_obj, dict) or not dict_obj:
        raise RuntimeError(
            "Ultima cella code non contiene un dizionario soglie assegnato. "
            "Controlla per_cyclone_threshold_analysis.ipynb"
        )

    out = {}
    for k, v in dict_obj.items():
        kid = normalize_cyclone_id(k)
        if not kid:
            continue
        try:
            out[kid] = float(v)
        except Exception:
            continue

    if not out:
        raise RuntimeError("Dizionario soglie estratto ma vuoto/non valido.")
    return out


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    lat1 = np.radians(float(lat1))
    lon1 = np.radians(float(lon1))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon / 2.0) ** 2)
    return 2.0 * r * np.arcsin(np.sqrt(a))


def build_tracking_error_eval(
    all_tile_preds: pd.DataFrame,
    manos_file: Path,
    output_csv: Path = None,
    manos_cl7_file: Path = None,
    era5_file: Path = None,
):
    if all_tile_preds.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    eval_src = all_tile_preds.copy()
    if "path" not in eval_src.columns:
        raise RuntimeError("all_tile_preds non contiene colonna 'path'.")

    path_base = eval_src["path"].astype(str).str.replace(r".*[\\/]", "", regex=True)
    dt_str = path_base.str.extract(r"^(\d{2}-\d{2}-\d{4}_\d{4})", expand=False)
    eval_src["tile_dt"] = pd.to_datetime(dt_str, format="%d-%m-%Y_%H%M", errors="coerce")
    eval_src = eval_src[eval_src["tile_dt"].notna()].copy()
    if eval_src.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    eval_src["cyclone_id"] = eval_src["cyclone_id"].apply(normalize_cyclone_id)
    eval_src = eval_src[eval_src["cyclone_id"] != ""].copy()
    eval_src["hour_key"] = eval_src["tile_dt"].dt.round("h")
    eval_src["dist_to_hour_sec"] = (eval_src["tile_dt"] - eval_src["hour_key"]).abs().dt.total_seconds()
    eval_src = (
        eval_src.sort_values(["cyclone_id", "hour_key", "dist_to_hour_sec", "tile_dt"])
        .drop_duplicates(subset=["cyclone_id", "hour_key"], keep="first")
        .reset_index(drop=True)
    )

    manos_cl7_file = manos_cl7_file or (REPO_ROOT / "notebooks" / "manos_CL7_pixel.csv")
    era5_file = era5_file or (REPO_ROOT / "moduli" / "videomae" / "medicane_data_input" / "era5_medicanes.csv")

    gt_main = pd.read_csv(manos_file)
    if "id_cyc_unico" not in gt_main.columns:
        raise RuntimeError(f"MANOS_FILE senza id_cyc_unico: {manos_file}")

    gt_main["cyclone_id"] = gt_main["id_cyc_unico"].apply(normalize_cyclone_id)
    gt_main["id_final_norm"] = gt_main["id_final"].apply(normalize_cyclone_id) if "id_final" in gt_main.columns else ""
    gt_main["idorig_norm"] = gt_main["idorig"].apply(normalize_cyclone_id) if "idorig" in gt_main.columns else ""
    gt_main["time"] = pd.to_datetime(gt_main.get("time"), errors="coerce")
    gt_main["lat"] = pd.to_numeric(gt_main.get("lat"), errors="coerce")
    gt_main["lon"] = pd.to_numeric(gt_main.get("lon"), errors="coerce")
    gt_main["start_time"] = pd.to_datetime(gt_main.get("start_time"), errors="coerce") if "start_time" in gt_main.columns else pd.NaT
    gt_main["end_time"] = pd.to_datetime(gt_main.get("end_time"), errors="coerce") if "end_time" in gt_main.columns else pd.NaT
    id_any_main = (
        gt_main["cyclone_id"].astype(str).ne("")
        | gt_main["id_final_norm"].astype(str).ne("")
        | gt_main["idorig_norm"].astype(str).ne("")
    )
    gt_main = gt_main[gt_main["time"].notna() & gt_main["lat"].notna() & gt_main["lon"].notna() & id_any_main].copy()

    if manos_cl7_file.exists():
        gt_cl7 = pd.read_csv(manos_cl7_file)
        gt_cl7["cyclone_id"] = gt_cl7["id_cyc_unico"].apply(normalize_cyclone_id) if "id_cyc_unico" in gt_cl7.columns else ""
        gt_cl7["id_final_norm"] = gt_cl7["id_final"].apply(normalize_cyclone_id) if "id_final" in gt_cl7.columns else ""
        gt_cl7["idorig_norm"] = gt_cl7["idorig"].apply(normalize_cyclone_id) if "idorig" in gt_cl7.columns else ""
        gt_cl7["time"] = pd.to_datetime(gt_cl7.get("time"), errors="coerce")
        gt_cl7["lat"] = pd.to_numeric(gt_cl7.get("lat"), errors="coerce")
        gt_cl7["lon"] = pd.to_numeric(gt_cl7.get("lon"), errors="coerce")
        gt_cl7["start_time"] = pd.NaT
        gt_cl7["end_time"] = pd.NaT
        gt_cl7 = gt_cl7[
            gt_cl7["time"].notna() & gt_cl7["lat"].notna() & gt_cl7["lon"].notna() & (gt_cl7["cyclone_id"] != "")
        ].copy()
    else:
        gt_cl7 = pd.DataFrame(columns=["cyclone_id", "id_final_norm", "idorig_norm", "time", "lat", "lon", "start_time", "end_time"])

    id_to_name = {}
    if era5_file.exists():
        gt_era5 = pd.read_csv(era5_file)
        gt_era5["time"] = pd.to_datetime(gt_era5.get("time"), errors="coerce")
        gt_era5["start_time"] = pd.to_datetime(gt_era5.get("start_time"), errors="coerce")
        gt_era5["end_time"] = pd.to_datetime(gt_era5.get("end_time"), errors="coerce")
        gt_era5["lat"] = pd.to_numeric(gt_era5.get("lat"), errors="coerce")
        gt_era5["lon"] = pd.to_numeric(gt_era5.get("lon"), errors="coerce")
        gt_era5["name_norm"] = gt_era5.get("name", "").astype(str).str.strip().str.lower()
        gt_era5 = gt_era5[
            gt_era5["time"].notna() & gt_era5["lat"].notna() & gt_era5["lon"].notna() & (gt_era5["name_norm"] != "")
        ].copy()

        name_to_id = {}
        if "name" in gt_main.columns:
            map_df = gt_main.copy()
            map_df["name_norm"] = map_df["name"].astype(str).str.strip().str.lower()
            map_df["name_orig"] = map_df["name"].astype(str).str.strip()
            map_df = map_df[
                map_df["name_norm"].notna()
                & (map_df["name_norm"] != "")
                & map_df["id_final_norm"].notna()
                & (map_df["id_final_norm"] != "")
                & (~map_df["id_final_norm"].astype(str).str.startswith("700"))
            ].copy()
            for n, g in map_df.groupby("name_norm"):
                ids = sorted(set(g["id_final_norm"].astype(str)))
                if len(ids) == 1:
                    name_to_id[n] = ids[0]
                    names = [x for x in g["name_orig"].astype(str).tolist() if x]
                    if names:
                        id_to_name[ids[0]] = names[0]

        gt_era5["cyclone_id"] = gt_era5["name_norm"].map(name_to_id).fillna("")
        gt_era5["id_final_norm"] = gt_era5["cyclone_id"]
        gt_era5["idorig_norm"] = ""
        gt_era5 = gt_era5[gt_era5["cyclone_id"] != ""].copy()
    else:
        gt_era5 = pd.DataFrame(columns=["cyclone_id", "id_final_norm", "idorig_norm", "time", "lat", "lon", "start_time", "end_time"])

    gt_cols = ["cyclone_id", "id_final_norm", "idorig_norm", "time", "lat", "lon", "start_time", "end_time"]
    gt_match_df = pd.concat(
        [
            gt_main[gt_cols] if not gt_main.empty else pd.DataFrame(columns=gt_cols),
            gt_cl7[gt_cols] if not gt_cl7.empty else pd.DataFrame(columns=gt_cols),
            gt_era5[gt_cols] if not gt_era5.empty else pd.DataFrame(columns=gt_cols),
        ],
        ignore_index=True,
    )
    if not gt_match_df.empty:
        gt_match_df = gt_match_df.drop_duplicates(
            subset=["cyclone_id", "id_final_norm", "idorig_norm", "time", "lat", "lon"], keep="first"
        )
    gt_match_df["hour_key"] = pd.to_datetime(gt_match_df.get("time"), errors="coerce").dt.round("h")
    gt_main["hour_key"] = pd.to_datetime(gt_main.get("time"), errors="coerce").dt.round("h")

    gt_index = {}
    for row in gt_match_df.itertuples(index=False):
        keys = []
        for c in ["cyclone_id", "id_final_norm", "idorig_norm"]:
            v = getattr(row, c, "")
            if isinstance(v, str) and v != "":
                keys.append(v)
        for k in set(keys):
            gt_index.setdefault((k, pd.Timestamp(row.hour_key)), []).append((float(row.lat), float(row.lon)))

    alias_good_window = {}
    alias_time_window_main = {}
    for row in gt_main.itertuples(index=False):
        keys = []
        for c in ["cyclone_id", "id_final_norm", "idorig_norm"]:
            v = getattr(row, c, "")
            if isinstance(v, str) and v != "":
                keys.append(v)
        keys = list(set(keys))
        row_time = pd.Timestamp(getattr(row, "time"))
        st = getattr(row, "start_time", pd.NaT)
        en = getattr(row, "end_time", pd.NaT)
        st = pd.Timestamp(st) if pd.notna(st) else pd.NaT
        en = pd.Timestamp(en) if pd.notna(en) else pd.NaT
        if pd.isna(st):
            st = row_time
        if pd.isna(en):
            en = row_time
        if en < st:
            st, en = en, st
        for k in keys:
            if k in alias_good_window:
                cur_st, cur_en = alias_good_window[k]
                alias_good_window[k] = (min(cur_st, st), max(cur_en, en))
            else:
                alias_good_window[k] = (st, en)
            if k in alias_time_window_main:
                cur_st, cur_en = alias_time_window_main[k]
                alias_time_window_main[k] = (min(cur_st, row_time), max(cur_en, row_time))
            else:
                alias_time_window_main[k] = (row_time, row_time)

    alias_time_window_match = {}
    for row in gt_match_df.itertuples(index=False):
        keys = []
        for c in ["cyclone_id", "id_final_norm", "idorig_norm"]:
            v = getattr(row, c, "")
            if isinstance(v, str) and v != "":
                keys.append(v)
        row_time = pd.Timestamp(getattr(row, "time"))
        for k in set(keys):
            if k in alias_time_window_match:
                cur_st, cur_en = alias_time_window_match[k]
                alias_time_window_match[k] = (min(cur_st, row_time), max(cur_en, row_time))
            else:
                alias_time_window_match[k] = (row_time, row_time)

    cid_good_window_cache = {}
    for cid in sorted(eval_src["cyclone_id"].dropna().astype(str).unique()):
        aliases = _build_id_aliases(cid)
        starts = []
        ends = []
        for a in aliases:
            if a in alias_good_window:
                w = alias_good_window[a]
                starts.append(pd.Timestamp(w[0]))
                ends.append(pd.Timestamp(w[1]))
        if not starts:
            for a in aliases:
                if a in alias_time_window_main:
                    w = alias_time_window_main[a]
                    starts.append(pd.Timestamp(w[0]))
                    ends.append(pd.Timestamp(w[1]))
        if not starts:
            for a in aliases:
                if a in alias_time_window_match:
                    w = alias_time_window_match[a]
                    starts.append(pd.Timestamp(w[0]))
                    ends.append(pd.Timestamp(w[1]))
        cid_good_window_cache[cid] = (min(starts), max(ends)) if starts else (pd.NaT, pd.NaT)

    rows_eval = []
    for row in eval_src.itertuples(index=False):
        cid = normalize_cyclone_id(getattr(row, "cyclone_id", ""))
        hk = pd.Timestamp(getattr(row, "hour_key"))
        pred_lat = pd.to_numeric(getattr(row, "pred_lat", np.nan), errors="coerce")
        pred_lon = pd.to_numeric(getattr(row, "pred_lon", np.nan), errors="coerce")
        gw_st, gw_en = cid_good_window_cache.get(cid, (pd.NaT, pd.NaT))
        in_good_window = int(pd.notna(gw_st) and pd.notna(gw_en) and (hk >= gw_st) and (hk <= gw_en))
        out = {
            "cyclone_id": cid,
            "path": getattr(row, "path", ""),
            "tile_dt": getattr(row, "tile_dt", pd.NaT),
            "hour_key": hk,
            "dist_to_hour_sec": float(getattr(row, "dist_to_hour_sec", np.nan)),
            "pred_lat": pred_lat,
            "pred_lon": pred_lon,
            "gt_lat": np.nan,
            "gt_lon": np.nan,
            "err_km_eval": np.nan,
            "has_gt_match": 0,
            "in_good_window": in_good_window,
            "good_window_start": gw_st,
            "good_window_end": gw_en,
            "orig_has_target": pd.to_numeric(getattr(row, "has_target", np.nan), errors="coerce"),
            "orig_err_km": pd.to_numeric(getattr(row, "err_km", np.nan), errors="coerce"),
        }

        if np.isfinite(pred_lat) and np.isfinite(pred_lon):
            aliases = _build_id_aliases(cid)
            cand = []
            for a in aliases:
                cand.extend(gt_index.get((a, hk), []))
            if cand:
                arr = np.asarray(cand, dtype=float)
                d = _haversine_km(pred_lat, pred_lon, arr[:, 0], arr[:, 1])
                i = int(np.argmin(d))
                out["gt_lat"] = float(arr[i, 0])
                out["gt_lon"] = float(arr[i, 1])
                out["err_km_eval"] = float(d[i])
                out["has_gt_match"] = 1

        rows_eval.append(out)

    err_eval_df = pd.DataFrame(rows_eval)
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        err_eval_df.to_csv(output_csv, index=False)
    err_df = err_eval_df[err_eval_df["err_km_eval"].notna() & (pd.to_numeric(err_eval_df["err_km_eval"], errors="coerce") > 0)].copy()
    return err_eval_df, err_df, id_to_name


def render_tracking_error_histograms(
    err_df: pd.DataFrame,
    batch_out_dir: Path,
    id_to_name: dict = None,
    hist_bins: int = 30,
    hist_xmax_km: float = 500.0,
) -> pd.DataFrame:
    batch_out_dir.mkdir(parents=True, exist_ok=True)
    hist_out_dir = batch_out_dir / "hist_per_cyclone"
    hist_out_dir.mkdir(parents=True, exist_ok=True)

    if err_df.empty:
        return pd.DataFrame()

    save_error_histogram(
        err_df["err_km_eval"],
        out_png=batch_out_dir / "tracking_error_distribution_all_cyclones.png",
        title="Distribution of Tracking Error (km) - All cyclones",
        bins=hist_bins,
        xmax=hist_xmax_km,
    )

    if "in_good_window" in err_df.columns:
        in_mask_all = pd.to_numeric(err_df["in_good_window"], errors="coerce").fillna(0).astype(int) == 1
        vals_in_all = pd.to_numeric(err_df.loc[in_mask_all, "err_km_eval"], errors="coerce").dropna()
        vals_out_all = pd.to_numeric(err_df.loc[~in_mask_all, "err_km_eval"], errors="coerce").dropna()
        vals_in_all = vals_in_all[vals_in_all > 0]
        vals_out_all = vals_out_all[vals_out_all > 0]
        if (not vals_in_all.empty) or (not vals_out_all.empty):
            save_split_error_histogram(
                values_in=vals_in_all,
                values_out=vals_out_all,
                out_png=batch_out_dir / "tracking_error_distribution_all_cyclones_mature_split.png",
                title="Distribution of Tracking Error (km) - All cyclones",
                bins=hist_bins,
                xmax=hist_xmax_km,
            )

    hist_rows = []
    for cid, g in err_df.groupby("cyclone_id"):
        cid = normalize_cyclone_id(cid)
        if not cid:
            continue
        vals = pd.to_numeric(g["err_km_eval"], errors="coerce")
        vals = vals[vals.notna() & (vals > 0)]
        if vals.empty:
            continue
        g2 = g.loc[vals.index].copy()
        in_mask = pd.to_numeric(g2.get("in_good_window", 0), errors="coerce").fillna(0).astype(int) == 1
        vals_in = pd.to_numeric(g2.loc[in_mask, "err_km_eval"], errors="coerce").dropna()
        vals_out = pd.to_numeric(g2.loc[~in_mask, "err_km_eval"], errors="coerce").dropna()
        vals_in = vals_in[vals_in > 0]
        vals_out = vals_out[vals_out > 0]
        if vals_in.empty and vals_out.empty:
            continue

        all_vals = pd.concat([vals_in, vals_out], ignore_index=True)
        cid_plot = build_hist_cyclone_label(cid, id_to_name=id_to_name or {})
        out_png = hist_out_dir / f"tracking_error_distribution_{cid}.png"
        save_split_error_histogram(
            values_in=vals_in,
            values_out=vals_out,
            out_png=out_png,
            title=f"Distribution of Tracking Error (km) - Cyclone {cid_plot}",
            bins=hist_bins,
            xmax=hist_xmax_km,
        )
        hist_rows.append(
            {
                "cyclone_id": cid,
                "n_total": int(len(all_vals)),
                "n_in_good_window": int(len(vals_in)),
                "n_outside_good_window": int(len(vals_out)),
                "median_err_km_total": float(np.median(all_vals)),
                "mean_err_km_total": float(np.mean(all_vals)),
                "median_err_km_in": float(np.median(vals_in)) if not vals_in.empty else np.nan,
                "median_err_km_out": float(np.median(vals_out)) if not vals_out.empty else np.nan,
                "hist_path": str(out_png),
            }
        )

    return pd.DataFrame(hist_rows).sort_values("cyclone_id").reset_index(drop=True)


def build_all_cyclone_maps(
    source_df: pd.DataFrame,
    maps_out_dir: Path,
    manos_file: Path,
    more_manos_file: Path = None,
    manos_cl7_file: Path = None,
    mercad_manos_path: Path = None,
    use_nearest_hour="auto",
    annotation_every_hours: int = None,
    dpi: int = 150,
) -> pd.DataFrame:
    more_manos_file = more_manos_file or (REPO_ROOT / "moduli" / "videomae" / "medicane_data_input" / "more_medicanes_time_updated.csv")
    manos_cl7_file = manos_cl7_file or (REPO_ROOT / "notebooks" / "manos_CL7_pixel.csv")
    mercad_manos_path = mercad_manos_path or (REPO_ROOT / "notebooks" / "medicanes_new_windows_with_mercad.csv")

    manos_df = pd.read_csv(manos_file)
    if "id_cyc_unico" not in manos_df.columns:
        raise RuntimeError(f"manos_file senza colonna id_cyc_unico: {manos_file}")
    manos_df["cyclone_id"] = manos_df["id_cyc_unico"].apply(normalize_cyclone_id)
    manos_df["id_final_norm"] = manos_df["id_final"].apply(normalize_cyclone_id) if "id_final" in manos_df.columns else ""
    manos_df["idorig_norm"] = manos_df["idorig"].apply(normalize_cyclone_id) if "idorig" in manos_df.columns else ""
    manos_df["time"] = pd.to_datetime(manos_df.get("time"), errors="coerce")
    manos_df["lat"] = pd.to_numeric(manos_df.get("lat"), errors="coerce")
    manos_df["lon"] = pd.to_numeric(manos_df.get("lon"), errors="coerce")
    if "start_time" in manos_df.columns:
        manos_df["start_time"] = pd.to_datetime(manos_df.get("start_time"), errors="coerce")
    if "end_time" in manos_df.columns:
        manos_df["end_time"] = pd.to_datetime(manos_df.get("end_time"), errors="coerce")
    id_any_mask = (
        (manos_df["cyclone_id"] != "")
        | (manos_df["id_final_norm"] != "")
        | (manos_df["idorig_norm"] != "")
    )
    manos_df = manos_df[manos_df["time"].notna() & manos_df["lat"].notna() & manos_df["lon"].notna() & id_any_mask].copy()

    if more_manos_file.exists():
        more_df = pd.read_csv(more_manos_file)
        more_df["cyclone_id"] = more_df["id_cyc_unico"].apply(normalize_cyclone_id) if "id_cyc_unico" in more_df.columns else ""
        more_df["id_final_norm"] = more_df["id_final"].apply(normalize_cyclone_id) if "id_final" in more_df.columns else ""
        more_df["idorig_norm"] = more_df["idorig"].apply(normalize_cyclone_id) if "idorig" in more_df.columns else ""
        more_df["time"] = pd.to_datetime(more_df.get("time"), errors="coerce")
        more_df["lat"] = pd.to_numeric(more_df.get("lat"), errors="coerce")
        more_df["lon"] = pd.to_numeric(more_df.get("lon"), errors="coerce")
        if "start_time" in more_df.columns:
            more_df["start_time"] = pd.to_datetime(more_df.get("start_time"), errors="coerce")
        if "end_time" in more_df.columns:
            more_df["end_time"] = pd.to_datetime(more_df.get("end_time"), errors="coerce")
        id_any_more = (
            (more_df["cyclone_id"] != "")
            | (more_df["id_final_norm"] != "")
            | (more_df["idorig_norm"] != "")
        )
        more_df = more_df[more_df["time"].notna() & more_df["lat"].notna() & more_df["lon"].notna() & id_any_more].copy()
    else:
        more_df = pd.DataFrame(columns=["cyclone_id", "id_final_norm", "idorig_norm", "time", "lat", "lon", "start_time", "end_time", "name"])

    if manos_cl7_file.exists():
        manos_cl7_df = pd.read_csv(manos_cl7_file)
        manos_cl7_df["cyclone_id"] = manos_cl7_df["id_cyc_unico"].apply(normalize_cyclone_id) if "id_cyc_unico" in manos_cl7_df.columns else ""
        manos_cl7_df["id_final_norm"] = ""
        manos_cl7_df["idorig_norm"] = ""
        manos_cl7_df["time"] = pd.to_datetime(manos_cl7_df.get("time"), errors="coerce")
        manos_cl7_df["lat"] = pd.to_numeric(manos_cl7_df.get("lat"), errors="coerce")
        manos_cl7_df["lon"] = pd.to_numeric(manos_cl7_df.get("lon"), errors="coerce")
        manos_cl7_df["start_time"] = pd.to_datetime(manos_cl7_df.get("start_time"), errors="coerce") if "start_time" in manos_cl7_df.columns else pd.NaT
        manos_cl7_df["end_time"] = pd.to_datetime(manos_cl7_df.get("end_time"), errors="coerce") if "end_time" in manos_cl7_df.columns else pd.NaT
        manos_cl7_df["name"] = manos_cl7_df.get("name", "")
        manos_cl7_df = manos_cl7_df[
            manos_cl7_df["cyclone_id"].astype(str).str.startswith("700")
            & manos_cl7_df["time"].notna()
            & manos_cl7_df["lat"].notna()
            & manos_cl7_df["lon"].notna()
        ].copy()
    else:
        manos_cl7_df = pd.DataFrame(columns=["cyclone_id", "id_final_norm", "idorig_norm", "time", "lat", "lon", "start_time", "end_time", "name"])

    if mercad_manos_path.exists():
        mercad_src = pd.read_csv(mercad_manos_path)
        mercad_src["cyclone_id"] = mercad_src["id_cyc_unico"].apply(normalize_cyclone_id)
        mercad_src["id_final_norm"] = mercad_src["id_final"].apply(normalize_cyclone_id) if "id_final" in mercad_src.columns else ""
        mercad_src["idorig_norm"] = mercad_src["idorig"].apply(normalize_cyclone_id) if "idorig" in mercad_src.columns else ""
        mercad_src["time"] = pd.to_datetime(mercad_src.get("time"), errors="coerce")
        mercad_src["mercad_lat"] = pd.to_numeric(mercad_src.get("mercad_lat"), errors="coerce")
        mercad_src["mercad_lon"] = pd.to_numeric(mercad_src.get("mercad_lon"), errors="coerce")
        mercad_src = mercad_src[mercad_src["time"].notna() & mercad_src["mercad_lat"].notna() & mercad_src["mercad_lon"].notna()].copy()
    else:
        mercad_src = pd.DataFrame(columns=["cyclone_id", "id_final_norm", "idorig_norm", "time", "mercad_lat", "mercad_lon"])

    maps_out_dir.mkdir(parents=True, exist_ok=True)
    map_rows = []
    for _, row in source_df.iterrows():
        cid = normalize_cyclone_id(row["cyclone_id"])
        out_dir = Path(row["output_dir"])
        pred_csv = out_dir / "tracking_inference_predictions.csv"
        if not pred_csv.exists():
            map_rows.append(
                {
                    "cyclone_id": cid,
                    "status": "missing_pred_csv",
                    "pred_rows_raw": 0,
                    "pred_rows_hourly": 0,
                    "gt_rows": 0,
                    "mercad_rows": 0,
                    "map_path": "",
                }
            )
            continue

        pred_df = _load_tracking_predictions_csv(pred_csv)
        pred_df = pred_df[
            (pred_df["has_cyclone"] == 1)
            & pred_df["datetime"].notna()
            & pred_df["pred_lat"].notna()
            & pred_df["pred_lon"].notna()
        ].copy().sort_values("datetime")
        aliases = _build_id_aliases(cid)
        gt_base = _select_rows_by_aliases(manos_df, aliases, ["cyclone_id", "id_final_norm", "idorig_norm"]).sort_values("time")
        gt_more = _select_rows_by_aliases(more_df, aliases, ["cyclone_id", "id_final_norm", "idorig_norm"]).sort_values("time")
        gt_cl7_ext = (
            _select_rows_by_aliases(manos_cl7_df, aliases, ["cyclone_id", "id_final_norm", "idorig_norm"]).sort_values("time")
            if cid.startswith("700")
            else pd.DataFrame(columns=manos_cl7_df.columns)
        )
        gt_parts = [d for d in [gt_base, gt_more, gt_cl7_ext] if not d.empty]
        if gt_parts:
            gt_df = pd.concat(gt_parts, ignore_index=True)
            gt_df = gt_df.drop_duplicates(subset=["time", "lat", "lon"], keep="first").sort_values("time").reset_index(drop=True)
        else:
            gt_df = pd.DataFrame(columns=["time", "lat", "lon", "start_time", "end_time", "name"])

        pred_rows_raw = int(len(pred_df))
        if use_nearest_hour == "auto":
            should_downsample = not gt_df.empty
        else:
            should_downsample = bool(use_nearest_hour)
        if should_downsample:
            pred_df = _nearest_to_hour(pred_df, "datetime")

        gt_window = _infer_gt_interval(gt_base if not gt_base.empty else gt_df)
        gt_window_labels = gt_window
        if not gt_cl7_ext.empty:
            ext_window = _infer_gt_interval(gt_cl7_ext)
            if pd.notna(ext_window[0]) and pd.notna(ext_window[1]):
                gt_window_labels = ext_window

        mercad_cid = _select_rows_by_aliases(mercad_src, aliases, ["cyclone_id", "id_final_norm", "idorig_norm"]).sort_values("time")
        if pred_df.empty and gt_df.empty and mercad_cid.empty:
            map_rows.append(
                {
                    "cyclone_id": cid,
                    "status": "no_data",
                    "pred_rows_raw": pred_rows_raw,
                    "pred_rows_hourly": 0,
                    "gt_rows": 0,
                    "mercad_rows": 0,
                    "map_path": "",
                }
            )
            continue

        out_png = maps_out_dir / f"mediterranean_track_{cid}.png"
        _plot_med_tracks_map(
            out_png=out_png,
            title=_build_track_title(cid, gt_df=gt_df),
            gt_df=gt_df[["time", "lat", "lon", "start_time", "end_time", "name"]] if not gt_df.empty else gt_df,
            pred_df=pred_df.rename(columns={"datetime": "time", "pred_lat": "lat", "pred_lon": "lon"}),
            mercad_df=(
                mercad_cid[["time", "mercad_lat", "mercad_lon"]].rename(columns={"mercad_lat": "lat", "mercad_lon": "lon"})
                if not mercad_cid.empty
                else mercad_cid
            ),
            gt_label="Ground Truth",
            pred_label="DeMeTra",
            mercad_label="MERCAD",
            pred_connect_window=gt_window,
            annotation_window=gt_window_labels,
            annotation_every_hours=annotation_every_hours,
            dpi=dpi,
        )
        map_rows.append(
            {
                "cyclone_id": cid,
                "status": "ok",
                "pred_rows_raw": pred_rows_raw,
                "pred_rows_hourly": int(len(pred_df)),
                "gt_rows": int(len(gt_df)),
                "gt_rows_base": int(len(gt_base)),
                "gt_rows_more": int(len(gt_more)),
                "gt_rows_cl7_ext": int(len(gt_cl7_ext)),
                "mercad_rows": int(len(mercad_cid)),
                "map_path": str(out_png),
            }
        )

    return pd.DataFrame(map_rows).sort_values("cyclone_id").reset_index(drop=True)


__all__ = [
    "build_all_cyclone_maps",
    "build_hist_cyclone_label",
    "build_tracking_error_eval",
    "extract_thresholds_from_last_cell",
    "normalize_cyclone_id",
    "render_tracking_error_histograms",
    "save_error_histogram",
    "save_split_error_histogram",
    "_annotate_track_window_triplet",
    "_build_id_aliases",
    "_build_track_title",
    "_clean_track_df",
    "_compute_zoom_bbox",
    "_create_zoom_basemap",
    "_infer_gt_interval",
    "_nearest_to_hour",
    "_plot_med_tracks_map",
    "_select_rows_by_aliases",
    "_temporal_gradient_colors",
    "_window_mask",
    "_window_slice",
]
