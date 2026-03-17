#!/usr/bin/env python3
"""
Crea immagini Air Mass RGB partendo da file HRSEVIRI full-disk gia' scaricati in locale.

Input atteso:
  - cartella con ZIP EUMETSAT contenenti file .nat

Output:
  - PNG airmass_rgb_YYYYMMDD_HHMM.png

Note:
  - La composizione RGB e la conversione 8 bit seguono la logica di
    download_airmassRGB.py
  - Non fa download da cloud.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import shutil
import tempfile
import traceback
import zipfile
from pathlib import Path

import numpy as np
import xarray as xr

try:
    import imageio.v2 as imageio
except ImportError:
    import imageio  # type: ignore

try:
    from satpy import Scene
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Manca 'satpy'. Installa l'ambiente con le dipendenze satellite prima di eseguire."
    ) from exc

try:
    from geo_const import default_basem_obj, lat_min, lat_max, lon_min, lon_max
except ImportError:
    from medicane_utils.geo_const import default_basem_obj, lat_min, lat_max, lon_min, lon_max


CHANNELS = ("WV_062", "WV_073", "IR_097", "IR_108")


def to_8bit_airmass(arr_float32: np.ndarray) -> np.ndarray:
    """
    Converte array float32 (y, x, 3) in 8 bit con recipe Air Mass.
    Canali in input:
      R = WV_062 - WV_073
      G = IR_097 - IR_108
      B = WV_062
    """
    r_chan = arr_float32[:, :, 0]
    g_chan = arr_float32[:, :, 1]
    b_chan = arr_float32[:, :, 2]

    rmin, rmax = -25, 0
    gmin, gmax = -40, 5
    bmin, bmax = 243, 208

    def scale_custom(x: np.ndarray, lower: float, upper: float, invert: bool = False) -> np.ndarray:
        lo, hi = min(lower, upper), max(lower, upper)
        x_clipped = np.clip(x, lo, hi)
        scaled = (x_clipped - lo) / (hi - lo)
        if invert:
            scaled = 1.0 - scaled
        return np.clip(scaled, 0.0, 1.0)

    r_scaled = scale_custom(r_chan, rmin, rmax, invert=False)
    g_scaled = scale_custom(g_chan, gmin, gmax, invert=False)
    b_scaled = scale_custom(b_chan, bmin, bmax, invert=True)

    rgb_normalized = np.stack([r_scaled, g_scaled, b_scaled], axis=-1)
    rgb_normalized = np.nan_to_num(rgb_normalized, nan=0.0, posinf=1.0, neginf=0.0)
    return (rgb_normalized * 255).astype(np.uint8)


def parse_dt_from_zip_name(zip_name: str) -> dt.datetime | None:
    """
    Estrae datetime da:
    MSG3-SEVI-...-20260117021242.930000000Z-NA.zip
    """
    m = re.search(r"-(\d{14})\.\d+Z-NA\.zip$", zip_name)
    if not m:
        return None
    try:
        return dt.datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def slot_dt_from_zip_name(zip_name: str) -> dt.datetime | None:
    parsed = parse_dt_from_zip_name(zip_name)
    if parsed is None:
        return None
    return parsed.replace(minute=(parsed.minute // 5) * 5, second=0, microsecond=0)


def _to_numpy_2d(data) -> np.ndarray:
    if hasattr(data, "compute"):
        data = data.compute()
    return np.asarray(data, dtype=np.float32)


def spatial_cut_geos(
    da: xr.DataArray,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    x_increasing: bool = True,
    y_increasing: bool = True,
    flip_north_up: bool = False,
) -> xr.DataArray:
    """
    Stessa logica di taglio usata in download_airmassRGB.py:
    selezione geostazionaria con coordinate x_geostationary/y_geostationary.
    """
    x_center, y_center = default_basem_obj(9.5, 0)
    x0, y0 = default_basem_obj(lon_min, lat_min)
    x1, y1 = default_basem_obj(lon_max, lat_max)

    x0 -= x_center
    y0 -= y_center
    x1 -= x_center
    y1 -= y_center

    if x_increasing:
        x_start, x_end = (min(x0, x1), max(x0, x1))
    else:
        x_start, x_end = (max(x0, x1), min(x0, x1))

    if y_increasing:
        y_start, y_end = (min(y0, y1), max(y0, y1))
    else:
        y_start, y_end = (max(y0, y1), min(y0, y1))

    # Manteniamo l'esatta selezione dello script storico.
    _ = (x_start, x_end, y_start, y_end)
    da_cut = da.sel(
        x_geostationary=slice(x1, x0),
        y_geostationary=slice(y0, y1),
    )

    if flip_north_up:
        da_cut = da_cut.isel(y_geostationary=slice(None, None, -1))
        da_cut = da_cut.isel(x_geostationary=slice(None, None, -1))

    return da_cut


def extract_nat_from_zip(zip_path: Path, tmp_dir: Path) -> Path:
    with zipfile.ZipFile(zip_path, "r") as zf:
        nat_members = [n for n in zf.namelist() if n.lower().endswith(".nat")]
        if not nat_members:
            raise RuntimeError("Nessun file .nat trovato nello zip")
        nat_member = nat_members[0]
        out_path = tmp_dir / Path(nat_member).name
        with zf.open(nat_member) as src, out_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    return out_path


def process_one_zip(zip_path: Path, output_dir: Path) -> Path:
    # Fast path: evita riprocessamento completo in caso di restart.
    t_hint = slot_dt_from_zip_name(zip_path.name)
    if t_hint is not None:
        out_hint = output_dir / f"airmass_rgb_{t_hint.strftime('%Y%m%d_%H%M')}.png"
        if out_hint.exists() and out_hint.stat().st_size > 0:
            return out_hint

    with tempfile.TemporaryDirectory(prefix="hrseviri_nat_") as td:
        nat_path = extract_nat_from_zip(zip_path, Path(td))

        scn = Scene(reader="seviri_l1b_native", filenames=[str(nat_path)])
        scn.load(list(CHANNELS))

        # DataArray multi-canale su coordinate geostazionarie, per replicare il taglio storico.
        ir097_da = scn["IR_097"].rename({"x": "x_geostationary", "y": "y_geostationary"})
        ir108_da = scn["IR_108"].rename({"x": "x_geostationary", "y": "y_geostationary"})
        wv062_da = scn["WV_062"].rename({"x": "x_geostationary", "y": "y_geostationary"})
        wv073_da = scn["WV_073"].rename({"x": "x_geostationary", "y": "y_geostationary"})

        ds_sub = xr.concat([ir097_da, ir108_da, wv062_da, wv073_da], dim="variable")
        ds_sub = ds_sub.assign_coords(variable=["IR_097", "IR_108", "WV_062", "WV_073"])
        ds_sub = ds_sub.transpose("y_geostationary", "x_geostationary", "variable")

        x_increasing = bool(ds_sub.x_geostationary[0] < ds_sub.x_geostationary[-1])
        y_increasing = bool(ds_sub.y_geostationary[0] < ds_sub.y_geostationary[-1])
        ds_sub_cut = spatial_cut_geos(
            ds_sub,
            lat_min,
            lat_max,
            lon_min,
            lon_max,
            x_increasing=x_increasing,
            y_increasing=y_increasing,
            flip_north_up=True,
        )

        wv62 = _to_numpy_2d(ds_sub_cut.sel(variable="WV_062").data)
        wv73 = _to_numpy_2d(ds_sub_cut.sel(variable="WV_073").data)
        ir097 = _to_numpy_2d(ds_sub_cut.sel(variable="IR_097").data)
        ir108 = _to_numpy_2d(ds_sub_cut.sel(variable="IR_108").data)

        arr_f32 = np.stack(
            [
                wv62 - wv73,
                ir097 - ir108,
                wv62,
            ],
            axis=-1,
        )

        arr_8bit = to_8bit_airmass(arr_f32)

        t = scn["WV_062"].attrs.get("start_time")
        if not isinstance(t, dt.datetime):
            t = slot_dt_from_zip_name(zip_path.name)
        if t is None:
            raise RuntimeError("Impossibile ricavare timestamp per il naming output")
        out_name = f"airmass_rgb_{t.strftime('%Y%m%d_%H%M')}.png"
        out_path = output_dir / out_name
        if out_path.exists() and out_path.stat().st_size > 0:
            return out_path
        imageio.imwrite(out_path, arr_8bit)
        return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crea Air Mass RGB da ZIP HRSEVIRI locali (.nat in archivio)."
    )
    parser.add_argument(
        "--input-dir",
        default="/media/isacDisk2/from_eumetsat/hrseviri_full_disk",
        help="Cartella dei .zip HRSEVIRI",
    )
    parser.add_argument(
        "--output-dir",
        default="./from_eumetsat_airmassRGB",
        help="Cartella output PNG airmass_rgb_*.png",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Numero massimo di file zip da processare",
    )
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_files = sorted(input_dir.glob("*.zip"), key=lambda p: (parse_dt_from_zip_name(p.name) or dt.datetime.min, p.name))
    if not zip_files:
        raise SystemExit(f"Nessun .zip trovato in {input_dir}")

    filtered = []
    for zp in zip_files:
        zdt = parse_dt_from_zip_name(zp.name)
        if zdt is None:
            continue
        filtered.append(zp)

    if args.limit is not None:
        filtered = filtered[: args.limit]

    print(f"Trovati {len(filtered)} file zip da processare")
    if not filtered:
        return

    ok = 0
    for i, zp in enumerate(filtered, start=1):
        print(f"[{i}/{len(filtered)}] {zp.name}")
        try:
            out_path = process_one_zip(
                zp,
                output_dir=output_dir,
            )
            ok += 1
            print(f"  -> salvato: {out_path.name}")
        except Exception as exc:
            print(f"  -> errore: {exc}")
            print(traceback.format_exc())

    print(f"Completato. PNG creati: {ok}/{len(filtered)}")


if __name__ == "__main__":
    main()
