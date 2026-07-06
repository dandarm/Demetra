#!/usr/bin/env python3
"""Convert a cyclone tracking CSV to a NetCDF file."""

import argparse
import importlib.util
from pathlib import Path
from typing import Optional

import pandas as pd
import xarray as xr


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a tracking CSV file to NetCDF, preserving all rows."
    )
    parser.add_argument("input_csv", type=Path, help="Path to the input CSV file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Path to the output NetCDF file. Defaults to input name with .nc suffix.",
    )
    parser.add_argument(
        "--title",
        default="Cyclone track",
        help="Global NetCDF title attribute.",
    )
    parser.add_argument(
        "--source",
        default="DeMeTra tracking inference CSV",
        help="Global NetCDF source attribute.",
    )
    return parser


def infer_output_path(input_csv: Path, output_path: Optional[Path]) -> Path:
    if output_path is not None:
        return output_path
    return input_csv.with_suffix(".nc")


def csv_to_dataset(df: pd.DataFrame, title: str, source: str) -> xr.Dataset:
    if "datetime" not in df.columns:
        raise ValueError("Missing required column: 'datetime'")

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    data_vars = {}
    for column in df.columns:
        if column == "datetime":
            continue
        data_vars[column] = ("time", df[column].to_numpy())

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={"time": df["datetime"].to_numpy()},
        attrs={
            "title": title,
            "source": source,
            "featureType": "trajectory",
            "history": "Converted from CSV with csv_track_to_netcdf.py",
        },
    )

    ds["time"].attrs = {"standard_name": "time"}

    if "pred_lat" in ds:
        ds["pred_lat"].attrs = {
            "standard_name": "latitude",
            "units": "degrees_north",
        }
    if "pred_lon" in ds:
        ds["pred_lon"].attrs = {
            "standard_name": "longitude",
            "units": "degrees_east",
        }
    if "has_cyclone" in ds:
        ds["has_cyclone"].attrs = {
            "long_name": "cyclone detected flag",
            "flag_values": [0, 1],
            "flag_meanings": "no_cyclone cyclone",
        }

    return ds


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_csv = args.input_csv.resolve()
    output_nc = infer_output_path(input_csv, args.output.resolve() if args.output else None)

    df = pd.read_csv(input_csv)
    ds = csv_to_dataset(df, title=args.title, source=args.source)

    has_h5netcdf = importlib.util.find_spec("h5netcdf") is not None
    if has_h5netcdf:
        encoding = {name: {"zlib": True, "complevel": 4} for name in ds.data_vars}
        ds.to_netcdf(output_nc, engine="h5netcdf", encoding=encoding)
    else:
        ds.to_netcdf(output_nc)

    print(f"Written NetCDF: {output_nc}")
    print(f"Rows exported: {ds.sizes['time']}")


if __name__ == "__main__":
    main()
