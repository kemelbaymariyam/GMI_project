#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create FULL-GLOBAL merged model-IO files for GMI training, BEFORE patch extraction.

Reference-style design
----------------------
For each target day t:
  Input sequence (6 daily slices):
    t-5, t-4, t-3, t-2  -> Gaussian-weighted daily GMI
    t-1, t              -> Daily with-gaps GMI

  Static channels:
    FRLAND
    FRLANDICE

  Target:
    t                   -> Gaussian-weighted daily GMI

Saved output
------------
One NetCDF file per target day with:
  - input_GMI[input_channel, lat, lon]     -> 78 channels = 6 * 13
  - input_static[static_channel, lat, lon] -> 2 channels  = FRLAND, FRLANDICE
  - target_GMI[target_channel, lat, lon]   -> 13 channels
  - input_time[input_step]
  - input_role[input_step]

This is the FULL-GLOBAL merged training representation.
You can inspect these files first, and only after that extract 40x40 patches.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import xarray as xr

# ============================================================
# USER SETTINGS
# ============================================================
GAUSSIAN_ROOT = Path("/lustre/home/mariyam/GMI_1C-R/swath_gaussian_scanfair_reduced")
WITH_GAPS_ROOT = Path("/lustre/home/mariyam/GMI_1C-R/gmi_daily_gridded_gaps")
FRLAND_FILE = Path("/lustre/home/mariyam/FRLAND_on_GMIgrid.nc")
FRLANDICE_FILE = Path("/lustre/home/mariyam/FRLANDICE_on_GMIgrid.nc")

OUTPUT_ROOT = Path("/lustre/home/mariyam/GMI_1C-R/model_io_daily_fullglobal")

ALL_YEARS = {2016, 2017, 2018, 2019, 2020}

GAUSSIAN_OFFSETS = [-5, -4, -3, -2]
WITH_GAPS_OFFSETS = [-1, 0]

ZLIB = True
COMPLEVEL = 4


# ============================================================
# BASIC HELPERS
# ============================================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_day_from_filename(path: Path) -> pd.Timestamp:
    m = re.search(r"(20\d{6})", path.name)
    if not m:
        raise ValueError(f"Could not parse date from filename: {path.name}")
    return pd.to_datetime(m.group(1), format="%Y%m%d").normalize()


def list_daily_files(root: Path, years: set[int]) -> Dict[pd.Timestamp, Path]:
    out: Dict[pd.Timestamp, Path] = {}
    for p in root.rglob("*.nc"):
        try:
            day = parse_day_from_filename(p)
        except Exception:
            continue
        if day.year in years:
            out[day] = p
    return out


def open_land_mask(mask_file: Path, var_name: str) -> xr.DataArray:
    ds = xr.open_dataset(mask_file)
    if var_name not in ds:
        raise KeyError(f"{var_name} not found in {mask_file}")
    da = ds[var_name]
    if "time" in da.dims:
        da = da.isel(time=0)
    return da.astype(np.float32)


def get_gmi_13ch(ds: xr.Dataset) -> xr.DataArray:
    if "S1_Tc" not in ds or "S2_Tc" not in ds:
        raise KeyError("Dataset must contain both S1_Tc and S2_Tc")

    s1 = ds["S1_Tc"].astype(np.float32)
    s2 = ds["S2_Tc"].astype(np.float32)

    s1_names = [f"S1_{str(c)}" for c in s1["channel_S1"].values]
    s2_names = [f"S2_{str(c)}" for c in s2["channel_S2"].values]

    s1 = s1.rename({"channel_S1": "gmi_channel"}).assign_coords(gmi_channel=s1_names)
    s2 = s2.rename({"channel_S2": "gmi_channel"}).assign_coords(gmi_channel=s2_names)

    gmi = xr.concat([s1, s2], dim="gmi_channel")
    return gmi.transpose("gmi_channel", "lat", "lon")


def open_daily_gmi(path: Path) -> xr.DataArray:
    ds = xr.open_dataset(path)
    return get_gmi_13ch(ds)


def require_same_grid(reference: xr.DataArray, other: xr.DataArray, label: str) -> None:
    same_lat = np.array_equal(reference["lat"].values, other["lat"].values)
    same_lon = np.array_equal(reference["lon"].values, other["lon"].values)
    if not (same_lat and same_lon):
        raise ValueError(f"Grid mismatch for {label}")


def build_input_channel_names(
    input_days: List[pd.Timestamp],
    input_roles: List[str],
    base_channels: List[str],
    target_day: pd.Timestamp,
) -> List[str]:
    names: List[str] = []
    for d, role in zip(input_days, input_roles):
        rel = (d - target_day).days
        day_tag = f"t{rel:+d}".replace("+", "p")
        for ch in base_channels:
            names.append(f"{day_tag}_{role}_{ch}")
    return names


# ============================================================
# BUILD ONE FULL-GLOBAL MERGED FILE
# ============================================================
def build_one_target_day(
    target_day: pd.Timestamp,
    gaussian_files: Dict[pd.Timestamp, Path],
    gaps_files: Dict[pd.Timestamp, Path],
    frland: xr.DataArray,
    frlandice: xr.DataArray,
) -> Optional[xr.Dataset]:
    gaussian_days = [target_day + pd.Timedelta(days=o) for o in GAUSSIAN_OFFSETS]
    gaps_days = [target_day + pd.Timedelta(days=o) for o in WITH_GAPS_OFFSETS]

    needed_gauss = gaussian_days + [target_day]
    needed_gaps = gaps_days

    missing = []
    for d in needed_gauss:
        if d not in gaussian_files:
            missing.append(f"Gaussian missing: {d.date()}")
    for d in needed_gaps:
        if d not in gaps_files:
            missing.append(f"With-gaps missing: {d.date()}")

    if missing:
        print(f"skip {target_day.date()} -> " + "; ".join(missing))
        return None

    target_da = open_daily_gmi(gaussian_files[target_day])  # [13, lat, lon]

    require_same_grid(target_da, frland, "FRLAND")
    require_same_grid(target_da, frlandice, "FRLANDICE")

    input_days = gaussian_days + gaps_days
    input_roles = ["gaussian"] * len(gaussian_days) + ["with_gaps"] * len(gaps_days)

    seq_list: List[xr.DataArray] = []
    for d, role in zip(input_days, input_roles):
        da = open_daily_gmi(gaussian_files[d] if role == "gaussian" else gaps_files[d])
        require_same_grid(target_da, da, f"{role} {d.date()}")
        seq_list.append(da)

    seq = xr.concat(seq_list, dim="input_step")  # [6, 13, lat, lon]

    base_channels = [str(v) for v in target_da["gmi_channel"].values]
    input_channel_names = build_input_channel_names(
        input_days=input_days,
        input_roles=input_roles,
        base_channels=base_channels,
        target_day=target_day,
    )

    input_arr = seq.values.astype(np.float32)  # [6, 13, lat, lon]
    nstep, nchan, ny, nx = input_arr.shape
    input_flat = input_arr.reshape(nstep * nchan, ny, nx)  # [78, lat, lon]

    static_arr = np.stack(
        [frland.values.astype(np.float32), frlandice.values.astype(np.float32)],
        axis=0,
    )  # [2, lat, lon]

    ds_out = xr.Dataset(
        {
            "input_GMI": xr.DataArray(
                input_flat,
                dims=("input_channel", "lat", "lon"),
                coords={
                    "input_channel": input_channel_names,
                    "lat": target_da["lat"].values,
                    "lon": target_da["lon"].values,
                },
            ),
            "input_static": xr.DataArray(
                static_arr,
                dims=("static_channel", "lat", "lon"),
                coords={
                    "static_channel": ["FRLAND", "FRLANDICE"],
                    "lat": target_da["lat"].values,
                    "lon": target_da["lon"].values,
                },
            ),
            "target_GMI": xr.DataArray(
                target_da.values.astype(np.float32),
                dims=("target_channel", "lat", "lon"),
                coords={
                    "target_channel": base_channels,
                    "lat": target_da["lat"].values,
                    "lon": target_da["lon"].values,
                },
            ),
            "input_time": xr.DataArray(
                np.asarray([str(d.date()) for d in input_days], dtype="U10"),
                dims=("input_step",),
                coords={"input_step": np.arange(len(input_days))},
            ),
            "input_role": xr.DataArray(
                np.asarray(input_roles, dtype="U16"),
                dims=("input_step",),
                coords={"input_step": np.arange(len(input_days))},
            ),
        }
    )

    ds_out.attrs["target_day"] = str(target_day.date())
    ds_out.attrs["input_design"] = "t-5..t-2 gaussian, t-1..t with_gaps"
    ds_out.attrs["gmi_channels_per_day"] = 13
    ds_out.attrs["n_input_steps"] = len(input_days)
    ds_out.attrs["n_input_channels_total"] = int(len(input_channel_names))
    ds_out.attrs["n_static_channels"] = 2
    ds_out.attrs["n_target_channels"] = 13

    return ds_out


def save_one(ds: xr.Dataset, out_root: Path, target_day: pd.Timestamp) -> Path:
    year_dir = out_root / f"{target_day.year}"
    ensure_dir(year_dir)
    out_path = year_dir / f"GMI_fullModelIO_{target_day:%Y%m%d}.nc"

    encoding = {v: {"zlib": ZLIB, "complevel": COMPLEVEL} for v in ds.data_vars}
    ds.to_netcdf(out_path, encoding=encoding)
    return out_path


# ============================================================
# DRIVER
# ============================================================
def run_pipeline(start_day: Optional[str], end_day: Optional[str]) -> None:
    ensure_dir(OUTPUT_ROOT)

    print("Indexing daily files...")
    gaussian_files = list_daily_files(GAUSSIAN_ROOT, ALL_YEARS)
    gaps_files = list_daily_files(WITH_GAPS_ROOT, ALL_YEARS)

    print(f"Gaussian files indexed: {len(gaussian_files)}")
    print(f"With-gaps files indexed: {len(gaps_files)}")

    frland = open_land_mask(FRLAND_FILE, "FRLAND")
    frlandice = open_land_mask(FRLANDICE_FILE, "FRLANDICE")

    candidate_days = sorted(set(gaussian_files.keys()) & set(gaps_files.keys()))
    if start_day is not None:
        candidate_days = [d for d in candidate_days if d >= pd.Timestamp(start_day).normalize()]
    if end_day is not None:
        candidate_days = [d for d in candidate_days if d <= pd.Timestamp(end_day).normalize()]

    saved = 0
    skipped = 0

    for target_day in candidate_days:
        print(f"\n[target] {target_day.date()}")
        ds = build_one_target_day(
            target_day=target_day,
            gaussian_files=gaussian_files,
            gaps_files=gaps_files,
            frland=frland,
            frlandice=frlandice,
        )
        if ds is None:
            skipped += 1
            continue

        out_path = save_one(ds, OUTPUT_ROOT, target_day)
        print(f"saved -> {out_path}")
        saved += 1

    print(f"\nDone. saved={saved}, skipped={skipped}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Create full-global merged GMI model-IO files")
    p.add_argument("--start-day", type=str, default=None, help="Optional YYYY-MM-DD")
    p.add_argument("--end-day", type=str, default=None, help="Optional YYYY-MM-DD")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_pipeline(start_day=args.start_day, end_day=args.end_day)
