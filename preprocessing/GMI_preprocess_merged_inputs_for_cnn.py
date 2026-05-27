#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocess merged FULL-GLOBAL GMI model-IO files for CNN training.

What it does
------------
For each merged file, this script processes INPUT channels so the CNN does not
see raw NaNs:

1) input_GMI:
   - apply finite-width boundary padding around missing regions
   - replace any remaining NaNs with a dummy value (default: 0)

2) input_static:
   - left unchanged by default
   - optional simple NaN fill if you use --process-static

3) target_GMI:
   - left unchanged on purpose

Recommended for your case
-------------------------
- keep target_GMI unchanged for now
- preprocess input_GMI
- use pad_pixels = 4
- use dummy_value = 0

This lets you later:
- inspect target missingness
- filter patches using target validity
- build masked losses later if needed
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

ZLIB = True
COMPLEVEL = 4


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_day_from_filename(path: Path) -> pd.Timestamp:
    m = re.search(r"(20\d{6})", path.name)
    if not m:
        raise ValueError(f"Could not parse date from filename: {path.name}")
    return pd.to_datetime(m.group(1), format="%Y%m%d").normalize()


def list_merged_files(root: Path):
    return sorted(root.rglob("*.nc"))


def shift_with_nan(arr: np.ndarray, dr: int, dc: int) -> np.ndarray:
    out = np.full(arr.shape, np.nan, dtype=np.float32)

    r_src_start = max(0, -dr)
    r_src_end = min(arr.shape[0], arr.shape[0] - dr) if dr >= 0 else arr.shape[0]
    c_src_start = max(0, -dc)
    c_src_end = min(arr.shape[1], arr.shape[1] - dc) if dc >= 0 else arr.shape[1]

    r_dst_start = max(0, dr)
    c_dst_start = max(0, dc)

    r_len = r_src_end - r_src_start
    c_len = c_src_end - c_src_start

    if r_len > 0 and c_len > 0:
        out[r_dst_start:r_dst_start + r_len, c_dst_start:c_dst_start + c_len] = \
            arr[r_src_start:r_src_end, c_src_start:c_src_end]
    return out


def pad_missing_band_nearest_like(field: np.ndarray, pad_pixels: int, dummy_value: float) -> np.ndarray:
    arr = np.array(field, dtype=np.float32, copy=True)

    if not np.any(np.isfinite(arr)):
        return np.full(arr.shape, np.float32(dummy_value), dtype=np.float32)

    padded = arr.copy()

    neighbors = [
        (-1,  0), (1,  0), (0, -1), (0,  1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    ]

    for _ in range(pad_pixels):
        current_valid = np.isfinite(padded)

        shifted_vals = [shift_with_nan(padded, dr, dc) for dr, dc in neighbors]
        shifted_valid = [np.isfinite(sv) for sv in shifted_vals]

        adjacency = np.zeros(current_valid.shape, dtype=bool)
        for sm in shifted_valid:
            adjacency |= sm

        to_fill = (~current_valid) & adjacency
        if not np.any(to_fill):
            break

        fill_values = np.full(padded.shape, np.nan, dtype=np.float32)
        still_unfilled = to_fill.copy()

        for sv, sm in zip(shifted_vals, shifted_valid):
            can_use = still_unfilled & sm
            fill_values[can_use] = sv[can_use]
            still_unfilled[can_use] = False
            if not np.any(still_unfilled):
                break

        padded[to_fill] = fill_values[to_fill]

    padded = np.where(np.isfinite(padded), padded, np.float32(dummy_value))
    return padded.astype(np.float32)


def preprocess_input_gmi(input_gmi: xr.DataArray, pad_pixels: int, dummy_value: float) -> xr.DataArray:
    data = input_gmi.values.astype(np.float32)
    out = np.empty_like(data, dtype=np.float32)

    for i in range(data.shape[0]):
        out[i] = pad_missing_band_nearest_like(
            data[i],
            pad_pixels=pad_pixels,
            dummy_value=dummy_value,
        )

    return xr.DataArray(
        out,
        dims=input_gmi.dims,
        coords=input_gmi.coords,
        attrs=input_gmi.attrs,
        name=input_gmi.name,
    )


def fill_nan_simple(da: xr.DataArray, dummy_value: float) -> xr.DataArray:
    data = da.values.astype(np.float32)
    data = np.where(np.isfinite(data), data, np.float32(dummy_value))
    return xr.DataArray(
        data,
        dims=da.dims,
        coords=da.coords,
        attrs=da.attrs,
        name=da.name,
    )


def preprocess_one_file(
    in_path: Path,
    out_path: Path,
    pad_pixels: int,
    dummy_value: float,
    process_static: bool,
) -> None:
    ds = xr.open_dataset(in_path)

    for var in ["input_GMI", "input_static", "target_GMI"]:
        if var not in ds:
            raise KeyError(f"{var} not found in {in_path}")

    input_gmi_new = preprocess_input_gmi(ds["input_GMI"], pad_pixels=pad_pixels, dummy_value=dummy_value)

    if process_static:
        input_static_new = fill_nan_simple(ds["input_static"], dummy_value=dummy_value)
    else:
        input_static_new = ds["input_static"]

    target_new = ds["target_GMI"]

    ds_out = xr.Dataset(
        {
            "input_GMI": input_gmi_new,
            "input_static": input_static_new,
            "target_GMI": target_new,
        }
    )

    for extra_var in ["input_time", "input_role"]:
        if extra_var in ds:
            ds_out[extra_var] = ds[extra_var]

    ds_out.attrs.update(ds.attrs)
    ds_out.attrs["cnn_preprocessed"] = "true"
    ds_out.attrs["cnn_preprocess_dummy_value"] = float(dummy_value)
    ds_out.attrs["cnn_preprocess_pad_pixels"] = int(pad_pixels)
    ds_out.attrs["cnn_preprocess_note"] = (
        "input_GMI channels padded near missing regions and remaining NaNs replaced by dummy value"
    )

    ensure_dir(out_path.parent)
    encoding = {v: {"zlib": ZLIB, "complevel": COMPLEVEL} for v in ds_out.data_vars}
    ds_out.to_netcdf(out_path, encoding=encoding)


def build_output_path(output_root: Path, input_root: Path, in_path: Path) -> Path:
    rel = in_path.relative_to(input_root)
    return output_root / rel


def run_pipeline(
    input_root: Path,
    output_root: Path,
    pad_pixels: int,
    dummy_value: float,
    process_static: bool,
    start_day: Optional[str],
    end_day: Optional[str],
) -> None:
    files = list_merged_files(input_root)
    print(f"Found {len(files)} merged files")

    if start_day is not None:
        start_ts = pd.Timestamp(start_day).normalize()
        files = [p for p in files if parse_day_from_filename(p) >= start_ts]
    if end_day is not None:
        end_ts = pd.Timestamp(end_day).normalize()
        files = [p for p in files if parse_day_from_filename(p) <= end_ts]

    saved = 0
    skipped = 0

    for i, in_path in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] {in_path.name}")
        out_path = build_output_path(output_root, input_root, in_path)

        try:
            preprocess_one_file(
                in_path=in_path,
                out_path=out_path,
                pad_pixels=pad_pixels,
                dummy_value=dummy_value,
                process_static=process_static,
            )
            print(f"  saved -> {out_path}")
            saved += 1
        except Exception as e:
            print(f"  skipped -> {e}")
            skipped += 1

    print(f"\nDone. saved={saved}, skipped={skipped}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Preprocess merged GMI inputs for CNN training")
    p.add_argument("--input-root", type=Path,
                   default=Path("/lustre/home/mariyam/GMI_1C-R/model_io_daily_fullglobal"))
    p.add_argument("--output-root", type=Path,
                   default=Path("/lustre/home/mariyam/GMI_1C-R/model_io_daily_fullglobal_cnnprep"))
    p.add_argument("--pad-pixels", type=int, default=4)
    p.add_argument("--dummy-value", type=float, default=0.0)
    p.add_argument("--process-static", action="store_true")
    p.add_argument("--start-day", type=str, default=None)
    p.add_argument("--end-day", type=str, default=None)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_pipeline(
        input_root=args.input_root,
        output_root=args.output_root,
        pad_pixels=args.pad_pixels,
        dummy_value=args.dummy_value,
        process_static=args.process_static,
        start_day=args.start_day,
        end_day=args.end_day,
    )
