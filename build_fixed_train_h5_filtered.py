#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import math
import random
import re
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import pandas as pd
import xarray as xr


PATCH_SIZE_DEFAULT = 40
STRIDE_DEFAULT = 20


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_day_from_filename(path: Path) -> pd.Timestamp:
    m = re.search(r"(20\d{6})", path.name)
    if not m:
        raise ValueError(f"Could not parse date from filename: {path.name}")
    return pd.to_datetime(m.group(1), format="%Y%m%d").normalize()


def list_patch_files(root: Path) -> List[Path]:
    return sorted(root.rglob("*.npz"))


def list_nc_files(root: Path) -> List[Path]:
    return sorted(root.rglob("*.nc"))


def open_merged_file(path: Path):
    ds = xr.open_dataset(path)
    for var in ["input_GMI", "input_static", "target_GMI", "lat"]:
        if var not in ds:
            raise KeyError(f"{var} not found in {path}")
    x_gmi = ds["input_GMI"].values.astype(np.float32)
    x_static = ds["input_static"].values.astype(np.float32)
    y = ds["target_GMI"].values.astype(np.float32)
    lat = ds["lat"].values.astype(np.float32)
    ds.close()
    return x_gmi, x_static, y, lat


def possible_stride_positions(length: int, patch_size: int, stride: int) -> List[int]:
    if length < patch_size:
        return []
    return list(range(0, length - patch_size + 1, stride))


def target_valid_fraction(y_patch: np.ndarray) -> float:
    valid_pixel = np.any(np.isfinite(y_patch), axis=0)
    return float(np.mean(valid_pixel))


def permanent_invalid_fraction_from_lat(
    lat_arr: np.ndarray,
    r: int,
    c: int,
    patch_size: int,
    lat_min: float,
    lat_max: float,
) -> float:
    if lat_arr.ndim == 1:
        lat_patch = lat_arr[r:r + patch_size]
        invalid_rows = (lat_patch < lat_min) | (lat_patch > lat_max)
        return float(np.mean(invalid_rows))

    elif lat_arr.ndim == 2:
        lat_patch = lat_arr[r:r + patch_size, c:c + patch_size]
        invalid = (lat_patch < lat_min) | (lat_patch > lat_max)
        return float(np.mean(invalid))

    else:
        raise ValueError(f"Unsupported lat shape: {lat_arr.shape}")


def patch_center_lat_ok(
    lat_arr: np.ndarray,
    r: int,
    c: int,
    patch_size: int,
    lat_min: float,
    lat_max: float,
) -> bool:
    rr = r + patch_size // 2

    if lat_arr.ndim == 1:
        center_lat = float(lat_arr[rr])
    elif lat_arr.ndim == 2:
        cc = c + patch_size // 2
        center_lat = float(lat_arr[rr, cc])
    else:
        raise ValueError(f"Unsupported lat shape: {lat_arr.shape}")

    return (lat_min <= center_lat <= lat_max)


def append_one_sample(
    f: h5py.File,
    xg_patch: np.ndarray,
    y_patch: np.ndarray,
    xs_patch: np.ndarray | None,
    rc: Tuple[int, int],
    source_file: str,
    target_day: str,
    use_static: bool,
) -> None:
    old_n = f["X_gmi"].shape[0]
    new_n = old_n + 1

    f["X_gmi"].resize((new_n, *f["X_gmi"].shape[1:]))
    f["X_gmi"][old_n] = xg_patch.astype(np.float32)

    f["Y"].resize((new_n, *f["Y"].shape[1:]))
    f["Y"][old_n] = y_patch.astype(np.float32)

    if use_static and xs_patch is not None:
        f["X_static"].resize((new_n, *f["X_static"].shape[1:]))
        f["X_static"][old_n] = xs_patch.astype(np.float32)

    f["patch_rc"].resize((new_n, 2))
    f["patch_rc"][old_n] = np.asarray(rc, dtype=np.int32)

    f["source_file"].resize((new_n,))
    f["source_file"][old_n] = source_file

    f["target_day"].resize((new_n,))
    f["target_day"][old_n] = target_day


def main():
    parser = argparse.ArgumentParser(
        description="Build fixed random training H5 with filtering for permanent invalid latitude bands"
    )
    parser.add_argument("--train-patch-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-h5", type=Path, required=True)

    parser.add_argument("--num-patches", type=int, default=360000)
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE_DEFAULT)
    parser.add_argument("--stride", type=int, default=STRIDE_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-static", action="store_true")

    parser.add_argument("--lat-min", type=float, default=-70.0)
    parser.add_argument("--lat-max", type=float, default=70.0)

    parser.add_argument("--min-target-valid-fraction", type=float, default=0.70)
    parser.add_argument("--max-permanent-invalid-fraction", type=float, default=0.30)

    parser.add_argument("--max-tries-per-day", type=int, default=300000)

    args = parser.parse_args()
    rng = random.Random(args.seed)

    # 1) Recover train days from existing train patch filenames
    train_patch_files = list_patch_files(args.train_patch_root)
    if not train_patch_files:
        raise FileNotFoundError(f"No train patch files found under {args.train_patch_root}")

    train_days = sorted({parse_day_from_filename(p) for p in train_patch_files})
    train_day_set = set(train_days)

    print(f"Found {len(train_patch_files)} train patch shard files")
    print(f"Unique train days extracted: {len(train_days)}")

    # 2) Match original NC files for those days
    nc_files_all = list_nc_files(args.input_root)
    if not nc_files_all:
        raise FileNotFoundError(f"No NC files found under {args.input_root}")

    nc_files = []
    for path in nc_files_all:
        try:
            day = parse_day_from_filename(path)
        except Exception:
            continue
        if day in train_day_set:
            nc_files.append(path)

    if not nc_files:
        raise ValueError("No matching NC files found for extracted train days")

    nc_files = sorted(nc_files)
    rng.shuffle(nc_files)

    print(f"Matching train NC files found: {len(nc_files)}")

    per_day_target = math.ceil(args.num_patches / len(nc_files))
    print(f"Approx target patches per day: {per_day_target}")

    ensure_dir(args.output_h5.parent)

    total_written = 0
    initialized = False

    with h5py.File(args.output_h5, "w") as f:
        f.attrs["num_patches_target"] = args.num_patches
        f.attrs["patch_size"] = args.patch_size
        f.attrs["stride"] = args.stride
        f.attrs["seed"] = args.seed
        f.attrs["use_static"] = args.use_static
        f.attrs["lat_min"] = args.lat_min
        f.attrs["lat_max"] = args.lat_max
        f.attrs["min_target_valid_fraction"] = args.min_target_valid_fraction
        f.attrs["max_permanent_invalid_fraction"] = args.max_permanent_invalid_fraction

        for day_idx, path in enumerate(nc_files, start=1):
            if total_written >= args.num_patches:
                break

            target_day = parse_day_from_filename(path)
            print(f"\n[{day_idx}/{len(nc_files)}] opening {path.name}")

            x_gmi, x_static, y, lat = open_merged_file(path)
            _, ny, nx = y.shape

            r_positions = possible_stride_positions(ny, args.patch_size, args.stride)
            c_positions = possible_stride_positions(nx, args.patch_size, args.stride)

            if not r_positions or not c_positions:
                print("  skipped: grid too small")
                continue

            if not initialized:
                xg_shape = (x_gmi.shape[0], args.patch_size, args.patch_size)
                y_shape = (y.shape[0], args.patch_size, args.patch_size)

                f.create_dataset(
                    "X_gmi",
                    shape=(0, *xg_shape),
                    maxshape=(None, *xg_shape),
                    dtype=np.float32,
                    chunks=True,
                    compression="lzf",
                )
                f.create_dataset(
                    "Y",
                    shape=(0, *y_shape),
                    maxshape=(None, *y_shape),
                    dtype=np.float32,
                    chunks=True,
                    compression="lzf",
                )

                if args.use_static:
                    xs_shape = (x_static.shape[0], args.patch_size, args.patch_size)
                    f.create_dataset(
                        "X_static",
                        shape=(0, *xs_shape),
                        maxshape=(None, *xs_shape),
                        dtype=np.float32,
                        chunks=True,
                        compression="lzf",
                    )

                f.create_dataset(
                    "patch_rc",
                    shape=(0, 2),
                    maxshape=(None, 2),
                    dtype=np.int32,
                    chunks=True,
                    compression="lzf",
                )

                str_dt = h5py.string_dtype(encoding="utf-8")
                f.create_dataset("source_file", shape=(0,), maxshape=(None,), dtype=str_dt, chunks=True)
                f.create_dataset("target_day", shape=(0,), maxshape=(None,), dtype=str_dt, chunks=True)

                initialized = True

            candidate_coords = [(r, c) for r in r_positions for c in c_positions]
            rng.shuffle(candidate_coords)

            remaining_global = args.num_patches - total_written
            target_here = min(per_day_target, remaining_global)

            accepted_here = 0
            tries = 0

            for (r, c) in candidate_coords:
                if accepted_here >= target_here:
                    break
                if tries >= args.max_tries_per_day:
                    break

                tries += 1

                # 1) keep patch center away from permanently dead bands
                if not patch_center_lat_ok(
                    lat_arr=lat,
                    r=r,
                    c=c,
                    patch_size=args.patch_size,
                    lat_min=args.lat_min,
                    lat_max=args.lat_max,
                ):
                    continue

                # 2) reject patches dominated by permanent invalid latitude zone
                perm_invalid_frac = permanent_invalid_fraction_from_lat(
                    lat_arr=lat,
                    r=r,
                    c=c,
                    patch_size=args.patch_size,
                    lat_min=args.lat_min,
                    lat_max=args.lat_max,
                )
                if perm_invalid_frac > args.max_permanent_invalid_fraction:
                    continue

                # 3) require enough valid target coverage
                y_patch = y[:, r:r + args.patch_size, c:c + args.patch_size]
                valid_frac = target_valid_fraction(y_patch)
                if valid_frac < args.min_target_valid_fraction:
                    continue

                xg_patch = x_gmi[:, r:r + args.patch_size, c:c + args.patch_size]
                xs_patch = None
                if args.use_static:
                    xs_patch = x_static[:, r:r + args.patch_size, c:c + args.patch_size]

                append_one_sample(
                    f=f,
                    xg_patch=xg_patch,
                    y_patch=y_patch,
                    xs_patch=xs_patch,
                    rc=(r, c),
                    source_file=str(path),
                    target_day=str(target_day.date()),
                    use_static=args.use_static,
                )

                accepted_here += 1
                total_written += 1

                if total_written % 1000 == 0:
                    print(f"  total written: {total_written}/{args.num_patches}")

                if total_written >= args.num_patches:
                    break

            print(f"  accepted from this day: {accepted_here}")
            print(f"  tries used: {tries}")
            print(f"  total written so far: {total_written}/{args.num_patches}")

        f.attrs["num_patches_actual"] = total_written

    print(f"\nSaved fixed train H5 to: {args.output_h5}")
    print(f"Actual patches written: {total_written}")


if __name__ == "__main__":
    main()