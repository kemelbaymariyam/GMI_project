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
MIN_VALID_FRACTION_DEFAULT = 0.70


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_day_from_filename(path: Path) -> pd.Timestamp:
    m = re.search(r"(20\d{6})", path.name)
    if not m:
        raise ValueError(f"Could not parse date from filename: {path.name}")
    return pd.to_datetime(m.group(1), format="%Y%m%d").normalize()


def list_patch_val_files(val_patch_root: Path) -> List[Path]:
    return sorted(val_patch_root.rglob("*.npz"))


def list_nc_files(input_root: Path) -> List[Path]:
    return sorted(input_root.rglob("*.nc"))


def open_merged_file(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ds = xr.open_dataset(path)
    for var in ["input_GMI", "input_static", "target_GMI"]:
        if var not in ds:
            raise KeyError(f"{var} not found in {path}")
    x_gmi = ds["input_GMI"].values.astype(np.float32)
    x_static = ds["input_static"].values.astype(np.float32)
    y = ds["target_GMI"].values.astype(np.float32)
    ds.close()
    return x_gmi, x_static, y


def target_valid_fraction(y_patch: np.ndarray) -> float:
    valid_pixel = np.any(np.isfinite(y_patch), axis=0)
    return float(np.mean(valid_pixel))


def keep_patch(y_patch: np.ndarray, min_valid_fraction: float) -> bool:
    valid_frac = target_valid_fraction(y_patch)
    if valid_frac == 0.0:
        return False
    return valid_frac >= min_valid_fraction


def possible_stride_positions(length: int, patch_size: int, stride: int) -> List[int]:
    if length < patch_size:
        return []
    return list(range(0, length - patch_size + 1, stride))


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
        description="Build fixed balanced validation H5 by random stride-aligned sampling from original validation days"
    )
    parser.add_argument(
        "--val-patch-root",
        type=Path,
        required=True,
        help="Existing validation patch folder, e.g. patches_split_byday_cnnprep/val",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Original merged full-global CNN-preprocessed NC root",
    )
    parser.add_argument("--output-h5", type=Path, required=True)
    parser.add_argument("--num-patches", type=int, default=50000)
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE_DEFAULT)
    parser.add_argument("--stride", type=int, default=STRIDE_DEFAULT)
    parser.add_argument("--min-valid-fraction", type=float, default=MIN_VALID_FRACTION_DEFAULT)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--use-static", action="store_true")
    parser.add_argument(
        "--max-tries-per-day",
        type=int,
        default=200000,
        help="Maximum random attempts per day",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # 1) Extract exact validation days from existing val patch filenames
    val_patch_files = list_patch_val_files(args.val_patch_root)
    if not val_patch_files:
        raise FileNotFoundError(f"No .npz validation patch files found under {args.val_patch_root}")

    val_days = sorted({parse_day_from_filename(p) for p in val_patch_files})
    val_day_set = set(val_days)

    print(f"Found {len(val_patch_files)} validation patch shard files")
    print(f"Unique validation days extracted: {len(val_days)}")

    # 2) Match those dates to original NC files
    nc_files_all = list_nc_files(args.input_root)
    if not nc_files_all:
        raise FileNotFoundError(f"No .nc files found under {args.input_root}")

    nc_files = []
    for path in nc_files_all:
        try:
            day = parse_day_from_filename(path)
        except Exception:
            continue
        if day in val_day_set:
            nc_files.append(path)

    if not nc_files:
        raise ValueError("No matching NC files found for extracted validation days")

    nc_files = sorted(nc_files)
    print(f"Matching validation NC files found: {len(nc_files)}")

    # balanced target per day
    patches_per_day_target = math.ceil(args.num_patches / len(nc_files))
    print(f"Approx target patches per day: {patches_per_day_target}")

    ensure_dir(args.output_h5.parent)

    total_written = 0
    initialized = False

    with h5py.File(args.output_h5, "w") as f:
        f.attrs["num_patches_target"] = args.num_patches
        f.attrs["patch_size"] = args.patch_size
        f.attrs["stride"] = args.stride
        f.attrs["min_valid_fraction"] = args.min_valid_fraction
        f.attrs["seed"] = args.seed
        f.attrs["use_static"] = args.use_static
        f.attrs["sampling"] = "balanced random stride-aligned sampling from original validation days"

        for day_idx, path in enumerate(nc_files, start=1):
            if total_written >= args.num_patches:
                break

            target_day = parse_day_from_filename(path)
            print(f"\n[{day_idx}/{len(nc_files)}] opening {path.name}")

            x_gmi, x_static, y = open_merged_file(path)
            _, ny, nx = y.shape

            r_positions = possible_stride_positions(ny, args.patch_size, args.stride)
            c_positions = possible_stride_positions(nx, args.patch_size, args.stride)

            if not r_positions or not c_positions:
                print("  skipped: grid too small for patch size")
                continue

            # initialize HDF5 datasets once we know shapes
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

            # all possible stride-aligned locations for this day
            candidate_coords = [(r, c) for r in r_positions for c in c_positions]
            rng.shuffle(candidate_coords)

            remaining_global = args.num_patches - total_written
            target_here = min(patches_per_day_target, remaining_global)

            accepted_here = 0
            tries = 0

            for (r, c) in candidate_coords:
                if accepted_here >= target_here:
                    break
                if tries >= args.max_tries_per_day:
                    break

                y_patch = y[:, r:r+args.patch_size, c:c+args.patch_size]
                if not keep_patch(y_patch, min_valid_fraction=args.min_valid_fraction):
                    tries += 1
                    continue

                xg_patch = x_gmi[:, r:r+args.patch_size, c:c+args.patch_size]
                xs_patch = None
                if args.use_static:
                    xs_patch = x_static[:, r:r+args.patch_size, c:c+args.patch_size]

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
                tries += 1

                if total_written % 500 == 0:
                    print(f"  total written: {total_written}/{args.num_patches}")

                if total_written >= args.num_patches:
                    break

            print(f"  accepted from this day: {accepted_here}")
            print(f"  tried candidate patches: {tries}")
            print(f"  total written so far: {total_written}/{args.num_patches}")

        f.attrs["num_patches_actual"] = total_written

    print(f"\nSaved fixed validation H5 to: {args.output_h5}")
    print(f"Actual patches written: {total_written}")


if __name__ == "__main__":
    main()