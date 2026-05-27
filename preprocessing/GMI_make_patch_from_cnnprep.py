#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create train/val/test patch shards from CNN-preprocessed merged GMI files
using a DAY-LEVEL split for train vs validation.

Split strategy
--------------
- test = all target days in 2020
- from 2016-2019, choose whole validation days
- all patches from a chosen day go entirely to val
- all patches from the remaining training days go to train

Recommended first setting
-------------------------
- patch size = 40
- stride = 20
- min_valid_fraction = 0.70
- val_day_fraction = 0.20
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

PATCH_SIZE_DEFAULT = 40
STRIDE_DEFAULT = 20
TRAIN_YEARS_DEFAULT = {2016, 2017, 2018, 2019}
TEST_YEARS_DEFAULT = {2020}
VAL_DAY_FRACTION_DEFAULT = 0.20
MIN_VALID_FRACTION_DEFAULT = 0.70
SEED_DEFAULT = 42


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_day_from_filename(path: Path) -> pd.Timestamp:
    m = re.search(r"(20\d{6})", path.name)
    if not m:
        raise ValueError(f"Could not parse date from filename: {path.name}")
    return pd.to_datetime(m.group(1), format="%Y%m%d").normalize()


def list_merged_files(root: Path) -> List[Path]:
    return sorted(root.rglob("*.nc"))


def open_merged_file(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ds = xr.open_dataset(path)
    for var in ["input_GMI", "input_static", "target_GMI"]:
        if var not in ds:
            raise KeyError(f"{var} not found in {path}")
    x_gmi = ds["input_GMI"].values.astype(np.float32)
    x_static = ds["input_static"].values.astype(np.float32)
    y = ds["target_GMI"].values.astype(np.float32)
    return x_gmi, x_static, y


def target_valid_fraction(y_patch: np.ndarray) -> float:
    valid_pixel = np.any(np.isfinite(y_patch), axis=0)
    return float(np.mean(valid_pixel))


def keep_patch(y_patch: np.ndarray, min_valid_fraction: float) -> bool:
    valid_frac = target_valid_fraction(y_patch)
    if valid_frac == 0.0:
        return False
    return valid_frac >= min_valid_fraction


def extract_patches(
    x_gmi: np.ndarray,
    x_static: np.ndarray,
    y: np.ndarray,
    patch_size: int,
    stride: int,
    min_valid_fraction: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _, ny, nx = x_gmi.shape

    Xg_list = []
    Xs_list = []
    Y_list = []
    RC_list = []

    for r in range(0, ny - patch_size + 1, stride):
        for c in range(0, nx - patch_size + 1, stride):
            xg = x_gmi[:, r:r + patch_size, c:c + patch_size]
            xs = x_static[:, r:r + patch_size, c:c + patch_size]
            yt = y[:, r:r + patch_size, c:c + patch_size]

            if not keep_patch(yt, min_valid_fraction=min_valid_fraction):
                continue

            Xg_list.append(xg)
            Xs_list.append(xs)
            Y_list.append(yt)
            RC_list.append((r, c))

    if not Xg_list:
        return (
            np.empty((0, x_gmi.shape[0], patch_size, patch_size), dtype=np.float32),
            np.empty((0, x_static.shape[0], patch_size, patch_size), dtype=np.float32),
            np.empty((0, y.shape[0], patch_size, patch_size), dtype=np.float32),
            np.empty((0, 2), dtype=np.int32),
        )

    return (
        np.stack(Xg_list).astype(np.float32),
        np.stack(Xs_list).astype(np.float32),
        np.stack(Y_list).astype(np.float32),
        np.asarray(RC_list, dtype=np.int32),
    )


def save_patch_file(
    out_path: Path,
    Xg: np.ndarray,
    Xs: np.ndarray,
    Y: np.ndarray,
    RC: np.ndarray,
    target_day: str,
    source_file: str,
    split: str,
) -> None:
    ensure_dir(out_path.parent)
    np.savez_compressed(
        out_path,
        X_gmi=Xg,
        X_static=Xs,
        Y=Y,
        patch_rc=RC,
        target_day=np.asarray(target_day),
        source_file=np.asarray(source_file),
        split=np.asarray(split),
    )


def save_if_nonempty(
    output_root: Path,
    split: str,
    target_day: pd.Timestamp,
    source_path: Path,
    Xg: np.ndarray,
    Xs: np.ndarray,
    Y: np.ndarray,
    RC: np.ndarray,
):
    if len(Xg) == 0:
        return None
    out_dir = output_root / split / f"{target_day.year}"
    stem = source_path.stem.replace("GMI_fullModelIO_", "patches_")
    out_path = out_dir / f"{stem}_{split}.npz"
    save_patch_file(out_path, Xg, Xs, Y, RC, str(target_day.date()), str(source_path), split)
    return out_path


def choose_validation_days(files: List[Path], val_day_fraction: float, seed: int) -> set:
    train_days = sorted({parse_day_from_filename(p) for p in files if parse_day_from_filename(p).year in TRAIN_YEARS_DEFAULT})
    if not train_days:
        return set()

    rng = np.random.default_rng(seed)
    idx = np.arange(len(train_days))
    rng.shuffle(idx)

    n_val_days = int(np.floor(len(train_days) * val_day_fraction))
    if len(train_days) > 1:
        n_val_days = min(max(n_val_days, 1), len(train_days) - 1)

    val_idx = idx[:n_val_days]
    return {train_days[i] for i in val_idx}


def run_pipeline(
    input_root: Path,
    output_root: Path,
    patch_size: int,
    stride: int,
    min_valid_fraction: float,
    val_day_fraction: float,
    seed: int,
    start_day: Optional[str],
    end_day: Optional[str],
) -> None:
    files = list_merged_files(input_root)
    print(f"Found {len(files)} merged full-global files")

    if start_day is not None:
        start_ts = pd.Timestamp(start_day).normalize()
        files = [p for p in files if parse_day_from_filename(p) >= start_ts]
    if end_day is not None:
        end_ts = pd.Timestamp(end_day).normalize()
        files = [p for p in files if parse_day_from_filename(p) <= end_ts]

    val_days = choose_validation_days(files, val_day_fraction=val_day_fraction, seed=seed)
    print(f"Validation days selected: {len(val_days)}")

    saved = 0
    skipped = 0

    for i, path in enumerate(files, start=1):
        target_day = parse_day_from_filename(path)
        print(f"[{i}/{len(files)}] {path.name}")

        try:
            x_gmi, x_static, y = open_merged_file(path)
        except Exception as e:
            print(f"  skip open error: {e}")
            skipped += 1
            continue

        Xg, Xs, Y, RC = extract_patches(
            x_gmi=x_gmi,
            x_static=x_static,
            y=y,
            patch_size=patch_size,
            stride=stride,
            min_valid_fraction=min_valid_fraction,
        )

        if len(Xg) == 0:
            print("  no patches kept")
            skipped += 1
            continue

        year = target_day.year
        if year in TEST_YEARS_DEFAULT:
            out = save_if_nonempty(output_root, "test", target_day, path, Xg, Xs, Y, RC)
            print(f"  saved TEST {len(Xg)} patches -> {out}")
            saved += 1
            continue

        if year in TRAIN_YEARS_DEFAULT:
            split = "val" if target_day in val_days else "train"
            out = save_if_nonempty(output_root, split, target_day, path, Xg, Xs, Y, RC)
            print(f"  saved {split.upper()} {len(Xg)} patches -> {out}")
            saved += 1
            continue

        print("  skipped: year not in configured split")
        skipped += 1

    print(f"\nDone. saved_days={saved}, skipped_days={skipped}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Create day-split train/val/test patch shards from CNN-preprocessed merged GMI files")
    p.add_argument("--input-root", type=Path,
                   default=Path("/lustre/home/mariyam/GMI_1C-R/model_io_daily_fullglobal_cnnprep"))
    p.add_argument("--output-root", type=Path,
                   default=Path("/lustre/home/mariyam/GMI_1C-R/patches_split_byday_cnnprep"))
    p.add_argument("--patch-size", type=int, default=PATCH_SIZE_DEFAULT)
    p.add_argument("--stride", type=int, default=STRIDE_DEFAULT)
    p.add_argument("--min-valid-fraction", type=float, default=MIN_VALID_FRACTION_DEFAULT)
    p.add_argument("--val-day-fraction", type=float, default=VAL_DAY_FRACTION_DEFAULT)
    p.add_argument("--seed", type=int, default=SEED_DEFAULT)
    p.add_argument("--start-day", type=str, default=None)
    p.add_argument("--end-day", type=str, default=None)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_pipeline(
        input_root=args.input_root,
        output_root=args.output_root,
        patch_size=args.patch_size,
        stride=args.stride,
        min_valid_fraction=args.min_valid_fraction,
        val_day_fraction=args.val_day_fraction,
        seed=args.seed,
        start_day=args.start_day,
        end_day=args.end_day,
    )
