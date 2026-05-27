#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GMI daily gridded-with-gaps product (NO temporal Gaussian weighting).

Purpose
-------
This script creates the "raw daily gridded data with gaps" analog for GMI:
- use ONLY swaths from the target day
- map them to a regular lat/lon grid
- if multiple raw pixels from the SAME scan fall into the SAME grid cell,
  average them first (fair scan-cell averaging)
- then average scan-cell means across all scans/files of that day
- leave unsupported cells as NaN

Output
------
Saves only:
- S1_Tc
- S2_Tc

This is meant to complement the Gaussian-weighted script:
- this file creates the daily gridded data WITH gaps
- your Gaussian script creates the temporally weighted daily composite
"""

from __future__ import annotations

import argparse
import hashlib
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import xarray as xr


# ============================================================
# USER SETTINGS
# ============================================================
INPUT_ROOT = Path("/lustre/common/GPM/PMM/L1C-R.GPM.GMI.v08A")
OUTPUT_ROOT = Path("/lustre/home/mariyam/GMI_1C-R/gmi_daily_gridded_gaps")
CACHE_ROOT = Path("/lustre/home/mariyam/GMI_1C-R/prepared_cache_scanfair")

#YEARS_TO_USE = {2016, 2017}
#YEARS_TO_USE = {2018, 2019}
YEARS_TO_USE = {2020, 2021}

GRID_RES_DEG = 0.125

ZLIB = True
COMPLEVEL = 4


# ============================================================
# BASIC HELPERS
# ============================================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_datetime_from_filename(path: Path) -> pd.Timestamp:
    name = path.name
    m = re.search(r"\.(20\d{6})-S(\d{6})-E\d{6}\.", name)
    if not m:
        raise ValueError(f"Could not parse datetime from filename: {name}")
    return pd.to_datetime(m.group(1) + m.group(2), format="%Y%m%d%H%M%S")


def parse_day_from_filename(path: Path) -> pd.Timestamp:
    return parse_datetime_from_filename(path).normalize()


def find_all_files(root: Path) -> List[Path]:
    exts = {".HDF5", ".hdf5", ".H5", ".h5", ".nc", ".NC"}
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix in exts:
            try:
                dt = parse_datetime_from_filename(p)
                if dt.year in YEARS_TO_USE:
                    files.append(p)
            except Exception:
                continue
    return sorted(files)


def make_lat_lon_bins(res_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    lat_bins = np.arange(-90.0, 90.0 + res_deg, res_deg)
    lon_bins = np.arange(-180.0, 180.0 + res_deg, res_deg)
    return lat_bins, lon_bins


def make_lat_lon_centers(lat_bins: np.ndarray, lon_bins: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lat = 0.5 * (lat_bins[:-1] + lat_bins[1:])
    lon = 0.5 * (lon_bins[:-1] + lon_bins[1:])
    return lat, lon


def channel_names_for_swath(swath: str, nchan: int) -> List[str]:
    if swath == "S1" and nchan == 9:
        return ["10V", "10H", "19V", "19H", "23V", "37V", "37H", "89V", "89H"]
    if swath == "S2" and nchan == 4:
        return ["166V", "166H", "183pm3V", "183pm7V"]
    return [f"ch{c+1:02d}" for c in range(nchan)]


# ============================================================
# HDF5 READERS
# ============================================================
def find_dataset_in_group(group: h5py.Group, candidates: List[str]) -> Optional[np.ndarray]:
    for c in candidates:
        if c in group and isinstance(group[c], h5py.Dataset):
            return group[c][...]
    for _, obj in group.items():
        if isinstance(obj, h5py.Group):
            out = find_dataset_in_group(obj, candidates)
            if out is not None:
                return out
    return None


def open_tc_lat_lon_secondofday(filepath: Path) -> Dict[str, Dict[str, np.ndarray]]:
    out: Dict[str, Dict[str, np.ndarray]] = {}

    with h5py.File(filepath, "r") as f:
        for swath in ["S1", "S2"]:
            if swath not in f:
                continue

            g = f[swath]

            if "Tc" in g and isinstance(g["Tc"], h5py.Dataset):
                tc = g["Tc"][...]
            else:
                tc = find_dataset_in_group(g, ["Tc", "tc"])
            if tc is None:
                raise KeyError(f"Could not find {swath}/Tc in {filepath}")

            lat = find_dataset_in_group(g, ["Latitude"])
            lon = find_dataset_in_group(g, ["Longitude"])
            if lat is None or lon is None:
                raise KeyError(f"Could not find Latitude/Longitude for {swath} in {filepath}")

            if "ScanTime" not in g or not isinstance(g["ScanTime"], h5py.Group):
                raise KeyError(f"Could not find {swath}/ScanTime in {filepath}")
            if "SecondOfDay" not in g["ScanTime"]:
                raise KeyError(f"Could not find {swath}/ScanTime/SecondOfDay in {filepath}")
            sod = g["ScanTime"]["SecondOfDay"][...]

            out[swath] = {
                "Tc": np.asarray(tc),
                "lat": np.asarray(lat),
                "lon": np.asarray(lon),
                "second_of_day": np.asarray(sod),
            }

    return out


# ============================================================
# SHAPE STANDARDIZATION
# ============================================================
def standardize_tc_lat_lon(
    tc: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    second_of_day: np.ndarray,
    expected_nchan: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tc = np.asarray(tc)
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    second_of_day = np.asarray(second_of_day)

    if tc.ndim != 3:
        raise ValueError(f"Tc must be 3D, got {tc.shape}")
    if lat.ndim != 2 or lon.ndim != 2:
        raise ValueError(f"Latitude/Longitude must be 2D, got {lat.shape}, {lon.shape}")
    if second_of_day.ndim != 1:
        raise ValueError(f"SecondOfDay must be 1D, got {second_of_day.shape}")

    candidate_axes = list(range(3))
    if expected_nchan is not None:
        candidate_axes = [ax for ax in range(3) if tc.shape[ax] == expected_nchan] or candidate_axes

    for ch_axis in candidate_axes:
        tc_ch_first = np.moveaxis(tc, ch_axis, 0)
        d1, d2 = tc_ch_first.shape[1:]

        if lat.shape == (d1, d2) and lon.shape == (d1, d2) and len(second_of_day) == d2:
            return tc_ch_first, lat, lon, second_of_day

        if lat.T.shape == (d1, d2) and lon.T.shape == (d1, d2) and len(second_of_day) == d2:
            return tc_ch_first, lat.T, lon.T, second_of_day

        if lat.shape == (d2, d1) and lon.shape == (d2, d1) and len(second_of_day) == d1:
            return np.transpose(tc_ch_first, (0, 2, 1)), lat, lon, second_of_day

        if lat.T.shape == (d2, d1) and lon.T.shape == (d2, d1) and len(second_of_day) == d1:
            return np.transpose(tc_ch_first, (0, 2, 1)), lat.T, lon.T, second_of_day

    raise ValueError(
        "Could not standardize shapes: "
        f"Tc={tc.shape}, lat={lat.shape}, lon={lon.shape}, second_of_day={second_of_day.shape}"
    )


# ============================================================
# PREPARED CACHE
# ============================================================
def cache_key_for_file(filepath: Path, grid_res_deg: float) -> str:
    raw = f"{filepath.resolve()}|{filepath.stat().st_mtime_ns}|{grid_res_deg}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def cache_path_for_file(filepath: Path, cache_root: Path, grid_res_deg: float) -> Path:
    key = cache_key_for_file(filepath, grid_res_deg)
    return cache_root / f"{filepath.stem[:80]}_{key}.npz"


def prepare_one_swath(
    swath: str,
    tc: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    second_of_day: np.ndarray,
    lat_bins: np.ndarray,
    lon_bins: np.ndarray,
) -> Dict[str, np.ndarray]:
    expected_nchan = 9 if swath == "S1" else 4 if swath == "S2" else None

    tc, lat, lon, second_of_day = standardize_tc_lat_lon(
        tc=tc,
        lat=lat,
        lon=lon,
        second_of_day=second_of_day,
        expected_nchan=expected_nchan,
    )

    tc = np.asarray(tc, dtype=np.float32)
    lat = np.asarray(lat, dtype=np.float32)
    lon = np.asarray(lon, dtype=np.float32)
    second_of_day = np.asarray(second_of_day, dtype=np.float64)

    lon = ((lon + 180.0) % 360.0) - 180.0

    ny = len(lat_bins) - 1
    nx = len(lon_bins) - 1

    lat_idx = np.digitize(lat, lat_bins) - 1
    lon_idx = np.digitize(lon, lon_bins) - 1

    geo_valid = (
        np.isfinite(lat) &
        np.isfinite(lon) &
        (lat_idx >= 0) & (lat_idx < ny) &
        (lon_idx >= 0) & (lon_idx < nx)
    )

    return {
        "tc": tc,
        "lat_idx": lat_idx.astype(np.int32),
        "lon_idx": lon_idx.astype(np.int32),
        "geo_valid": geo_valid.astype(np.uint8),
        "second_of_day": second_of_day,
        "channel_names": np.asarray(channel_names_for_swath(swath, tc.shape[0]), dtype="U16"),
    }


def prepare_granule(filepath: Path, lat_bins: np.ndarray, lon_bins: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
    raw = open_tc_lat_lon_secondofday(filepath)
    out: Dict[str, Dict[str, np.ndarray]] = {}

    for swath in ["S1", "S2"]:
        if swath not in raw:
            continue

        out[swath] = prepare_one_swath(
            swath=swath,
            tc=raw[swath]["Tc"],
            lat=raw[swath]["lat"],
            lon=raw[swath]["lon"],
            second_of_day=raw[swath]["second_of_day"],
            lat_bins=lat_bins,
            lon_bins=lon_bins,
        )

    return out


def save_prepared_granule(cache_path: Path, prepared: Dict[str, Dict[str, np.ndarray]]) -> None:
    ensure_dir(cache_path.parent)
    payload: Dict[str, np.ndarray] = {}
    for swath, d in prepared.items():
        for k, v in d.items():
            payload[f"{swath}_{k}"] = v
    np.savez_compressed(cache_path, **payload)


def load_prepared_granule(cache_path: Path) -> Dict[str, Dict[str, np.ndarray]]:
    z = np.load(cache_path, allow_pickle=False)
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for swath in ["S1", "S2"]:
        keys = [k for k in z.files if k.startswith(swath + "_")]
        if not keys:
            continue
        out[swath] = {}
        for k in keys:
            out[swath][k[len(swath) + 1:]] = z[k]
    return out


def get_prepared_granule(
    filepath: Path,
    lat_bins: np.ndarray,
    lon_bins: np.ndarray,
    cache_root: Path,
    use_cache: bool = True,
    verbose: bool = False,
) -> Dict[str, Dict[str, np.ndarray]]:
    cache_path = cache_path_for_file(filepath, cache_root, float(lat_bins[1] - lat_bins[0]))

    if use_cache and cache_path.exists():
        if verbose:
            print("    cache hit")
        return load_prepared_granule(cache_path)

    if verbose:
        print("    cache miss -> preparing")

    prepared = prepare_granule(filepath, lat_bins, lon_bins)

    if use_cache:
        save_prepared_granule(cache_path, prepared)

    return prepared


# ============================================================
# DAILY (UNWEIGHTED) FAIR ACCUMULATION
# ============================================================
def allocate_accumulators(nchan: int, ny: int, nx: int) -> Dict[str, np.ndarray]:
    return {
        "sum": np.zeros((nchan, ny, nx), dtype=np.float64),
        "count": np.zeros((nchan, ny, nx), dtype=np.float64),
    }


def accumulate_prepared_swath_to_grid_daily(
    prepared_swath: Dict[str, np.ndarray],
    acc: Dict[str, np.ndarray],
) -> None:
    tc = prepared_swath["tc"]
    lat_idx = prepared_swath["lat_idx"]
    lon_idx = prepared_swath["lon_idx"]
    geo_valid = prepared_swath["geo_valid"].astype(bool)

    nchan, _, nscan = tc.shape
    nx = acc["sum"].shape[2]

    for s in range(nscan):
        li = lat_idx[:, s]
        lj = lon_idx[:, s]
        gv = geo_valid[:, s]
        flat_index = li.astype(np.int64) * np.int64(nx) + lj.astype(np.int64)

        for c in range(nchan):
            vals = tc[c, :, s]
            valid = gv & np.isfinite(vals) & (vals > 0.0) & (vals < 400.0)
            if not np.any(valid):
                continue

            fi = flat_index[valid]
            vv = vals[valid]

            unique_fi, inverse = np.unique(fi, return_inverse=True)

            local_sum = np.zeros(len(unique_fi), dtype=np.float64)
            local_count = np.zeros(len(unique_fi), dtype=np.int32)

            np.add.at(local_sum, inverse, vv.astype(np.float64))
            np.add.at(local_count, inverse, 1)

            local_mean = local_sum / local_count

            out_li = (unique_fi // nx).astype(np.int64)
            out_lj = (unique_fi % nx).astype(np.int64)

            np.add.at(acc["sum"][c], (out_li, out_lj), local_mean)
            np.add.at(acc["count"][c], (out_li, out_lj), 1.0)


# ============================================================
# DATASET BUILDING
# ============================================================
def build_dataset_from_acc(
    acc: Dict[str, np.ndarray],
    swath_name: str,
    channel_names: List[str],
    lat_centers: np.ndarray,
    lon_centers: np.ndarray,
) -> xr.Dataset:
    if swath_name == "S1":
        chdim = "channel_S1"
        prefix = "S1"
    elif swath_name == "S2":
        chdim = "channel_S2"
        prefix = "S2"
    else:
        raise ValueError(f"Unsupported swath_name: {swath_name}")

    mean = np.full_like(acc["sum"], np.nan, dtype=np.float32)
    np.divide(acc["sum"], acc["count"], out=mean, where=acc["count"] > 0)

    coords = {chdim: channel_names, "lat": lat_centers, "lon": lon_centers}
    ds = xr.Dataset({
        f"{prefix}_Tc": xr.DataArray(mean, dims=(chdim, "lat", "lon"), coords=coords),
    })
    return ds


def candidate_files_for_target_day(files: List[Path], target_day: pd.Timestamp) -> List[Path]:
    target_day = target_day.normalize()
    return [fp for fp in files if parse_day_from_filename(fp) == target_day]


def daily_grid_for_target_day(
    target_day: pd.Timestamp,
    candidate_files: List[Path],
    lat_bins: np.ndarray,
    lon_bins: np.ndarray,
    cache_root: Path,
    use_cache: bool = True,
) -> Optional[xr.Dataset]:
    ny = len(lat_bins) - 1
    nx = len(lon_bins) - 1
    lat_centers, lon_centers = make_lat_lon_centers(lat_bins, lon_bins)

    s1_acc = None
    s2_acc = None
    s1_channels = None
    s2_channels = None

    total_start = time.perf_counter()
    used_files = 0
    skipped_files = 0

    for i, fp in enumerate(candidate_files, start=1):
        file_start = time.perf_counter()
        print(f"[{i}/{len(candidate_files)}] {fp.name}")

        try:
            prep_start = time.perf_counter()
            prepared = get_prepared_granule(
                filepath=fp,
                lat_bins=lat_bins,
                lon_bins=lon_bins,
                cache_root=cache_root,
                use_cache=use_cache,
                verbose=True,
            )
            prep_dt = time.perf_counter() - prep_start
            print(f"    prepare/load time: {prep_dt:.2f} s")
        except Exception as e:
            skipped_files += 1
            print(f"    skipped: {e}")
            continue

        if "S1" in prepared:
            if s1_acc is None:
                nchan = prepared["S1"]["tc"].shape[0]
                s1_acc = allocate_accumulators(nchan=nchan, ny=ny, nx=nx)
                s1_channels = list(prepared["S1"]["channel_names"].astype(str))
            accumulate_prepared_swath_to_grid_daily(prepared["S1"], s1_acc)

        if "S2" in prepared:
            if s2_acc is None:
                nchan = prepared["S2"]["tc"].shape[0]
                s2_acc = allocate_accumulators(nchan=nchan, ny=ny, nx=nx)
                s2_channels = list(prepared["S2"]["channel_names"].astype(str))
            accumulate_prepared_swath_to_grid_daily(prepared["S2"], s2_acc)

        file_dt = time.perf_counter() - file_start
        print(f"    total file time: {file_dt:.2f} s")

        used_files += 1

        if i % 10 == 0:
            elapsed = time.perf_counter() - total_start
            avg = elapsed / i
            remain = avg * (len(candidate_files) - i)
            print(f"--- processed {i}/{len(candidate_files)} | elapsed={elapsed:.1f}s | avg/file={avg:.2f}s | ETA={remain:.1f}s ---")

    if s1_acc is None and s2_acc is None:
        return None

    parts: List[xr.Dataset] = []

    if s1_acc is not None:
        parts.append(build_dataset_from_acc(
            acc=s1_acc,
            swath_name="S1",
            channel_names=s1_channels,
            lat_centers=lat_centers,
            lon_centers=lon_centers,
        ))

    if s2_acc is not None:
        parts.append(build_dataset_from_acc(
            acc=s2_acc,
            swath_name="S2",
            channel_names=s2_channels,
            lat_centers=lat_centers,
            lon_centers=lon_centers,
        ))

    ds = xr.merge(parts)
    ds.attrs["source"] = "GMI daily gridded-with-gaps product"
    ds.attrs["target_day"] = str(target_day)
    ds.attrs["grid_res_deg"] = GRID_RES_DEG
    ds.attrs["used_files"] = used_files
    ds.attrs["skipped_files"] = skipped_files
    ds.attrs["method_note"] = "No temporal Gaussian weighting; daily equal-weight average of scan-cell means"
    ds.attrs["fairness_note"] = "multiple raw pixels from same scan/cell are averaged before daily accumulation"

    total_dt = time.perf_counter() - total_start
    print(f"Finished target day {target_day.date()} in {total_dt:.2f} s | used_files={used_files} | skipped_files={skipped_files}")

    return ds


def save_daily_grid(target_day: pd.Timestamp, ds: xr.Dataset, out_root: Path) -> Path:
    year_dir = out_root / f"{target_day.year}"
    ensure_dir(year_dir)
    out_path = year_dir / f"GMI_dailyGriddedWithGaps_{target_day:%Y%m%d}.nc"

    encoding = {v: {"zlib": ZLIB, "complevel": COMPLEVEL} for v in ds.data_vars}
    ds.to_netcdf(out_path, encoding=encoding)
    return out_path


# ============================================================
# DIAGNOSTIC
# ============================================================
def diagnostic_one_file(filepath: Path) -> None:
    raw = open_tc_lat_lon_secondofday(filepath)

    print(f"FILE: {filepath}")
    for swath in ["S1", "S2"]:
        if swath not in raw:
            continue

        tc = raw[swath]["Tc"]
        lat = raw[swath]["lat"]
        lon = raw[swath]["lon"]
        sod = raw[swath]["second_of_day"]

        print(f"\n[{swath}] raw shapes")
        print(f"  Tc:           {tc.shape}")
        print(f"  Latitude:     {lat.shape}")
        print(f"  Longitude:    {lon.shape}")
        print(f"  SecondOfDay:  {sod.shape}")

        expected_nchan = 9 if swath == "S1" else 4
        tc2, lat2, lon2, sod2 = standardize_tc_lat_lon(
            tc=tc,
            lat=lat,
            lon=lon,
            second_of_day=sod,
            expected_nchan=expected_nchan,
        )

        print(f"[{swath}] standardized shapes")
        print(f"  Tc:           {tc2.shape} [channel, pixel, scan]")
        print(f"  Latitude:     {lat2.shape} [pixel, scan]")
        print(f"  Longitude:    {lon2.shape} [pixel, scan]")
        print(f"  SecondOfDay:  {sod2.shape} [scan]")


# ============================================================
# PIPELINE
# ============================================================
def run_single_day(target_day: pd.Timestamp, use_cache: bool = True) -> Optional[Path]:
    ensure_dir(OUTPUT_ROOT)
    ensure_dir(CACHE_ROOT)

    lat_bins, lon_bins = make_lat_lon_bins(GRID_RES_DEG)
    all_files = find_all_files(INPUT_ROOT)
    print(f"Found {len(all_files)} candidate files")

    candidate_files = candidate_files_for_target_day(all_files, target_day)
    print(f"Using {len(candidate_files)} files for exact target day {target_day.date()}")

    ds = daily_grid_for_target_day(
        target_day=target_day,
        candidate_files=candidate_files,
        lat_bins=lat_bins,
        lon_bins=lon_bins,
        cache_root=CACHE_ROOT,
        use_cache=use_cache,
    )

    if ds is None:
        print("No valid output produced.")
        return None

    out_path = save_daily_grid(target_day, ds, OUTPUT_ROOT)
    print(f"Saved output to: {out_path}")
    return out_path


def run_pipeline(use_cache: bool = True) -> None:
    ensure_dir(OUTPUT_ROOT)
    ensure_dir(CACHE_ROOT)

    lat_bins, lon_bins = make_lat_lon_bins(GRID_RES_DEG)
    all_files = find_all_files(INPUT_ROOT)
    print(f"Found {len(all_files)} candidate files")

    if not all_files:
        print("No files found.")
        return

    all_days = sorted({parse_day_from_filename(fp) for fp in all_files})

    for target_day in all_days:
        print(f"[target day] {target_day.date()}")
        candidate_files = candidate_files_for_target_day(all_files, target_day)
        print(f"  using {len(candidate_files)} files from that exact day")

        ds = daily_grid_for_target_day(
            target_day=target_day,
            candidate_files=candidate_files,
            lat_bins=lat_bins,
            lon_bins=lon_bins,
            cache_root=CACHE_ROOT,
            use_cache=use_cache,
        )

        if ds is None:
            print("  no valid output")
            continue

        out_path = save_daily_grid(target_day, ds, OUTPUT_ROOT)
        print(f"  saved: {out_path}")

    print("Done.")


def prepare_cache_for_all_files() -> None:
    ensure_dir(CACHE_ROOT)
    lat_bins, lon_bins = make_lat_lon_bins(GRID_RES_DEG)

    files = find_all_files(INPUT_ROOT)
    print(f"Found {len(files)} candidate files")

    for i, fp in enumerate(files, start=1):
        try:
            _ = get_prepared_granule(
                filepath=fp,
                lat_bins=lat_bins,
                lon_bins=lon_bins,
                cache_root=CACHE_ROOT,
                use_cache=True,
            )
            print(f"[{i}/{len(files)}] cached {fp.name}")
        except Exception as e:
            print(f"[{i}/{len(files)}] failed {fp.name}: {e}")


# ============================================================
# CLI
# ============================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GMI daily gridded-with-gaps product")
    p.add_argument("--mode", choices=["single", "pipeline", "prepare", "diagnostic"], default="single")
    p.add_argument("--target-day", type=str, default="2016-01-15", help="YYYY-MM-DD for single mode")
    p.add_argument("--file", type=str, default="", help="file path for diagnostic mode")
    p.add_argument("--no-cache", action="store_true")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    use_cache = not args.no_cache

    if args.mode == "prepare":
        prepare_cache_for_all_files()
    elif args.mode == "pipeline":
        run_pipeline(use_cache=use_cache)
    elif args.mode == "diagnostic":
        if not args.file:
            raise SystemExit("For --mode diagnostic, provide --file /path/to/file")
        diagnostic_one_file(Path(args.file))
    else:
        run_single_day(
            target_day=pd.Timestamp(args.target_day),
            use_cache=use_cache,
        )
