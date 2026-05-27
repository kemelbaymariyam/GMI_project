#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a manifest CSV for day-based GMI patch shard files.

For each .npz shard file, this script records:
- split
- year
- target_day (parsed from filename if possible)
- relative file path
- number of patches in the file

Output columns:
split,year,target_day,rel_path,n_patches
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Optional

import numpy as np


def parse_day_from_name(name: str) -> Optional[str]:
    m = re.search(r"(20\d{6})", name)
    if not m:
        return None
    s = m.group(1)
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"


def detect_split_and_year(data_root: Path, file_path: Path) -> tuple[str, str]:
    rel = file_path.relative_to(data_root)
    parts = rel.parts
    split = parts[0] if len(parts) >= 1 else ""
    year = parts[1] if len(parts) >= 2 else ""
    return split, year


def count_patches_in_file(path: Path) -> int:
    with np.load(path, allow_pickle=False) as data:
        if "X_gmi" not in data:
            raise KeyError(f"X_gmi not found in {path}")
        return int(data["X_gmi"].shape[0])


def build_manifest(data_root: Path, output_csv: Path) -> None:
    files = sorted(data_root.rglob("*.npz"))
    print(f"Found {len(files)} shard files under {data_root}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "year", "target_day", "rel_path", "n_patches"])

        total_patches = 0
        bad_files = 0

        for i, path in enumerate(files, start=1):
            rel = path.relative_to(data_root)
            split, year = detect_split_and_year(data_root, path)
            target_day = parse_day_from_name(path.name)

            try:
                n_patches = count_patches_in_file(path)
                total_patches += n_patches
                writer.writerow([split, year, target_day or "", str(rel), n_patches])
                print(f"[{i}/{len(files)}] OK   {rel}  patches={n_patches}")
            except Exception as e:
                bad_files += 1
                print(f"[{i}/{len(files)}] FAIL {rel}  error={e}")

    print("\nDone.")
    print(f"Manifest written to: {output_csv}")
    print(f"Total files scanned : {len(files)}")
    print(f"Bad files           : {bad_files}")
    print(f"Total patch count   : {total_patches}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build manifest CSV for GMI patch shard files")
    p.add_argument("--data-root", type=Path,
                   default=Path("/lustre/home/mariyam/GMI_1C-R/patches_split_byday_cnnprep"))
    p.add_argument("--output", type=Path,
                   default=Path("/lustre/home/mariyam/GMI_1C-R/patches_split_byday_cnnprep/manifest.csv"))
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    build_manifest(args.data_root, args.output)
