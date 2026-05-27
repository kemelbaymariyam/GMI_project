#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create smaller subset manifests from a full manifest.csv by sampling WHOLE shard files
(not individual patches) until a target number of patches is reached.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import List, Dict


def read_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def write_manifest(rows: List[Dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["split", "year", "target_day", "rel_path", "n_patches"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})


def sample_rows_for_split(
    rows: List[Dict[str, str]],
    split: str,
    target_patches: int,
    seed: int,
    year_filter: str | None = None,
) -> List[Dict[str, str]]:
    subset = [r for r in rows if r["split"] == split]
    if year_filter is not None:
        keep_years = {y.strip() for y in year_filter.split(",") if y.strip()}
        subset = [r for r in subset if r["year"] in keep_years]

    if not subset:
        raise ValueError(f"No rows available for split={split}, year_filter={year_filter}")

    rng = random.Random(seed)
    rng.shuffle(subset)

    chosen: List[Dict[str, str]] = []
    total = 0
    for row in subset:
        chosen.append(row)
        total += int(row["n_patches"])
        if total >= target_patches:
            break

    print(f"{split}: chose {len(chosen)} shard files, total patches ~ {total:,}")
    return chosen


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build smaller subset manifests by sampling shard files")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--train-output", type=Path, required=True)
    p.add_argument("--val-output", type=Path, required=True)
    p.add_argument("--train-patches", type=int, default=300000)
    p.add_argument("--val-patches", type=int, default=50000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-years", type=str, default=None)
    p.add_argument("--val-years", type=str, default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    rows = read_manifest(args.manifest)

    train_rows = sample_rows_for_split(
        rows=rows,
        split="train",
        target_patches=args.train_patches,
        seed=args.seed,
        year_filter=args.train_years,
    )
    val_rows = sample_rows_for_split(
        rows=rows,
        split="val",
        target_patches=args.val_patches,
        seed=args.seed + 1,
        year_filter=args.val_years,
    )

    write_manifest(train_rows, args.train_output)
    write_manifest(val_rows, args.val_output)

    print(f"Wrote: {args.train_output}")
    print(f"Wrote: {args.val_output}")


if __name__ == "__main__":
    main()
