#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np


def read_manifest(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def inspect_rows(data_root: Path, rows):
    total = 0
    sample = None
    has_patch_rc = False

    for row in rows:
        p = data_root / row["rel_path"]
        with np.load(p, allow_pickle=False) as data:
            n = int(data["X_gmi"].shape[0])
            total += n
            if sample is None:
                sample = {
                    "X_gmi": tuple(data["X_gmi"].shape[1:]),
                    "X_static": tuple(data["X_static"].shape[1:]),
                    "Y": tuple(data["Y"].shape[1:]),
                }
            if "patch_rc" in data:
                has_patch_rc = True
    return total, sample, has_patch_rc


def convert_to_h5(data_root: Path, manifest_path: Path, output_h5: Path, compression: str | None):
    rows = read_manifest(manifest_path)
    total, sample, has_patch_rc = inspect_rows(data_root, rows)

    print(f"Manifest rows : {len(rows)}")
    print(f"Total patches : {total}")
    print(f"Writing       : {output_h5}")

    output_h5.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_h5, "w") as h5:
        xg = h5.create_dataset("X_gmi", shape=(total, *sample["X_gmi"]), dtype="float32", chunks=True, compression=compression)
        xs = h5.create_dataset("X_static", shape=(total, *sample["X_static"]), dtype="float32", chunks=True, compression=compression)
        y = h5.create_dataset("Y", shape=(total, *sample["Y"]), dtype="float32", chunks=True, compression=compression)
        pr = None
        if has_patch_rc:
            pr = h5.create_dataset("patch_rc", shape=(total, 2), dtype="int32", chunks=True, compression=compression)

        td = h5.create_dataset("target_day", shape=(total,), dtype=h5py.string_dtype(encoding="utf-8"))
        sf = h5.create_dataset("source_file", shape=(total,), dtype=h5py.string_dtype(encoding="utf-8"))

        write_pos = 0
        for i, row in enumerate(rows, start=1):
            p = data_root / row["rel_path"]
            print(f"[{i}/{len(rows)}] {row['rel_path']}")
            with np.load(p, allow_pickle=False) as data:
                xg_np = data["X_gmi"].astype(np.float32)
                xs_np = data["X_static"].astype(np.float32)
                y_np = data["Y"].astype(np.float32)
                n = xg_np.shape[0]

                xg[write_pos:write_pos+n] = xg_np
                xs[write_pos:write_pos+n] = xs_np
                y[write_pos:write_pos+n] = y_np
                if pr is not None and "patch_rc" in data:
                    pr[write_pos:write_pos+n] = data["patch_rc"].astype(np.int32)

                td[write_pos:write_pos+n] = [row.get("target_day", "")] * n
                sf[write_pos:write_pos+n] = [row["rel_path"]] * n
                write_pos += n

        h5.attrs["num_patches"] = int(total)
        h5.attrs["source_manifest"] = str(manifest_path)

    print("Done.")


def build_parser():
    p = argparse.ArgumentParser(description="Convert subset-manifest NPZ shards into one HDF5 file")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output-h5", type=Path, required=True)
    p.add_argument("--compression", default=None, choices=[None, "gzip", "lzf"])
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    convert_to_h5(args.data_root, args.manifest, args.output_h5, args.compression)
