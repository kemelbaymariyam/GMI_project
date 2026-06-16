#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare old per-sample H5 loading vs direct contiguous batch-slice loading.

Run:
python benchmark_h5_batch_read.py \
  --h5 /lustre/home/mariyam/GMI_1C-R/fixed_train/train_fixed_360k.h5 \
  --batch-size 16 \
  --batches 200
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--batches", type=int, default=200)
    p.add_argument("--fill-target-nan", type=float, default=0.0)
    args = p.parse_args()

    with h5py.File(args.h5, "r") as f:
        x = f["X_gmi"]
        y = f["Y"]
        n = x.shape[0]
        total_batches = min(args.batches, (n + args.batch_size - 1) // args.batch_size)

        print("H5:", args.h5)
        print("X_gmi shape:", x.shape, "chunks:", x.chunks, "compression:", x.compression)
        print("Y shape:", y.shape, "chunks:", y.chunks, "compression:", y.compression)
        print("Batch size:", args.batch_size)
        print("Batches:", total_batches)

        # Direct batch slice: what fastseq uses.
        t0 = time.time()
        samples = 0
        for b in range(total_batches):
            s = b * args.batch_size
            e = min(s + args.batch_size, n)
            xb = np.asarray(x[s:e], dtype=np.float32)
            yb = np.asarray(y[s:e], dtype=np.float32)
            yb = np.nan_to_num(yb, nan=args.fill_target_nan, copy=False)
            samples += e - s
        dt = time.time() - t0
        print("\nDirect contiguous batch-slice:")
        print(f"  seconds: {dt:.3f}")
        print(f"  batches/min: {total_batches / dt * 60:.1f}")
        print(f"  samples/sec: {samples / dt:.1f}")

        # Per-sample inside batch: similar to normal Dataset/DataLoader.
        t0 = time.time()
        samples = 0
        for b in range(total_batches):
            s = b * args.batch_size
            e = min(s + args.batch_size, n)
            xs = []
            ys = []
            for i in range(s, e):
                xs.append(x[i])
                yy = np.asarray(y[i], dtype=np.float32)
                yy = np.nan_to_num(yy, nan=args.fill_target_nan, copy=False)
                ys.append(yy)
            xb = np.stack(xs, axis=0).astype(np.float32, copy=False)
            yb = np.stack(ys, axis=0).astype(np.float32, copy=False)
            samples += e - s
        dt = time.time() - t0
        print("\nPer-sample reads then stack:")
        print(f"  seconds: {dt:.3f}")
        print(f"  batches/min: {total_batches / dt * 60:.1f}")
        print(f"  samples/sec: {samples / dt:.1f}")


if __name__ == "__main__":
    main()
