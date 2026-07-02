#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live-ish monitor for training history.csv from train_fixed_h5.py.

Usage in another tmux pane:
  watch -n 30 'python -m scripts.monitor_training --history /path/to/history.csv --tail 10'

Or just:
  python -m scripts.monitor_training --history /path/to/history.csv --plot
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--tail", type=int, default=10)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if not args.history.exists():
        raise FileNotFoundError(f"History file not found yet: {args.history}")

    df = pd.read_csv(args.history)
    if df.empty:
        print("history.csv exists but has no rows yet.")
        return

    cols = ["epoch", "train_loss", "val_loss", "lr", "best_val_loss_so_far", "best_epoch", "seconds_epoch"]
    cols = [c for c in cols if c in df.columns]

    print("\nLatest rows:")
    print(df[cols].tail(args.tail).to_string(index=False))

    best_i = df["val_loss"].idxmin()
    best = df.loc[best_i]
    print(
        f"\nBest so far: epoch={int(best['epoch'])}, "
        f"val_loss={best['val_loss']:.6f}, train_loss={best['train_loss']:.6f}"
    )

    if args.plot:
        import matplotlib.pyplot as plt

        out = args.out or (args.history.parent / "loss_curve.png")
        plt.figure()
        plt.plot(df["epoch"], df["train_loss"], label="train_loss")
        plt.plot(df["epoch"], df["val_loss"], label="val_loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        print(f"\nSaved plot to: {out}")


if __name__ == "__main__":
    main()
