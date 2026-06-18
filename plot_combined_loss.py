#!/usr/bin/env python3
"""
Combine multiple training history.csv files into one continuous loss curve.

Example:
python plot_combined_loss.py \
  --history \
    checkpoints_fastseq_bs64_val2_gpu2_uuid/history.csv \
    checkpoints_recover_epoch14_lr1e4_noamp_clip/history.csv \
    checkpoints_recover_lr3e5_from_epoch38/history.csv \
  --output combined_loss_curve.png \
  --combined-csv combined_history.csv

The script:
- concatenates all CSV files
- sorts by epoch
- removes duplicate epochs, keeping the last supplied record
- plots train and validation loss
- marks learning-rate changes
- reports the best validation epoch
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_COLUMNS = {"epoch", "train_loss", "val_loss", "lr"}


def load_history(path: Path, source_order: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"History file not found: {path}")

    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    df = df.copy()
    df["source_file"] = str(path)
    df["source_order"] = source_order

    for column in ["epoch", "train_loss", "val_loss", "lr"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Remove failed NaN epochs from the plotted curve.
    df = df.dropna(subset=["epoch", "train_loss", "val_loss", "lr"])
    df["epoch"] = df["epoch"].astype(int)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--history",
        type=Path,
        nargs="+",
        required=True,
        help="History CSV files in chronological/resume order.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("combined_loss_curve.png"),
    )
    parser.add_argument(
        "--combined-csv",
        type=Path,
        default=Path("combined_history.csv"),
    )
    parser.add_argument(
        "--title",
        default="U-Net training and validation loss",
    )
    parser.add_argument(
        "--log-y",
        action="store_true",
        help="Use logarithmic loss axis; useful when early losses are much larger.",
    )
    args = parser.parse_args()

    frames = [
        load_history(path, source_order=i)
        for i, path in enumerate(args.history)
    ]
    combined = pd.concat(frames, ignore_index=True)

    # Later files correspond to later resume stages.
    # For duplicate epochs, retain the record from the last supplied file.
    combined = (
        combined.sort_values(["epoch", "source_order"])
        .drop_duplicates(subset=["epoch"], keep="last")
        .sort_values("epoch")
        .reset_index(drop=True)
    )

    args.combined_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.combined_csv, index=False)

    best_idx = combined["val_loss"].idxmin()
    best = combined.loc[best_idx]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        combined["epoch"],
        combined["train_loss"],
        linewidth=2,
        label="Training loss",
    )
    ax.plot(
        combined["epoch"],
        combined["val_loss"],
        linewidth=2,
        label="Validation loss",
    )

    ax.scatter(
        [best["epoch"]],
        [best["val_loss"]],
        s=60,
        zorder=5,
        label=f"Best validation: {best['val_loss']:.3f} (epoch {int(best['epoch'])})",
    )

    # Identify the first epoch at every new learning rate.
    lr_changed = combined["lr"].ne(combined["lr"].shift())
    lr_points = combined.loc[lr_changed, ["epoch", "lr"]]

    for _, row in lr_points.iterrows():
        epoch = int(row["epoch"])
        lr = float(row["lr"])
        ax.axvline(epoch, linestyle="--", linewidth=1, alpha=0.55)
        ax.text(
            epoch,
            0.98,
            f"LR={lr:g}",
            transform=ax.get_xaxis_transform(),
            rotation=90,
            va="top",
            ha="right",
            fontsize=8,
        )

    ax.set_title(args.title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean squared error loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    ax.margins(x=0.01)

    if args.log_y:
        ax.set_yscale("log")

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot: {args.output}")
    print(f"Saved merged history: {args.combined_csv}")
    print(
        f"Best validation loss: {best['val_loss']:.6f} "
        f"at epoch {int(best['epoch'])}"
    )
    print(f"Epochs plotted: {combined['epoch'].min()}–{combined['epoch'].max()}")


if __name__ == "__main__":
    main()
