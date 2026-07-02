#!/usr/bin/env python3
"""
Create a presentation-ready combined loss curve from resumed training runs.

Produces:
  1) full_training_loss.png   - full trajectory, log scale
  2) finetuning_loss.png      - zoomed linear view of later epochs

Example:
python plot_loss_for_report.py \
  --history \
    checkpoints_fastseq_bs64_val2_gpu2_uuid/history.csv \
    checkpoints_recover_epoch14_lr1e4_noamp_clip/history.csv \
    checkpoints_recover_lr3e5_from_epoch38/history.csv \
  --out-dir report_plots \
  --zoom-start 14
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_and_merge(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for order, path in enumerate(paths):
        df = pd.read_csv(path)
        df = df.copy()
        df["source_order"] = order
        for col in ["epoch", "train_loss", "val_loss", "lr"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["epoch", "train_loss", "val_loss", "lr"])
        df["epoch"] = df["epoch"].astype(int)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = (
        combined.sort_values(["epoch", "source_order"])
        .drop_duplicates("epoch", keep="last")
        .sort_values("epoch")
        .reset_index(drop=True)
    )
    return combined


def add_lr_markers(ax, df: pd.DataFrame, min_epoch: int | None = None):
    lr_changed = df["lr"].ne(df["lr"].shift())
    pts = df.loc[lr_changed, ["epoch", "lr"]]

    for _, row in pts.iterrows():
        epoch = int(row["epoch"])
        if min_epoch is not None and epoch < min_epoch:
            continue
        lr = float(row["lr"])
        ax.axvline(epoch, linestyle="--", linewidth=1, alpha=0.5)
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


def make_plot(
    df: pd.DataFrame,
    output: Path,
    title: str,
    log_y: bool,
    min_epoch: int | None = None,
):
    plot_df = df if min_epoch is None else df[df["epoch"] >= min_epoch].copy()

    best_idx = plot_df["val_loss"].idxmin()
    best = plot_df.loc[best_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(plot_df["epoch"], plot_df["train_loss"], linewidth=2, label="Training loss")
    ax.plot(plot_df["epoch"], plot_df["val_loss"], linewidth=2, label="Validation loss")

    ax.scatter(
        [best["epoch"]],
        [best["val_loss"]],
        s=65,
        zorder=5,
        label=f"Best validation: {best['val_loss']:.3f} (epoch {int(best['epoch'])})",
    )

    add_lr_markers(ax, df, min_epoch=min_epoch)

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean squared error")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    if log_y:
        ax.set_yscale("log")

    if min_epoch is not None:
        ymin = min(plot_df["train_loss"].min(), plot_df["val_loss"].min())
        ymax = max(plot_df["train_loss"].max(), plot_df["val_loss"].max())
        padding = max((ymax - ymin) * 0.08, 0.5)
        ax.set_ylim(ymin - padding, ymax + padding)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", nargs="+", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("report_plots"))
    parser.add_argument("--zoom-start", type=int, default=14)
    args = parser.parse_args()

    df = load_and_merge(args.history)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "combined_history.csv", index=False)

    make_plot(
        df,
        args.out_dir / "full_training_loss.png",
        "U-Net training history",
        log_y=True,
        min_epoch=None,
    )

    make_plot(
        df,
        args.out_dir / "finetuning_loss.png",
        f"U-Net fine-tuning loss from epoch {args.zoom_start}",
        log_y=False,
        min_epoch=args.zoom_start,
    )

    best = df.loc[df["val_loss"].idxmin()]
    print("Saved:", args.out_dir / "full_training_loss.png")
    print("Saved:", args.out_dir / "finetuning_loss.png")
    print("Saved:", args.out_dir / "combined_history.csv")
    print(
        f"Best validation loss: {best['val_loss']:.6f} "
        f"at epoch {int(best['epoch'])}"
    )


if __name__ == "__main__":
    main()
