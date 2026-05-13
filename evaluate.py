#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a trained U-Net on val or test patch splits.

Why this file is useful
-----------------------
- train.py is for fitting the model
- evaluate.py is for measuring performance after training
- usually you use:
    * val during development / model selection
    * test once at the end for final reporting

This script:
- loads a saved checkpoint
- rebuilds the model with the correct in/out channels
- loads the requested split from .npz patch shards
- computes average MSE, MAE, and RMSE over all batches

Example
-------
# evaluate on validation split
python evaluate.py \
  --data-root /lustre/home/mariyam/GMI_1C-R/patches_split_byday_cnnprep \
  --checkpoint checkpoints/best_unet.pt \
  --split val \
  --use-static

# evaluate on final 2020 test split
python evaluate.py \
  --data-root /lustre/home/mariyam/GMI_1C-R/patches_split_byday_cnnprep \
  --checkpoint checkpoints/best_unet.pt \
  --split test \
  --use-static
"""

from __future__ import annotations

import argparse
from pathlib import Path
import math
import random

import numpy as np
import torch
import torch.nn.functional as F

from dataset import GMIPatchDataset, make_dataloader
from unet import UNet


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()

    total_mse = 0.0
    total_mae = 0.0
    total_samples = 0

    for x, y in dataloader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(x)

        mse = F.mse_loss(pred, y, reduction="mean")
        mae = F.l1_loss(pred, y, reduction="mean")

        bs = x.size(0)
        total_mse += mse.item() * bs
        total_mae += mae.item() * bs
        total_samples += bs

    mean_mse = total_mse / total_samples
    mean_mae = total_mae / total_samples
    mean_rmse = math.sqrt(mean_mse)

    return {
        "mse": mean_mse,
        "mae": mean_mae,
        "rmse": mean_rmse,
        "num_samples": total_samples,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate trained U-Net on GMI patch split")
    p.add_argument("--data-root", type=Path,
                   default=Path("/lustre/home/mariyam/GMI_1C-R/patches_split_byday_cnnprep"))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--use-static", action="store_true")
    p.add_argument("--fill-target-nan", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p


def main():
    args = build_parser().parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ds = GMIPatchDataset(
        root=args.data_root,
        split=args.split,
        use_static=args.use_static,
        fill_target_nan=args.fill_target_nan,
        return_metadata=False,
    )
    loader = make_dataloader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    in_channels = ds.gmi_channels + (ds.static_channels if args.use_static else 0)
    out_channels = ds.target_channels

    ckpt = torch.load(args.checkpoint, map_location=device)

    base_filters = tuple(ckpt.get("base_filters", (8, 16, 32, 64)))

    model = UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_filters=base_filters,
        use_batchnorm=False,
        final_activation="identity",
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])

    metrics = evaluate(model, loader, device=device)

    print(f"Split      : {args.split}")
    print(f"Patches    : {len(ds):,}")
    print(f"In channels: {in_channels}")
    print(f"Out chans  : {out_channels}")
    print(f"MSE        : {metrics['mse']:.6f}")
    print(f"RMSE       : {metrics['rmse']:.6f}")
    print(f"MAE        : {metrics['mae']:.6f}")


if __name__ == "__main__":
    main()
