#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train U-Net directly from day-based .npz patch shards.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dataset_new import GMIPatchDataset, make_dataloader
from unet import UNet, count_parameters


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, dataloader, optimizer, loss_fn, device, epoch):
    model.train()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}", leave=False)

    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        pbar.set_postfix(loss=f"{loss.item():.6f}")

    return total_loss / len(dataloader.dataset)

@torch.no_grad()
def validate(model, dataloader, loss_fn, device, epoch):
    model.eval()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Val Epoch {epoch}", leave=False)

    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(x)
        loss = loss_fn(pred, y)
        total_loss += loss.item() * x.size(0)

        pbar.set_postfix(loss=f"{loss.item():.6f}")

    return total_loss / len(dataloader.dataset)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train paper-inspired U-Net from GMI .npz patch shards")
    p.add_argument("--data-root", type=Path,
                   default=Path("/lustre/home/mariyam/GMI_1C-R/patches_split_byday_cnnprep"))
    p.add_argument("--save-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--use-static", action="store_true",
                   help="Use X_static together with X_gmi")
    p.add_argument("--fill-target-nan", type=float, default=0.0)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-val-samples", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--early-stop-patience", type=int, default=12)
    p.add_argument("--subset-seed", type=int, default=42)
    p.add_argument("--subset-id", type=int, default=0,
               help="Which subset chunk to use")
    p.add_argument("--no-random-subset", action="store_true",
                help="Use first max_samples patches instead of random subset")
    return p


def main():
    args = build_parser().parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading training dataset ...")
    train_ds = GMIPatchDataset(
        root=args.data_root,
        split="train",
        use_static=args.use_static,
        fill_target_nan=args.fill_target_nan,
        return_metadata=False,
        max_samples=args.max_train_samples,
        subset_seed=args.subset_seed,
        subset_id=args.subset_id,
        random_subset=not args.no_random_subset,
    )
    print("First 20 train index entries:")
    print(train_ds.index[:20])

    print("Loading validation dataset ...")
    val_ds = GMIPatchDataset(
        root=args.data_root,
        split="val",
        use_static=args.use_static,
        fill_target_nan=args.fill_target_nan,
        return_metadata=False,
        max_samples=args.max_val_samples,
        subset_seed=999,
        subset_id=0,
        random_subset=True,
    )

    print("Building dataloaders ...")
    train_loader = make_dataloader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = make_dataloader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    in_channels = train_ds.gmi_channels + (train_ds.static_channels if args.use_static else 0)
    out_channels = train_ds.target_channels
    print("Building model ...")
    model = UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_filters=(8, 16, 32, 64),
        use_batchnorm=False,
        final_activation="identity",
    ).to(device)

    print(f"Train patches: {len(train_ds):,}")
    print(f"Val patches  : {len(val_ds):,}")
    print(f"In channels  : {in_channels}")
    print(f"Out channels : {out_channels}")
    print(f"Patch size   : {train_ds.patch_size}")
    print(f"Parameters   : {count_parameters(model):,}")

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=8,
    )

    args.save_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nStarting epoch {epoch} ...")
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch)
        print("Validation starting ...")
        val_loss = validate(model, val_loader, loss_fn, device, epoch)

        scheduler.step(val_loss)

        print(
            f"Epoch [{epoch:03d}/{args.epochs}] "
            f"Train Loss: {train_loss:.6f} "
            f"Val Loss: {val_loss:.6f} "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        latest_ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "use_static": args.use_static,
            "base_filters": (8, 16, 32, 64),
        }
        torch.save(latest_ckpt, args.save_dir / "latest_unet.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            best_ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "in_channels": in_channels,
                "out_channels": out_channels,
                "use_static": args.use_static,
                "base_filters": (8, 16, 32, 64),
            }
            torch.save(best_ckpt, args.save_dir / "best_unet.pt")
            print(f"  Saved new best model at epoch {epoch} (val={best_val_loss:.6f})")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch}. Best epoch was {best_epoch}.")
            break

    print("Training complete.")
    print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
