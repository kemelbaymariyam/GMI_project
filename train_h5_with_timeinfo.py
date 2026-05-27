#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
import random

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time 

from dataset_h5 import GMIPatchH5Dataset, make_dataloader
from unet import UNet, count_parameters


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, dataloader, optimizer, loss_fn, device, epoch):
    model.train()
    total_loss = 0.0

    data_wait_total = 0.0
    h2d_total = 0.0
    forward_total = 0.0
    backward_total = 0.0
    optim_total = 0.0

    end_prev = time.perf_counter()

    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}", leave=False)

    for batch_idx, (x, y) in enumerate(pbar):
        t0 = time.perf_counter()
        data_wait = t0 - end_prev
        data_wait_total += data_wait

        t1 = time.perf_counter()
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        h2d_total += (t2 - t1)

        optimizer.zero_grad()

        t3 = time.perf_counter()
        pred = model(x)
        loss = loss_fn(pred, y)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t4 = time.perf_counter()
        forward_total += (t4 - t3)

        t5 = time.perf_counter()
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t6 = time.perf_counter()
        backward_total += (t6 - t5)

        t7 = time.perf_counter()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t8 = time.perf_counter()
        optim_total += (t8 - t7)

        total_loss += loss.item() * x.size(0)

        pbar.set_postfix(
            loss=f"{loss.item():.3f}",
            data=f"{data_wait:.2f}s",
            h2d=f"{(t2-t1):.2f}s",
            fwd=f"{(t4-t3):.2f}s",
            bwd=f"{(t6-t5):.2f}s",
        )

        end_prev = time.perf_counter()

        # only print detailed averages for first few batches
        if batch_idx == 9:
            n = batch_idx + 1
            print(f"\n[DEBUG after {n} batches]")
            print(f"avg data wait   : {data_wait_total / n:.3f} s")
            print(f"avg host->device: {h2d_total / n:.3f} s")
            print(f"avg forward     : {forward_total / n:.3f} s")
            print(f"avg backward    : {backward_total / n:.3f} s")
            print(f"avg optim step  : {optim_total / n:.3f} s")
            print("Stopping early for profiling check.")
            break

    return total_loss / ((batch_idx + 1) * dataloader.batch_size)


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


def build_parser():
    p = argparse.ArgumentParser(description="Train U-Net from HDF5 patch files")
    p.add_argument("--train-h5", type=Path, required=True)
    p.add_argument("--val-h5", type=Path, required=True)
    p.add_argument("--save-dir", type=Path, default=Path("checkpoints_h5"))
    p.add_argument("--use-static", action="store_true")
    p.add_argument("--fill-target-nan", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--early-stop-patience", type=int, default=12)
    return p


def main():
    args = build_parser().parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Train H5: {args.train_h5}")
    print(f"Val H5  : {args.val_h5}")

    print("Loading training dataset ...")
    train_ds = GMIPatchH5Dataset(args.train_h5, use_static=args.use_static, fill_target_nan=args.fill_target_nan)
    print("Loading validation dataset ...")
    val_ds = GMIPatchH5Dataset(args.val_h5, use_static=args.use_static, fill_target_nan=args.fill_target_nan)

    print("Building dataloaders ...")
    train_loader = make_dataloader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    val_loader = make_dataloader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"))

    in_channels = train_ds.gmi_channels + (train_ds.static_channels if args.use_static else 0)
    out_channels = train_ds.target_channels

    print("Building model ...")
    model = UNet(in_channels=in_channels, out_channels=out_channels, base_filters=(8, 16, 32, 64), use_batchnorm=False, final_activation="identity").to(device)

    print(f"Train patches: {len(train_ds):,}")
    print(f"Val patches  : {len(val_ds):,}")
    print(f"In channels  : {in_channels}")
    print(f"Out channels : {out_channels}")
    print(f"Patch size   : {train_ds.patch_size}")
    print(f"Parameters   : {count_parameters(model):,}")
    print(f"Model device : {next(model.parameters()).device}")

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)

    args.save_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nStarting epoch {epoch} ...")
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch)
        print("Validation starting ...")
        val_loss = validate(model, val_loader, loss_fn, device, epoch)
        scheduler.step(val_loss)

        print(f"Epoch [{epoch:03d}/{args.epochs}] Train Loss: {train_loss:.6f} Val Loss: {val_loss:.6f} LR: {optimizer.param_groups[0]['lr']:.2e}")

        torch.save({
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
            "train_h5": str(args.train_h5),
            "val_h5": str(args.val_h5),
        }, args.save_dir / "latest_unet.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            bad_epochs = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "in_channels": in_channels,
                "out_channels": out_channels,
                "use_static": args.use_static,
                "base_filters": (8, 16, 32, 64),
                "train_h5": str(args.train_h5),
                "val_h5": str(args.val_h5),
            }, args.save_dir / "best_unet.pt")
            print(f"  Saved new best model at epoch {epoch} (val={best_val_loss:.6f})")
        else:
            bad_epochs += 1

        if bad_epochs >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch}. Best epoch was {best_epoch}.")
            break

    print("Training complete.")
    print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
