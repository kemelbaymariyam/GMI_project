#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chunked/resumable training for GMI .npz patch shards.

Main features:
- resume from checkpoint
- optionally advance subset_id each epoch
- fixed validation subset
- save latest / best checkpoints
- optional per-epoch checkpoints
- write training history to CSV + JSONL
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Optional

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


def train_one_epoch(model, dataloader, optimizer, loss_fn, device, epoch: int) -> float:
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
def validate(model, dataloader, loss_fn, device, epoch: int) -> float:
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


def append_history_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()

    fieldnames = [
        "epoch",
        "train_subset_id",
        "train_loss",
        "val_loss",
        "lr",
        "best_val_loss_so_far",
        "is_best",
        "resume_path",
    ]

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def append_history_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Chunked/resumable training for GMI .npz patch shards")

    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("/lustre/home/mariyam/GMI_1C-R/patches_split_byday_cnnprep"),
    )
    p.add_argument("--save-dir", type=Path, default=Path("checkpoints_chunked"))

    p.add_argument("--use-static", action="store_true", help="Use X_static together with X_gmi")
    p.add_argument("--fill-target-nan", type=float, default=0.0)

    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-val-samples", type=int, default=None)

    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=10, help="Final epoch number to train up to")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--early-stop-patience", type=int, default=12)

    # train subset controls
    p.add_argument("--subset-seed", type=int, default=42)
    p.add_argument("--subset-id", type=int, default=0, help="Used if not advancing each epoch")
    p.add_argument(
        "--start-subset-id",
        type=int,
        default=None,
        help="Subset id to start from when advancing each epoch; defaults to --subset-id",
    )
    p.add_argument(
        "--advance-subset-each-epoch",
        action="store_true",
        help="Use a new subset_id each epoch",
    )
    p.add_argument(
        "--no-random-subset",
        action="store_true",
        help="Use contiguous chunks instead of random shuffled subset chunks",
    )

    # validation subset controls
    p.add_argument("--val-subset-seed", type=int, default=999)
    p.add_argument("--val-subset-id", type=int, default=0)
    p.add_argument("--val-random-subset", action="store_true", default=True)

    # resume / saving
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument(
        "--save-every-epoch",
        action="store_true",
        help="Also save one checkpoint file per epoch",
    )

    return p


def main():
    args = build_parser().parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    args.save_dir.mkdir(parents=True, exist_ok=True)
    history_csv = args.save_dir / "history.csv"
    history_jsonl = args.save_dir / "history.jsonl"
    per_epoch_dir = args.save_dir / "per_epoch"

    # fixed validation dataset/loader
    print("Loading validation dataset ...")
    val_ds = GMIPatchDataset(
        root=args.data_root,
        split="val",
        use_static=args.use_static,
        fill_target_nan=args.fill_target_nan,
        return_metadata=False,
        max_samples=args.max_val_samples,
        subset_seed=args.val_subset_seed,
        subset_id=args.val_subset_id,
        random_subset=args.val_random_subset,
    )
    print("Building validation dataloader ...")
    val_loader = make_dataloader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # temporary train dataset just to infer channels/shapes
    initial_subset_id = args.start_subset_id if args.start_subset_id is not None else args.subset_id
    print(f"Loading initial training dataset (subset_id={initial_subset_id}) ...")
    train_ds_for_shape = GMIPatchDataset(
        root=args.data_root,
        split="train",
        use_static=args.use_static,
        fill_target_nan=args.fill_target_nan,
        return_metadata=False,
        max_samples=args.max_train_samples,
        subset_seed=args.subset_seed,
        subset_id=initial_subset_id,
        random_subset=not args.no_random_subset,
    )

    in_channels = train_ds_for_shape.gmi_channels + (
        train_ds_for_shape.static_channels if args.use_static else 0
    )
    out_channels = train_ds_for_shape.target_channels

    print("Building model ...")
    model = UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_filters=(8, 16, 32, 64),
        use_batchnorm=False,
        final_activation="identity",
    ).to(device)

    print(f"Val patches  : {len(val_ds):,}")
    print(f"In channels  : {in_channels}")
    print(f"Out channels : {out_channels}")
    print(f"Patch size   : {train_ds_for_shape.patch_size}")
    print(f"Parameters   : {count_parameters(model):,}")

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=8,
    )

    # resume
    start_epoch = 1
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    resume_path_str: Optional[str] = None

    if args.resume is not None:
        print(f"Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        best_epoch = int(ckpt.get("epoch", 0))
        resume_path_str = str(args.resume)

        # optional: restore scheduler to current lr if optimizer resumed fine
        print(f"Resume start epoch: {start_epoch}")
        print(f"Best val so far   : {best_val_loss:.6f}")

    if start_epoch > args.epochs:
        print(f"Nothing to do: start_epoch={start_epoch} > epochs={args.epochs}")
        return

    for epoch in range(start_epoch, args.epochs + 1):
        if args.advance_subset_each_epoch:
            base_subset_id = args.start_subset_id if args.start_subset_id is not None else args.subset_id
            current_subset_id = base_subset_id + (epoch - start_epoch)
        else:
            current_subset_id = args.subset_id

        print(f"\nStarting epoch {epoch} ...")
        print(f"Using training subset_id = {current_subset_id}")

        print("Loading training dataset ...")
        train_ds = GMIPatchDataset(
            root=args.data_root,
            split="train",
            use_static=args.use_static,
            fill_target_nan=args.fill_target_nan,
            return_metadata=False,
            max_samples=args.max_train_samples,
            subset_seed=args.subset_seed,
            subset_id=current_subset_id,
            random_subset=not args.no_random_subset,
        )

        print("Building training dataloader ...")
        train_loader = make_dataloader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=False,   # keep ordered reading for speed
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

        print(f"Train patches: {len(train_ds):,}")

        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch)
        print("Validation starting ...")
        val_loss = validate(model, val_loader, loss_fn, device, epoch)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch:03d}/{args.epochs}] "
            f"subset_id={current_subset_id} "
            f"Train Loss: {train_loss:.6f} "
            f"Val Loss: {val_loss:.6f} "
            f"LR: {current_lr:.2e}"
        )

        latest_ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "use_static": args.use_static,
            "base_filters": (8, 16, 32, 64),
            "subset_seed": args.subset_seed,
            "subset_id": current_subset_id,
            "random_subset": not args.no_random_subset,
            "max_train_samples": args.max_train_samples,
            "max_val_samples": args.max_val_samples,
            "resume_path": resume_path_str,
        }
        save_checkpoint(args.save_dir / "latest_unet.pt", latest_ckpt)

        is_best = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            is_best = True

            best_ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "in_channels": in_channels,
                "out_channels": out_channels,
                "use_static": args.use_static,
                "base_filters": (8, 16, 32, 64),
                "subset_seed": args.subset_seed,
                "subset_id": current_subset_id,
                "random_subset": not args.no_random_subset,
                "max_train_samples": args.max_train_samples,
                "max_val_samples": args.max_val_samples,
                "resume_path": resume_path_str,
            }
            save_checkpoint(args.save_dir / "best_unet.pt", best_ckpt)
            print(f"  Saved new best model at epoch {epoch} (val={best_val_loss:.6f})")
        else:
            epochs_without_improvement += 1

        if args.save_every_epoch:
            per_epoch_ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "subset_seed": args.subset_seed,
                "subset_id": current_subset_id,
                "random_subset": not args.no_random_subset,
                "use_static": args.use_static,
                "in_channels": in_channels,
                "out_channels": out_channels,
                "base_filters": (8, 16, 32, 64),
            }
            save_checkpoint(per_epoch_dir / f"epoch_{epoch:03d}_subset_{current_subset_id:04d}.pt", per_epoch_ckpt)

        history_row = {
            "epoch": epoch,
            "train_subset_id": current_subset_id,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": current_lr,
            "best_val_loss_so_far": best_val_loss,
            "is_best": int(is_best),
            "resume_path": resume_path_str or "",
        }
        append_history_csv(history_csv, history_row)
        append_history_jsonl(history_jsonl, history_row)

        if epochs_without_improvement >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch}. Best epoch was {best_epoch}.")
            break

    print("Training complete.")
    print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
    print(f"History CSV  : {history_csv}")
    print(f"History JSONL: {history_jsonl}")


if __name__ == "__main__":
    main()