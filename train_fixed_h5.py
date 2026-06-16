#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train U-Net using a FIXED training H5 file and a FIXED validation H5 file.

Designed for your GMI_project structure:
  - dataset_h5.py provides GMIPatchH5Dataset
  - unet.py provides UNet
  - build_fixed_train_h5_filtered.py already created the fixed train H5

Main outputs in --save-dir:
  - latest_unet.pt
  - best_unet.pt
  - history.csv
  - history.jsonl
  - train.log

Example:
python train_fixed_h5.py \
  --train-h5 /lustre/home/mariyam/GMI_1C-R/h5_subsets/fixed_train_filtered.h5 \
  --val-h5   /lustre/home/mariyam/GMI_1C-R/h5_subsets/fixed_val.h5 \
  --save-dir checkpoints_fixed_h5 \
  --use-static \
  --batch-size 16 \
  --epochs 100 \
  --lr 1e-3 \
  --num-workers 4 \
  --amp
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataset_h5 import GMIPatchH5Dataset, make_dataloader
from unet import UNet, count_parameters


# ----------------------------
# Reproducibility
# ----------------------------
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # This improves reproducibility, but can make training slower.
    # For maximum speed, you may set --deterministic off by not using it.
    torch.backends.cudnn.benchmark = True


# ----------------------------
# Small logger: prints to terminal AND saves to train.log
# ----------------------------
class TeeLogger:
    def __init__(self, log_path: Path):
        self.terminal = sys.stdout
        self.log = log_path.open("a", encoding="utf-8", buffering=1)

    def write(self, message: str):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


# ----------------------------
# H5 sanity check
# ----------------------------
def inspect_h5(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"H5 file not found: {path}")

    info = {"path": str(path)}
    with h5py.File(path, "r") as f:
        required = ["X_gmi", "Y"]
        for key in required:
            if key not in f:
                raise KeyError(f"{key} not found in {path}")

        info["n"] = int(f["X_gmi"].shape[0])
        info["x_gmi_shape"] = tuple(f["X_gmi"].shape)
        info["y_shape"] = tuple(f["Y"].shape)
        info["has_static"] = "X_static" in f
        info["x_static_shape"] = tuple(f["X_static"].shape) if "X_static" in f else None
        info["attrs"] = {k: str(v) for k, v in f.attrs.items()}

        if info["n"] != int(f["Y"].shape[0]):
            raise ValueError(f"X_gmi and Y have different sample counts in {path}")

        if "X_static" in f and info["n"] != int(f["X_static"].shape[0]):
            raise ValueError(f"X_gmi and X_static have different sample counts in {path}")

    return info


def assert_compatible(train_info: dict, val_info: dict, use_static: bool) -> None:
    # Channel and patch-size compatibility
    train_x = train_info["x_gmi_shape"]
    val_x = val_info["x_gmi_shape"]
    train_y = train_info["y_shape"]
    val_y = val_info["y_shape"]

    if train_x[1:] != val_x[1:]:
        raise ValueError(f"Train/val X_gmi shapes differ after N: {train_x[1:]} vs {val_x[1:]}")
    if train_y[1:] != val_y[1:]:
        raise ValueError(f"Train/val Y shapes differ after N: {train_y[1:]} vs {val_y[1:]}")

    if use_static:
        if not train_info["has_static"]:
            raise ValueError("You used --use-static but train H5 has no X_static")
        if not val_info["has_static"]:
            raise ValueError("You used --use-static but val H5 has no X_static")
        if train_info["x_static_shape"][1:] != val_info["x_static_shape"][1:]:
            raise ValueError(
                "Train/val X_static shapes differ after N: "
                f"{train_info['x_static_shape'][1:]} vs {val_info['x_static_shape'][1:]}"
            )


# ----------------------------
# Loss
# ----------------------------
def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    MSE only over finite target pixels.
    This is useful if your Y contains NaNs and you do NOT want to replace them by 0.

    Use with:
      --loss masked_mse --fill-target-nan none
    """
    mask = torch.isfinite(target)
    if not mask.any():
        # Avoid crashing on a completely invalid batch.
        # This returns zero but keeps graph connected.
        return pred.sum() * 0.0
    diff = pred[mask] - target[mask]
    return torch.mean(diff * diff)


def get_loss_fn(name: str):
    if name == "mse":
        return nn.MSELoss()
    if name == "l1":
        return nn.L1Loss()
    if name == "smooth_l1":
        return nn.SmoothL1Loss()
    if name == "masked_mse":
        return masked_mse_loss
    raise ValueError(f"Unknown loss: {name}")


# ----------------------------
# Training and validation
# ----------------------------
def train_one_epoch(
    model,
    dataloader,
    optimizer,
    loss_fn,
    device,
    epoch: int,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    grad_clip: Optional[float] = None,
    log_every: int = 100,
) -> float:
    model.train()
    total_loss = 0.0
    total_seen = 0

    pbar = tqdm(dataloader, desc=f"Train {epoch}", leave=False, dynamic_ncols=True)

    for step, (x, y) in enumerate(pbar, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                pred = model(x)
                loss = loss_fn(pred, y)

            scaler.scale(loss).backward()

            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

        bs = x.size(0)
        total_loss += float(loss.item()) * bs
        total_seen += bs

        running = total_loss / max(total_seen, 1)
        pbar.set_postfix(batch_loss=f"{loss.item():.6f}", avg=f"{running:.6f}")

        if step % log_every == 0:
            print(
                f"[epoch {epoch:03d} step {step:06d}/{len(dataloader):06d}] "
                f"batch_loss={loss.item():.6f} running_train_loss={running:.6f}",
                flush=True,
            )

    return total_loss / max(total_seen, 1)


@torch.no_grad()
def validate(model, dataloader, loss_fn, device, epoch: int) -> float:
    model.eval()
    total_loss = 0.0
    total_seen = 0

    pbar = tqdm(dataloader, desc=f"Val {epoch}", leave=False, dynamic_ncols=True)

    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(x)
        loss = loss_fn(pred, y)

        bs = x.size(0)
        total_loss += float(loss.item()) * bs
        total_seen += bs

        pbar.set_postfix(loss=f"{loss.item():.6f}")

    return total_loss / max(total_seen, 1)


# ----------------------------
# History and checkpoints
# ----------------------------
def append_history_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "epoch",
        "train_loss",
        "val_loss",
        "lr",
        "best_val_loss_so_far",
        "best_epoch",
        "is_best",
        "seconds_epoch",
    ]
    file_exists = path.exists()
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
    p = argparse.ArgumentParser(description="Train U-Net from fixed train H5 and fixed validation H5")

    p.add_argument("--train-h5", type=Path, required=True)
    p.add_argument("--val-h5", type=Path, required=True)
    p.add_argument("--save-dir", type=Path, required=True)

    p.add_argument("--use-static", action="store_true")
    p.add_argument(
        "--fill-target-nan",
        type=str,
        default="0.0",
        help="Value used to replace NaN in Y. Use 'none' with --loss masked_mse to keep NaNs masked.",
    )

    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--loss", type=str, default="mse", choices=["mse", "l1", "smooth_l1", "masked_mse"])
    p.add_argument("--early-stop-patience", type=int, default=15)
    p.add_argument("--scheduler-patience", type=int, default=8)
    p.add_argument("--scheduler-factor", type=float, default=0.5)
    p.add_argument("--min-lr", type=float, default=1e-6)

    p.add_argument("--base-filters", type=int, nargs=4, default=[8, 16, 32, 64])
    p.add_argument("--use-batchnorm", action="store_true")
    p.add_argument("--final-activation", type=str, default="identity", choices=["identity", "sigmoid", "tanh"])

    p.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA")
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--save-every-epoch", action="store_true")
    p.add_argument("--log-every", type=int, default=100)

    return p


def main():
    args = build_parser().parse_args()
    seed_everything(args.seed)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.save_dir / "train.log"

    tee = TeeLogger(log_path)
    old_stdout = sys.stdout
    sys.stdout = tee

    try:
        print("=" * 80)
        print(f"Started: {datetime.now().isoformat(timespec='seconds')}")
        print(f"Command args: {vars(args)}")
        print("=" * 80)

        if args.fill_target_nan.lower() == "none":
            fill_target_nan = None
        else:
            fill_target_nan = float(args.fill_target_nan)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        if device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA memory allocated at start: {torch.cuda.memory_allocated() / 1024**3:.3f} GB")

        print("\nInspecting H5 files ...")
        train_info = inspect_h5(args.train_h5)
        val_info = inspect_h5(args.val_h5)
        assert_compatible(train_info, val_info, args.use_static)

        print(f"Train H5: {train_info}")
        print(f"Val H5  : {val_info}")

        print("\nLoading datasets ...")
        train_ds = GMIPatchH5Dataset(
            h5_path=args.train_h5,
            use_static=args.use_static,
            fill_target_nan=fill_target_nan,
            return_metadata=False,
        )
        val_ds = GMIPatchH5Dataset(
            h5_path=args.val_h5,
            use_static=args.use_static,
            fill_target_nan=fill_target_nan,
            return_metadata=False,
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

        print("\nBuilding model ...")
        model = UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_filters=tuple(args.base_filters),
            use_batchnorm=args.use_batchnorm,
            final_activation=args.final_activation,
        ).to(device)

        print(f"Train patches : {len(train_ds):,}")
        print(f"Val patches   : {len(val_ds):,}")
        print(f"In channels   : {in_channels}")
        print(f"Out channels  : {out_channels}")
        print(f"Patch size    : {train_ds.patch_size}")
        print(f"Base filters  : {tuple(args.base_filters)}")
        print(f"Parameters    : {count_parameters(model):,}")

        loss_fn = get_loss_fn(args.loss)
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
            min_lr=args.min_lr,
        )

        use_amp = bool(args.amp and device.type == "cuda")
        scaler = torch.cuda.amp.GradScaler() if use_amp else None
        print(f"AMP enabled   : {use_amp}")
        print(f"Loss function : {args.loss}")

        history_csv = args.save_dir / "history.csv"
        history_jsonl = args.save_dir / "history.jsonl"
        per_epoch_dir = args.save_dir / "per_epoch"

        start_epoch = 1
        best_val_loss = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0

        if args.resume is not None:
            print(f"\nResuming from: {args.resume}")
            ckpt = torch.load(args.resume, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])

            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])

            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
            best_epoch = int(ckpt.get("best_epoch", ckpt.get("epoch", 0)))
            epochs_without_improvement = int(ckpt.get("epochs_without_improvement", 0))

            print(f"Resume start epoch: {start_epoch}")
            print(f"Best val so far   : {best_val_loss:.6f}")
            print(f"Best epoch so far : {best_epoch}")

        if start_epoch > args.epochs:
            print(f"Nothing to do: start_epoch={start_epoch} > epochs={args.epochs}")
            return

        for epoch in range(start_epoch, args.epochs + 1):
            epoch_start = datetime.now()
            print("\n" + "-" * 80)
            print(f"Starting epoch {epoch}/{args.epochs} at {epoch_start.isoformat(timespec='seconds')}")

            train_loss = train_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=device,
                epoch=epoch,
                scaler=scaler,
                grad_clip=args.grad_clip,
                log_every=args.log_every,
            )

            print("Validation starting ...")
            val_loss = validate(model, val_loader, loss_fn, device, epoch)
            scheduler.step(val_loss)

            current_lr = float(optimizer.param_groups[0]["lr"])
            seconds_epoch = (datetime.now() - epoch_start).total_seconds()

            is_best = False
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                is_best = True
            else:
                epochs_without_improvement += 1

            ckpt_common = {
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
                "base_filters": tuple(args.base_filters),
                "use_batchnorm": args.use_batchnorm,
                "final_activation": args.final_activation,
                "loss": args.loss,
                "fill_target_nan": fill_target_nan,
                "train_h5": str(args.train_h5),
                "val_h5": str(args.val_h5),
                "args": vars(args),
            }

            save_checkpoint(args.save_dir / "latest_unet.pt", ckpt_common)

            if is_best:
                save_checkpoint(args.save_dir / "best_unet.pt", ckpt_common)
                print(f"Saved new best model at epoch {epoch} with val_loss={best_val_loss:.6f}")

            if args.save_every_epoch:
                save_checkpoint(per_epoch_dir / f"epoch_{epoch:03d}.pt", ckpt_common)

            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": current_lr,
                "best_val_loss_so_far": best_val_loss,
                "best_epoch": best_epoch,
                "is_best": int(is_best),
                "seconds_epoch": seconds_epoch,
            }
            append_history_csv(history_csv, row)
            append_history_jsonl(history_jsonl, row)

            print(
                f"Epoch [{epoch:03d}/{args.epochs}] "
                f"train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} "
                f"best_val={best_val_loss:.6f} "
                f"best_epoch={best_epoch} "
                f"lr={current_lr:.2e} "
                f"time={seconds_epoch:.1f}s"
            )

            if device.type == "cuda":
                print(f"CUDA max memory allocated: {torch.cuda.max_memory_allocated() / 1024**3:.3f} GB")

            if epochs_without_improvement >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch}. Best epoch was {best_epoch}.")
                break

        print("\nTraining complete.")
        print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
        print(f"History CSV : {history_csv}")
        print(f"History JSONL: {history_jsonl}")
        print(f"Log file    : {log_path}")

    finally:
        sys.stdout = old_stdout
        tee.close()


if __name__ == "__main__":
    main()
