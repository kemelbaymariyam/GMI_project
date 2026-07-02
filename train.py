#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast sequential H5 trainer.

Why this exists:
The normal PyTorch DataLoader calls Dataset.__getitem__ once per sample.
For batch_size=16 that can mean 16 separate HDF5 reads per batch.
This script reads one whole contiguous H5 slice per batch:
    X_gmi[start:end], Y[start:end]
That is much better for large H5 files on Lustre when shuffle=False.

Best use:
  1) Build/train H5 in a physically randomized order once.
  2) Train sequentially from that H5 with this script.

Example:
CUDA_VISIBLE_DEVICES=2 python train.py \
  --train-h5 /lustre/home/mariyam/GMI_1C-R/fixed_train/train_fixed_360k.h5 \
  --val-h5 /lustre/home/mariyam/GMI_1C-R/fixed_val/val_fixed_50k.h5 \
  --save-dir checkpoints_fastseq \
  --batch-size 64 \
  --val-batch-size 128 \
  --epochs 100 \
  --lr 1e-3 \
  --amp \
  --val-every 2
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from gmi.model import UNet, count_parameters


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


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


def inspect_h5(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"H5 file not found: {path}")

    with h5py.File(path, "r") as f:
        if "X_gmi" not in f or "Y" not in f:
            raise KeyError(f"{path} must contain X_gmi and Y")

        info = {
            "path": str(path),
            "n": int(f["X_gmi"].shape[0]),
            "x_gmi_shape": tuple(f["X_gmi"].shape),
            "y_shape": tuple(f["Y"].shape),
            "has_static": "X_static" in f,
            "x_static_shape": tuple(f["X_static"].shape) if "X_static" in f else None,
            "x_gmi_chunks": f["X_gmi"].chunks,
            "y_chunks": f["Y"].chunks,
            "compression_x_gmi": f["X_gmi"].compression,
            "compression_y": f["Y"].compression,
            "attrs": {k: str(v) for k, v in f.attrs.items()},
        }
        if info["has_static"]:
            info["x_static_chunks"] = f["X_static"].chunks
            info["compression_x_static"] = f["X_static"].compression
        return info


def assert_compatible(train_info: dict, val_info: dict, use_static: bool) -> None:
    if train_info["x_gmi_shape"][1:] != val_info["x_gmi_shape"][1:]:
        raise ValueError(f"X_gmi train/val shape mismatch: {train_info['x_gmi_shape']} vs {val_info['x_gmi_shape']}")
    if train_info["y_shape"][1:] != val_info["y_shape"][1:]:
        raise ValueError(f"Y train/val shape mismatch: {train_info['y_shape']} vs {val_info['y_shape']}")
    if use_static:
        if not train_info["has_static"]:
            raise ValueError("You used --use-static but train H5 has no X_static")
        if not val_info["has_static"]:
            raise ValueError("You used --use-static but val H5 has no X_static")
        if train_info["x_static_shape"][1:] != val_info["x_static_shape"][1:]:
            raise ValueError("X_static train/val shape mismatch")


class SequentialH5Batches:
    """
    Iterable that yields already-batched tensors by reading contiguous H5 slices.
    This avoids per-sample __getitem__ overhead.
    """

    def __init__(
        self,
        h5_path: Path,
        batch_size: int,
        use_static: bool = False,
        fill_target_nan: Optional[float] = 0.0,
        device: Optional[torch.device] = None,
        pin_memory: bool = False,
        max_batches: Optional[int] = None,
    ):
        self.h5_path = Path(h5_path)
        self.batch_size = int(batch_size)
        self.use_static = bool(use_static)
        self.fill_target_nan = fill_target_nan
        self.device = device
        self.pin_memory = pin_memory
        self.max_batches = max_batches

        info = inspect_h5(self.h5_path)
        self.n = info["n"]
        self.gmi_channels = info["x_gmi_shape"][1]
        self.target_channels = info["y_shape"][1]
        self.patch_size = info["x_gmi_shape"][-1]
        self.static_channels = info["x_static_shape"][1] if self.use_static else 0

    def __len__(self) -> int:
        total = math.ceil(self.n / self.batch_size)
        if self.max_batches is not None:
            return min(total, self.max_batches)
        return total

    def __iter__(self):
        # Open inside iterator so the file handle is fresh for this pass.
        with h5py.File(self.h5_path, "r") as f:
            xg = f["X_gmi"]
            yds = f["Y"]
            xs = f["X_static"] if self.use_static else None

            yielded = 0
            for start in range(0, self.n, self.batch_size):
                if self.max_batches is not None and yielded >= self.max_batches:
                    break

                end = min(start + self.batch_size, self.n)

                # One contiguous HDF5 read per array.
                x_np = xg[start:end]
                if self.use_static:
                    static_np = xs[start:end]
                    x_np = np.concatenate([x_np, static_np], axis=1)

                y_np = yds[start:end]

                # Convert dtype and handle NaNs. This is not usually the bottleneck.
                x_np = np.asarray(x_np, dtype=np.float32)
                y_np = np.asarray(y_np, dtype=np.float32)

                if self.fill_target_nan is not None:
                    # copy=False when possible.
                    y_np = np.nan_to_num(y_np, nan=float(self.fill_target_nan), copy=False)

                x = torch.from_numpy(x_np)
                y = torch.from_numpy(y_np)

                if self.pin_memory and torch.cuda.is_available():
                    x = x.pin_memory()
                    y = y.pin_memory()

                if self.device is not None:
                    x = x.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)

                yielded += 1
                yield x, y


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = torch.isfinite(target)
    if not mask.any():
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
    raise ValueError(name)


def append_csv(path: Path, row: dict) -> None:
    fieldnames = [
        "timestamp", "epoch", "train_loss", "val_loss", "lr",
        "best_val_loss_so_far", "best_epoch", "is_best", "seconds_epoch"
    ]
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow(row)


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def save_ckpt(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def train_one_epoch(model, batches, optimizer, loss_fn, epoch, scaler=None, grad_clip=None, log_every=100):
    model.train()
    total_loss = 0.0
    total_seen = 0
    pbar = tqdm(batches, total=len(batches), desc=f"Train {epoch}", dynamic_ncols=True, leave=False)

    for step, (x, y) in enumerate(pbar, start=1):
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                pred = model(x)
                loss = loss_fn(pred, y)
                if not torch.isfinite(loss):
                    print(f"Non-finite loss at epoch {epoch}, step {step}: {loss.item()}", flush=True)
                    print(f"x finite: {torch.isfinite(x).all().item()}", flush=True)
                    print(f"y finite: {torch.isfinite(y).all().item()}", flush=True)
                    print(f"pred finite: {torch.isfinite(pred).all().item()}", flush=True)
                    raise RuntimeError("Stopping because loss became NaN/Inf")
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(x)
            loss = loss_fn(pred, y)
            if not torch.isfinite(loss):
                print(f"Non-finite loss at epoch {epoch}, step {step}: {loss.item()}", flush=True)
                print(f"x finite: {torch.isfinite(x).all().item()}", flush=True)
                print(f"y finite: {torch.isfinite(y).all().item()}", flush=True)
                print(f"pred finite: {torch.isfinite(pred).all().item()}", flush=True)
                raise RuntimeError("Stopping because loss became NaN/Inf")
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        bs = x.shape[0]
        total_loss += float(loss.item()) * bs
        total_seen += bs
        avg = total_loss / max(total_seen, 1)
        pbar.set_postfix(batch_loss=f"{loss.item():.4f}", avg=f"{avg:.4f}")

        if step % log_every == 0:
            print(f"[epoch {epoch:03d} step {step:06d}/{len(batches):06d}] "
                  f"batch_loss={loss.item():.6f} running_train_loss={avg:.6f}",
                  flush=True)

    return total_loss / max(total_seen, 1)


@torch.no_grad()
def validate(model, batches, loss_fn, epoch):
    model.eval()
    total_loss = 0.0
    total_seen = 0
    pbar = tqdm(batches, total=len(batches), desc=f"Val {epoch}", dynamic_ncols=True, leave=False)

    for x, y in pbar:
        pred = model(x)
        loss = loss_fn(pred, y)
        bs = x.shape[0]
        total_loss += float(loss.item()) * bs
        total_seen += bs
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(total_seen, 1)


def build_parser():
    p = argparse.ArgumentParser(description="Fast sequential fixed-H5 U-Net trainer")
    p.add_argument("--train-h5", type=Path, required=True)
    p.add_argument("--val-h5", type=Path, required=True)
    p.add_argument("--save-dir", type=Path, required=True)

    p.add_argument("--use-static", action="store_true")
    p.add_argument("--fill-target-nan", type=str, default="0.0")
    p.add_argument("--loss", choices=["mse", "l1", "smooth_l1", "masked_mse"], default="mse")

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--val-batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--val-every", type=int, default=1, help="Run full validation every N epochs")
    p.add_argument("--max-train-batches", type=int, default=None, help="debug only")
    p.add_argument("--max-val-batches", type=int, default=None, help="debug only")

    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--base-filters", type=int, nargs=4, default=[8, 16, 32, 64])
    p.add_argument("--use-batchnorm", action="store_true")
    p.add_argument("--final-activation", choices=["identity", "sigmoid", "tanh"], default="identity")

    p.add_argument("--amp", action="store_true")
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--save-every-epoch", action="store_true")
    p.add_argument("--early-stop-patience", type=int, default=20)
    p.add_argument("--scheduler-patience", type=int, default=6)
    p.add_argument("--scheduler-factor", type=float, default=0.5)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--log-every", type=int, default=100)
    return p


def main():
    args = build_parser().parse_args()
    seed_everything(args.seed)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    tee = TeeLogger(args.save_dir / "train.log")
    old_stdout = sys.stdout
    sys.stdout = tee

    try:
        print("=" * 80)
        print(f"Started: {datetime.now().isoformat(timespec='seconds')}")
        print(f"Args: {vars(args)}")
        print("=" * 80)

        if args.fill_target_nan.lower() == "none":
            fill_target_nan = None
        else:
            fill_target_nan = float(args.fill_target_nan)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")
        if device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        train_info = inspect_h5(args.train_h5)
        val_info = inspect_h5(args.val_h5)
        assert_compatible(train_info, val_info, args.use_static)

        print("\nTrain H5 info:")
        for k, v in train_info.items():
            print(f"  {k}: {v}")
        print("\nVal H5 info:")
        for k, v in val_info.items():
            print(f"  {k}: {v}")

        val_bs = args.val_batch_size or args.batch_size

        # Move batches to GPU inside iterator. This keeps training loop simple.
        train_batches = lambda: SequentialH5Batches(
            args.train_h5, args.batch_size, args.use_static, fill_target_nan,
            device=device, pin_memory=False, max_batches=args.max_train_batches
        )
        val_batches = lambda: SequentialH5Batches(
            args.val_h5, val_bs, args.use_static, fill_target_nan,
            device=device, pin_memory=False, max_batches=args.max_val_batches
        )

        in_channels = train_info["x_gmi_shape"][1] + (train_info["x_static_shape"][1] if args.use_static else 0)
        out_channels = train_info["y_shape"][1]

        model = UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_filters=tuple(args.base_filters),
            use_batchnorm=args.use_batchnorm,
            final_activation=args.final_activation,
        ).to(device)

        print("\nModel:")
        print(f"  in_channels: {in_channels}")
        print(f"  out_channels: {out_channels}")
        print(f"  parameters: {count_parameters(model):,}")
        print(f"  train batches/epoch: {len(train_batches()):,}")
        print(f"  val batches: {len(val_batches()):,}")
        print(f"  train batch size: {args.batch_size}")
        print(f"  val batch size: {val_bs}")

        loss_fn = get_loss_fn(args.loss)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=args.scheduler_factor,
            patience=args.scheduler_patience, min_lr=args.min_lr
        )

        scaler = torch.amp.GradScaler("cuda") if (args.amp and device.type == "cuda") else None
        print(f"AMP: {scaler is not None}")
        print(f"Loss: {args.loss}")

        start_epoch = 1
        best_val_loss = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0
        last_val_loss = float("nan")

        if args.resume is not None:
            print(f"\nResuming from {args.resume}")
            ckpt = torch.load(args.resume, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr
            print(f"Reset optimizer LR to {args.lr}")
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
            best_epoch = int(ckpt.get("best_epoch", ckpt.get("epoch", 0)))
            epochs_without_improvement = int(ckpt.get("epochs_without_improvement", 0))
            print(f"  start_epoch: {start_epoch}")
            print(f"  best_val_loss: {best_val_loss}")
            print(f"  best_epoch: {best_epoch}")

        history_csv = args.save_dir / "history.csv"
        history_jsonl = args.save_dir / "history.jsonl"

        for epoch in range(start_epoch, args.epochs + 1):
            t0 = time.time()
            print("\n" + "-" * 80)
            print(f"Starting epoch {epoch}/{args.epochs} at {datetime.now().isoformat(timespec='seconds')}")

            train_loss = train_one_epoch(
                model, train_batches(), optimizer, loss_fn, epoch,
                scaler=scaler, grad_clip=args.grad_clip, log_every=args.log_every
            )

            run_val = (epoch == 1) or (epoch % args.val_every == 0) or (epoch == args.epochs)

            is_best = False
            if run_val:
                print("Validation starting ...")
                val_loss = validate(model, val_batches(), loss_fn, epoch)
                last_val_loss = val_loss
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    epochs_without_improvement = 0
                    is_best = True
                else:
                    epochs_without_improvement += 1
            else:
                val_loss = last_val_loss
                print(f"Skipping validation this epoch because --val-every {args.val_every}")

            current_lr = float(optimizer.param_groups[0]["lr"])
            seconds_epoch = time.time() - t0

            ckpt_payload = {
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

            save_ckpt(args.save_dir / "latest_unet.pt", ckpt_payload)
            if is_best:
                save_ckpt(args.save_dir / "best_unet.pt", ckpt_payload)
                print(f"Saved new best model: val_loss={best_val_loss:.6f}")

            if args.save_every_epoch:
                save_ckpt(args.save_dir / "per_epoch" / f"epoch_{epoch:03d}.pt", ckpt_payload)

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
            append_csv(history_csv, row)
            append_jsonl(history_jsonl, row)

            print(
                f"Epoch [{epoch:03d}/{args.epochs}] "
                f"train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} "
                f"best_val={best_val_loss:.6f} "
                f"best_epoch={best_epoch} "
                f"lr={current_lr:.2e} "
                f"time={seconds_epoch/60:.1f} min"
            )

            if run_val and epochs_without_improvement >= args.early_stop_patience:
                print(f"Early stopping. Best epoch was {best_epoch}.")
                break

        print("\nDone.")
        print(f"Best val loss: {best_val_loss:.6f} at epoch {best_epoch}")

    finally:
        sys.stdout = old_stdout
        tee.close()


if __name__ == "__main__":
    main()
