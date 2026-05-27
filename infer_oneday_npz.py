#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from unet import UNet


def build_parser():
    p = argparse.ArgumentParser(description="Run inference on one daily NPZ shard and save Panoply-friendly HDF5")
    p.add_argument("--npz-file", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-h5", type=Path, required=True)
    p.add_argument("--use-static", action="store_true")
    p.add_argument("--start-patch", type=int, default=0)
    p.add_argument("--num-patches", type=int, default=4)
    return p


def main():
    args = build_parser().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    with np.load(args.npz_file, allow_pickle=False) as data:
        X_gmi = data["X_gmi"].astype(np.float32)
        X_static = data["X_static"].astype(np.float32) if "X_static" in data else None
        Y = data["Y"].astype(np.float32)
        patch_rc = data["patch_rc"].astype(np.int32) if "patch_rc" in data else None
        target_day = str(data["target_day"].item()) if "target_day" in data else ""

    ckpt = torch.load(args.checkpoint, map_location=device)

    model = UNet(
        in_channels=ckpt["in_channels"],
        out_channels=ckpt["out_channels"],
        base_filters=tuple(ckpt.get("base_filters", (8, 16, 32, 64))),
        use_batchnorm=False,
        final_activation="identity",
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    args.output_h5.parent.mkdir(parents=True, exist_ok=True)

    end_patch = min(args.start_patch + args.num_patches, X_gmi.shape[0])

    with h5py.File(args.output_h5, "w") as f:
        f.attrs["npz_file"] = str(args.npz_file)
        f.attrs["checkpoint"] = str(args.checkpoint)
        f.attrs["target_day"] = target_day
        f.attrs["use_static"] = args.use_static

        for patch_idx in range(args.start_patch, end_patch):
            if args.use_static and X_static is not None:
                x = np.concatenate([X_gmi[patch_idx], X_static[patch_idx]], axis=0)
            else:
                x = X_gmi[patch_idx]

            y_true = Y[patch_idx]

            with torch.no_grad():
                x_t = torch.from_numpy(x).unsqueeze(0).to(device)
                y_pred = model(x_t).cpu().squeeze(0).numpy().astype(np.float32)

            grp = f.create_group(f"patch_{patch_idx:03d}")

            # full tensors
            grp.create_dataset("X", data=x.astype(np.float32), compression="gzip")
            grp.create_dataset("Y_true", data=y_true.astype(np.float32), compression="gzip")
            grp.create_dataset("Y_pred", data=y_pred.astype(np.float32), compression="gzip")

            if patch_rc is not None:
                grp.create_dataset("patch_rc", data=patch_rc[patch_idx].astype(np.int32))

            # Panoply-friendly 2D datasets for each output channel
            for ch in range(y_true.shape[0]):
                grp.create_dataset(
                    f"Y_true_ch{ch:02d}",
                    data=y_true[ch].astype(np.float32),
                    compression="gzip",
                )
                grp.create_dataset(
                    f"Y_pred_ch{ch:02d}",
                    data=y_pred[ch].astype(np.float32),
                    compression="gzip",
                )

            print(f"Saved patch {patch_idx}")

    print("Saved:", args.output_h5)


if __name__ == "__main__":
    main()