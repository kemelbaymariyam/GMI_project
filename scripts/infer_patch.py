#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal one-patch inference for Panoply.

Saves only the important variables for every output channel:
  Y_true_chXX
  Y_pred_chXX
  Y_filled_chXX
  missing_mask_chXX
  abs_error_chXX

Missing-value choices:
  --mask-source target_nan
      Missing where Y is NaN/Inf.

  --mask-source target_zero
      Missing where abs(Y) <= --zero-threshold.

  --mask-source input_channel
      Missing according to one X_gmi channel.

Display choices:
  --display-missing nan
      Missing values in Y_true are saved as NaN. Best for blank/transparent display.

  --display-missing zero
      Missing values in Y_true are saved as 0.

Y_filled always keeps known Y values and inserts prediction only where missing_mask=True.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from gmi.model import UNet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--npz-file", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-h5", type=Path, required=True)

    p.add_argument("--patch-index", type=int, default=0)
    p.add_argument("--use-static", action="store_true")

    p.add_argument(
        "--mask-source",
        choices=["target_nan", "target_zero", "input_channel"],
        default="target_nan",
    )
    p.add_argument(
        "--input-mask-channel",
        type=int,
        default=None,
        help="Required only for --mask-source input_channel.",
    )
    p.add_argument(
        "--zero-threshold",
        type=float,
        default=0.0,
        help="Values with abs(value) <= threshold are treated as zero/missing.",
    )
    p.add_argument(
        "--display-missing",
        choices=["nan", "zero"],
        default="nan",
        help="How missing pixels are stored in Y_true for Panoply.",
    )
    return p.parse_args()


def build_missing_mask(
    y_true: np.ndarray,
    x_gmi: np.ndarray,
    mask_source: str,
    zero_threshold: float,
    input_mask_channel: int | None,
) -> np.ndarray:
    if mask_source == "target_nan":
        return ~np.isfinite(y_true)

    if mask_source == "target_zero":
        return (~np.isfinite(y_true)) | (np.abs(y_true) <= zero_threshold)

    if input_mask_channel is None:
        raise ValueError(
            "--input-mask-channel is required with --mask-source input_channel"
        )

    if not 0 <= input_mask_channel < x_gmi.shape[0]:
        raise ValueError(
            f"input mask channel must be between 0 and {x_gmi.shape[0] - 1}"
        )

    mask_2d = (~np.isfinite(x_gmi[input_mask_channel])) | (
        np.abs(x_gmi[input_mask_channel]) <= zero_threshold
    )

    return np.broadcast_to(mask_2d[None, :, :], y_true.shape).copy()


def save_2d(group, name: str, data: np.ndarray, description: str):
    ds = group.create_dataset(
        name,
        data=np.asarray(data, dtype=np.float32),
        compression="gzip",
        shuffle=True,
    )
    ds.attrs["description"] = description
    return ds


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    with np.load(args.npz_file, allow_pickle=False) as npz:
        x_gmi_all = np.asarray(npz["X_gmi"], dtype=np.float32)
        y_all = np.asarray(npz["Y"], dtype=np.float32)

        x_static_all = (
            np.asarray(npz["X_static"], dtype=np.float32)
            if "X_static" in npz
            else None
        )

        patch_rc_all = (
            np.asarray(npz["patch_rc"])
            if "patch_rc" in npz
            else None
        )

        target_day = (
            str(npz["target_day"].item())
            if "target_day" in npz
            else ""
        )

    if not 0 <= args.patch_index < x_gmi_all.shape[0]:
        raise IndexError(
            f"patch index {args.patch_index} outside 0..{x_gmi_all.shape[0]-1}"
        )

    x_gmi = x_gmi_all[args.patch_index]
    y_true_raw = y_all[args.patch_index]

    if args.use_static:
        if x_static_all is None:
            raise ValueError("--use-static was given, but NPZ has no X_static")
        x_model = np.concatenate(
            [x_gmi, x_static_all[args.patch_index]],
            axis=0,
        )
    else:
        x_model = x_gmi

    ckpt = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    model = UNet(
        in_channels=int(ckpt["in_channels"]),
        out_channels=int(ckpt["out_channels"]),
        base_filters=tuple(ckpt.get("base_filters", (8, 16, 32, 64))),
        use_batchnorm=bool(ckpt.get("use_batchnorm", False)),
        final_activation=str(ckpt.get("final_activation", "identity")),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.inference_mode():
        x_tensor = torch.from_numpy(x_model).unsqueeze(0).to(device)
        y_pred = (
            model(x_tensor)
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    missing_mask = build_missing_mask(
        y_true=y_true_raw,
        x_gmi=x_gmi,
        mask_source=args.mask_source,
        zero_threshold=args.zero_threshold,
        input_mask_channel=args.input_mask_channel,
    )

    # Keep a clean known-value array for filling.
    y_known = np.nan_to_num(
        y_true_raw,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)

    # Panoply display version of the ground truth.
    y_true_display = y_known.copy()
    if args.display_missing == "nan":
        y_true_display[missing_mask] = np.nan
    else:
        y_true_display[missing_mask] = 0.0

    # Final reconstructed product:
    # known pixels from Y, missing pixels from model.
    y_filled = np.where(
        missing_mask,
        y_pred,
        y_known,
    ).astype(np.float32)

    # Error only where original target is truly available.
    valid_gt = np.isfinite(y_true_raw) & (~missing_mask)
    abs_error = np.where(
        valid_gt,
        np.abs(y_pred - y_true_raw),
        np.nan,
    ).astype(np.float32)

    args.output_h5.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.output_h5, "w") as f:
        f.attrs["npz_file"] = str(args.npz_file)
        f.attrs["checkpoint"] = str(args.checkpoint)
        f.attrs["patch_index"] = int(args.patch_index)
        f.attrs["target_day"] = target_day
        f.attrs["mask_source"] = args.mask_source
        f.attrs["display_missing"] = args.display_missing
        f.attrs["zero_threshold"] = float(args.zero_threshold)
        f.attrs["best_epoch"] = int(
            ckpt.get("best_epoch", ckpt.get("epoch", -1))
        )
        f.attrs["best_val_loss"] = float(
            ckpt.get("best_val_loss", np.nan)
        )

        if patch_rc_all is not None:
            f.create_dataset(
                "patch_rc",
                data=patch_rc_all[args.patch_index],
            )

        f.create_dataset(
            "missing_mask_all_channels",
            data=missing_mask.astype(np.uint8),
            compression="gzip",
        )

        for ch in range(y_true_raw.shape[0]):
            save_2d(
                f,
                f"Y_true_ch{ch:02d}",
                y_true_display[ch],
                "Ground truth for Panoply; missing pixels use selected display convention.",
            )
            save_2d(
                f,
                f"Y_pred_ch{ch:02d}",
                y_pred[ch],
                "Full model prediction for this output channel.",
            )
            save_2d(
                f,
                f"Y_filled_ch{ch:02d}",
                y_filled[ch],
                "Known target retained; missing pixels replaced by prediction.",
            )
            save_2d(
                f,
                f"missing_mask_ch{ch:02d}",
                missing_mask[ch].astype(np.float32),
                "1 means reconstructed/missing pixel; 0 means known pixel.",
            )
            save_2d(
                f,
                f"abs_error_ch{ch:02d}",
                abs_error[ch],
                "Absolute prediction error where original target is known.",
            )

    print("Saved:", args.output_h5)
    print("Patch index:", args.patch_index)
    print("Missing fraction:", float(missing_mask.mean()))
    print("Channels saved:", y_true_raw.shape[0])


if __name__ == "__main__":
    main()
