#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class GMIPatchH5Dataset(Dataset):
    def __init__(
        self,
        h5_path: str | Path,
        use_static: bool = True,
        fill_target_nan: Optional[float] = 0.0,
        return_metadata: bool = False,
    ) -> None:
        super().__init__()
        self.h5_path = Path(h5_path)
        self.use_static = use_static
        self.fill_target_nan = fill_target_nan
        self.return_metadata = return_metadata

        if not self.h5_path.exists():
            raise FileNotFoundError(f"H5 file not found: {self.h5_path}")

        self._h5 = None

        with h5py.File(self.h5_path, "r") as f:
            self.n = int(f["X_gmi"].shape[0])
            self.gmi_channels = int(f["X_gmi"].shape[1])
            self.static_channels = int(f["X_static"].shape[1]) if "X_static" in f else 0
            self.target_channels = int(f["Y"].shape[1])
            self.patch_size = int(f["X_gmi"].shape[2])
            self.has_patch_rc = "patch_rc" in f
            self.has_target_day = "target_day" in f
            self.has_source_file = "source_file" in f

    def _ensure_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int):
        self._ensure_open()

        x_gmi = self._h5["X_gmi"][idx].astype(np.float32)
        y = self._h5["Y"][idx].astype(np.float32)

        if self.use_static:
            x_static = self._h5["X_static"][idx].astype(np.float32)
            x = np.concatenate([x_gmi, x_static], axis=0)
        else:
            x = x_gmi

        if self.fill_target_nan is not None:
            y = np.where(np.isfinite(y), y, np.float32(self.fill_target_nan))

        x_t = torch.from_numpy(x)
        y_t = torch.from_numpy(y)

        if not self.return_metadata:
            return x_t, y_t

        meta = {"index": idx}
        if self.has_patch_rc:
            meta["patch_rc"] = self._h5["patch_rc"][idx].astype(np.int32)
        if self.has_target_day:
            v = self._h5["target_day"][idx]
            meta["target_day"] = v.decode("utf-8") if isinstance(v, bytes) else str(v)
        if self.has_source_file:
            v = self._h5["source_file"][idx]
            meta["source_file"] = v.decode("utf-8") if isinstance(v, bytes) else str(v)
        return x_t, y_t, meta


def make_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
