#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch dataset for GMI patch shards saved as compressed .npz files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class DatasetConfig:
    root: str
    split: str = "train"
    use_static: bool = True
    fill_target_nan: Optional[float] = 0.0
    return_metadata: bool = False


def _find_npz_files(root: Path, split: str) -> List[Path]:
    split_dir = root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split folder not found: {split_dir}")
    files = sorted(split_dir.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found under: {split_dir}")
    return files


class GMIPatchDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        use_static: bool = True,
        fill_target_nan: Optional[float] = 0.0,
        return_metadata: bool = False,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.use_static = use_static
        self.fill_target_nan = fill_target_nan
        self.return_metadata = return_metadata

        self.files: List[Path] = _find_npz_files(self.root, self.split)
        self.index: List[Tuple[int, int]] = []
        self.file_patch_counts: List[int] = []

        for file_id, path in enumerate(self.files):
            with np.load(path, allow_pickle=False) as data:
                n = int(data["X_gmi"].shape[0])
            self.file_patch_counts.append(n)
            self.index.extend((file_id, patch_id) for patch_id in range(n))

        self._cache_file_id: Optional[int] = None
        self._cache_data: Optional[Dict[str, np.ndarray]] = None

        first = self._load_shard(0)
        self.gmi_channels = int(first["X_gmi"].shape[1])
        self.static_channels = int(first["X_static"].shape[1]) if "X_static" in first else 0
        self.target_channels = int(first["Y"].shape[1])
        self.patch_size = int(first["X_gmi"].shape[2])

    def __len__(self) -> int:
        return len(self.index)

    def _load_shard(self, file_id: int) -> Dict[str, np.ndarray]:
        if self._cache_file_id == file_id and self._cache_data is not None:
            return self._cache_data

        path = self.files[file_id]
        with np.load(path, allow_pickle=False) as data:
            shard = {k: data[k] for k in data.files}

        self._cache_file_id = file_id
        self._cache_data = shard
        return shard

    def __getitem__(self, idx: int):
        file_id, patch_id = self.index[idx]
        shard = self._load_shard(file_id)

        x_gmi = shard["X_gmi"][patch_id].astype(np.float32)
        y = shard["Y"][patch_id].astype(np.float32)

        if self.use_static:
            x_static = shard["X_static"][patch_id].astype(np.float32)
            x = np.concatenate([x_gmi, x_static], axis=0)
        else:
            x = x_gmi

        if self.fill_target_nan is not None:
            y = np.where(np.isfinite(y), y, np.float32(self.fill_target_nan))

        x_tensor = torch.from_numpy(x)
        y_tensor = torch.from_numpy(y)

        if not self.return_metadata:
            return x_tensor, y_tensor

        patch_rc = shard["patch_rc"][patch_id].astype(np.int32)
        td = shard["target_day"]
        target_day = str(td.item()) if np.ndim(td) == 0 else str(td)
        meta = {
            "file_path": str(self.files[file_id]),
            "file_id": file_id,
            "patch_id": patch_id,
            "patch_rc": patch_rc,
            "target_day": target_day,
            "split": self.split,
        }
        return x_tensor, y_tensor, meta


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


if __name__ == "__main__":
    root = "/lustre/home/mariyam/GMI_1C-R/patches_split_byday_cnnprep"
    split = "train"
    try:
        ds = GMIPatchDataset(root=root, split=split, use_static=True, return_metadata=True)
        print("Number of shard files :", len(ds.files))
        print("Total number of patches:", len(ds))
        print("GMI channels          :", ds.gmi_channels)
        print("Static channels       :", ds.static_channels)
        print("Target channels       :", ds.target_channels)
        print("Patch size            :", ds.patch_size)
        x, y, meta = ds[0]
        print("Sample X shape:", tuple(x.shape))
        print("Sample Y shape:", tuple(y.shape))
        print("Sample meta   :", meta)
    except Exception as e:
        print("Smoke test skipped / failed:", e)
