#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Training script for the paper-inspired U-Net.

Expected data format:
- X_train.pt: tensor with shape [N, 80, H, W]
- Y_train.pt: tensor with shape [N, 13, H, W]
- X_val.pt:   tensor with shape [N, 80, H, W]
- Y_val.pt:   tensor with shape [N, 13, H, W]

Example:
    python train.py
"""

from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

from unet import UNetPaperLike, count_parameters


# -----------------------------
# 1. Dataset
# -----------------------------

class TensorDataset2D(Dataset):
    def __init__(self, x_path: str, y_path: str):
        self.x = torch.load(x_path).float()
        self.y = torch.load(y_path).float()

        if self.x.ndim != 4:
            raise ValueError(f"X must have shape [N, C, H, W], got {self.x.shape}")

        if self.y.ndim != 4:
            raise ValueError(f"Y must have shape [N, C, H, W], got {self.y.shape}")

        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError("X and Y must have the same number of samples")

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# -----------------------------
# 2. Train one epoch
# -----------------------------

def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()

    total_loss = 0.0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        # Forward pass
        pred = model(x)

        # Compute loss
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()

        # Gradient descent step
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss


# -----------------------------
# 3. Validate
# -----------------------------

@torch.no_grad()
def validate(model, dataloader, loss_fn, device):
    model.eval()

    total_loss = 0.0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)

        total_loss += loss.item() * x.size(0)

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss


# -----------------------------
# 4. Main training function
# -----------------------------

def main():
    # Paths
    data_dir = Path("data")

    x_train_path = data_dir / "X_train.pt"
    y_train_path = data_dir / "Y_train.pt"
    x_val_path = data_dir / "X_val.pt"
    y_val_path = data_dir / "Y_val.pt"

    # Training settings
    batch_size = 16
    num_epochs = 100
    learning_rate = 1e-3

    in_channels = 80
    out_channels = 13

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Datasets and loaders
    train_dataset = TensorDataset2D(x_train_path, y_train_path)
    val_dataset = TensorDataset2D(x_val_path, y_val_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
    )

    # Model
    model = UNetPaperLike(
        in_channels=in_channels,
        out_channels=out_channels,
        base_filters=(8, 16, 32, 64),
        use_batchnorm=False,
        final_activation="identity",
    ).to(device)

    print(f"Trainable parameters: {count_parameters(model):,}")

    # Loss function
    # For reconstruction / gap-filling, MSE is common.
    loss_fn = nn.MSELoss()

    # Optimizer
    # Adam is gradient descent with adaptive learning rates.
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Optional learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=8,
    )

    # Save directory
    save_dir = Path("checkpoints")
    save_dir.mkdir(exist_ok=True)

    best_val_loss = float("inf")

    # Training loop
    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )

        val_loss = validate(
            model=model,
            dataloader=val_loader,
            loss_fn=loss_fn,
            device=device,
        )

        scheduler.step(val_loss)

        print(
            f"Epoch [{epoch:03d}/{num_epochs}] "
            f"Train Loss: {train_loss:.6f} "
            f"Val Loss: {val_loss:.6f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "in_channels": in_channels,
                "out_channels": out_channels,
                "base_filters": (8, 16, 32, 64),
            }

            torch.save(checkpoint, save_dir / "best_unet.pt")
            print(f"Saved best model with val loss: {best_val_loss:.6f}")

    print("Training complete.")


if __name__ == "__main__":
    main()