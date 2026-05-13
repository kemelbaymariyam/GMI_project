#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper-inspired U-Net for GMI gap-filling / reconstruction.

Flexible channels:
- in_channels = 78  -> GMI only
- in_channels = 80  -> GMI + FRLAND + FRLANDICE
- out_channels = 13 -> target GMI channels
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class UNetConfig:
    in_channels: int = 80
    out_channels: int = 13
    base_filters: tuple[int, int, int, int] = (8, 16, 32, 64)
    use_batchnorm: bool = False
    final_activation: str = "identity"  # identity | sigmoid | tanh


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_batchnorm: bool = False) -> None:
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=not use_batchnorm)]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=not use_batchnorm))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, use_batchnorm: bool = False) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch, use_batchnorm=use_batchnorm)

    @staticmethod
    def _match_size(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        _, _, rh, rw = ref.shape

        dh = rh - h
        dw = rw - w

        if dh > 0 or dw > 0:
            pad_left = max(dw // 2, 0)
            pad_right = max(dw - pad_left, 0)
            pad_top = max(dh // 2, 0)
            pad_bottom = max(dh - pad_top, 0)
            x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))

        _, _, h, w = x.shape
        if h > rh:
            top = (h - rh) // 2
            x = x[:, :, top:top + rh, :]
        if w > rw:
            left = (w - rw) // 2
            x = x[:, :, :, left:left + rw]
        return x

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self._match_size(x, skip)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetPaperLike(nn.Module):
    """
    Paper-inspired U-Net:
      encoder filters: 8, 16, 32, 64
      bottleneck: 128
      decoder: 64, 32, 16, 8
    """
    def __init__(
        self,
        in_channels: int = 80,
        out_channels: int = 13,
        base_filters: tuple[int, int, int, int] = (8, 16, 32, 64),
        use_batchnorm: bool = False,
        final_activation: str = "identity",
    ) -> None:
        super().__init__()
        if len(base_filters) != 4:
            raise ValueError("base_filters must have length 4")

        f1, f2, f3, f4 = base_filters
        bottleneck_ch = f4 * 2

        self.enc1 = ConvBlock(in_channels, f1, use_batchnorm)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(f1, f2, use_batchnorm)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(f2, f3, use_batchnorm)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(f3, f4, use_batchnorm)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(f4, bottleneck_ch, use_batchnorm)

        self.dec4 = UpBlock(bottleneck_ch, f4, f4, use_batchnorm)
        self.dec3 = UpBlock(f4, f3, f3, use_batchnorm)
        self.dec2 = UpBlock(f3, f2, f2, use_batchnorm)
        self.dec1 = UpBlock(f2, f1, f1, use_batchnorm)

        self.out_conv = nn.Conv2d(f1, out_channels, kernel_size=1)

        if final_activation == "identity":
            self.final_activation = nn.Identity()
        elif final_activation == "sigmoid":
            self.final_activation = nn.Sigmoid()
        elif final_activation == "tanh":
            self.final_activation = nn.Tanh()
        else:
            raise ValueError("final_activation must be identity, sigmoid, or tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        x = self.pool1(s1)

        s2 = self.enc2(x)
        x = self.pool2(s2)

        s3 = self.enc3(x)
        x = self.pool3(s3)

        s4 = self.enc4(x)
        x = self.pool4(s4)

        x = self.bottleneck(x)

        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        x = self.out_conv(x)
        return self.final_activation(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    cfg = UNetConfig(in_channels=80, out_channels=13)
    model = UNetPaperLike(
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        base_filters=cfg.base_filters,
        use_batchnorm=cfg.use_batchnorm,
        final_activation=cfg.final_activation,
    )
    x = torch.randn(2, cfg.in_channels, 40, 40)
    y = model(x)
    print("Input shape :", tuple(x.shape))
    print("Output shape:", tuple(y.shape))
    print("Trainable params:", count_parameters(model))
