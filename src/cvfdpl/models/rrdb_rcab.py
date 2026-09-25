"""Compact RRDB plus RCAB carrier used by the v3.1 experiments."""

from __future__ import annotations

import torch
from torch import nn


class RRDB(nn.Module):
    """Residual-in-residual dense block."""

    def __init__(self, num_feat: int = 32, num_grow_ch: int = 16) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.c2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.c3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_feat, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.scale = 0.25

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.act(self.c1(x))
        x2 = self.act(self.c2(torch.cat([x, x1], dim=1)))
        x3 = self.c3(torch.cat([x, x1, x2], dim=1))
        return x + x3 * self.scale


class RCAB(nn.Module):
    """Lightweight residual channel attention block."""

    def __init__(self, num_feat: int = 32) -> None:
        super().__init__()
        if num_feat < 4:
            raise ValueError("num_feat must be at least 4")
        self.feat = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            nn.BatchNorm2d(num_feat),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            nn.BatchNorm2d(num_feat),
        )
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // 4, num_feat, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.feat(x)
        return x + residual * self.attn(residual)


class RRDBPlusLightRCAB(nn.Module):
    """4 x RRDB trunk plus an ``n_rcab`` RCAB high-frequency branch."""

    def __init__(
        self,
        in_channels: int = 3,
        num_feat: int = 32,
        num_grow_ch: int = 16,
        num_rrdb: int = 4,
        num_rcab: int = 3,
    ) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, num_feat, 3, 1, 1)
        self.body = nn.Sequential(
            *[RRDB(num_feat, num_grow_ch) for _ in range(num_rrdb)]
        )
        self.branch_high = nn.Sequential(*[RCAB(num_feat) for _ in range(num_rcab)])
        self.conv_fusion = nn.Conv2d(2 * num_feat, num_feat, 3, 1, 1)
        self.conv_out = nn.Conv2d(num_feat, in_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_in(x)
        feat_body = self.body(feat)
        feat_high = self.branch_high(feat)
        fused = self.conv_fusion(torch.cat([feat_body, feat_high], dim=1))
        return self.conv_out(fused) + x


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Return the number of model parameters."""

    parameters = model.parameters()
    if trainable_only:
        return sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)
