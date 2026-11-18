"""Fixed, non-learnable spatial frequency operators."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def channel_mean(value: torch.Tensor) -> torch.Tensor:
    """Return the per-image, per-channel spatial mean."""
    if value.ndim != 4:
        raise ValueError(f"expected BCHW input, got {tuple(value.shape)}")
    return value.mean(dim=(-2, -1), keepdim=True)


def remove_channel_mean(value: torch.Tensor) -> torch.Tensor:
    """Remove the per-image, per-channel spatial mean."""
    return value - channel_mean(value)


class GaussianLowPass(nn.Module):
    """Depthwise, fixed 5 x 5 Gaussian low-pass with reflect padding."""

    def __init__(self, sigma: float = 1.0, radius: int = 2) -> None:
        super().__init__()
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        if radius < 1:
            raise ValueError("radius must be at least 1")
        offsets = torch.arange(-radius, radius + 1, dtype=torch.float64)
        one_dimensional = torch.exp(-(offsets.square()) / (2.0 * sigma**2))
        one_dimensional = one_dimensional / one_dimensional.sum()
        kernel = one_dimensional[:, None] * one_dimensional[None, :]
        self.register_buffer("kernel", kernel[None, None].float(), persistent=True)
        self.padding = radius
        self.sigma = sigma

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4:
            raise ValueError(f"expected BCHW input, got {tuple(value.shape)}")
        channels = value.shape[1]
        weight = self.kernel.expand(channels, 1, *self.kernel.shape[-2:])
        value = F.pad(value, (self.padding,) * 4, mode="reflect")
        return F.conv2d(
            value,
            weight,
            bias=None,
            stride=1,
            padding=0,
            groups=channels,
        )

    @torch.no_grad()
    def minus_three_db_nyquist_fraction(self, fft_size: int = 65536) -> float:
        """Measure the discrete kernel's -3 dB point along one frequency axis."""
        radius = self.padding
        offsets = torch.arange(
            -radius,
            radius + 1,
            device=self.kernel.device,
            dtype=self.kernel.dtype,
        )
        frequencies = torch.fft.fftshift(
            torch.fft.fftfreq(fft_size, d=1.0, device=self.kernel.device)
        )
        kernel_1d = self.kernel[0, 0].sum(dim=0).to(torch.complex64)
        response = torch.abs(
            torch.exp(-2j * math.pi * frequencies[:, None] * offsets[None, :])
            @ kernel_1d
        )
        response = response / response.max()
        target = 1.0 / math.sqrt(2.0)
        positive = frequencies >= 0
        frequencies = frequencies[positive]
        response = response[positive]
        crossing = torch.nonzero(response <= target, as_tuple=False)[0, 0]
        if crossing == 0:
            return 0.0
        f0, f1 = frequencies[crossing - 1], frequencies[crossing]
        r0, r1 = response[crossing - 1], response[crossing]
        frequency = f0 + (target - r0) * (f1 - f0) / (r1 - r0)
        return float((frequency / 0.5).item())
