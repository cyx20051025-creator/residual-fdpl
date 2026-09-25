"""Frequency-domain perceptual losses."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import VGG19_Weights, vgg19


class BrightnessAwareLoss(nn.Module):
    """Apply extra weight to bright target pixels."""

    def __init__(
        self,
        base_loss: nn.Module | None = None,
        brightness_threshold: float = 0.6,
        weight_factor: float = 1.5,
    ) -> None:
        super().__init__()
        self.base_loss = base_loss if base_loss is not None else nn.MSELoss()
        self.brightness_threshold = brightness_threshold
        self.weight_factor = weight_factor

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        brightness = (
            0.299 * target[:, 0]
            + 0.587 * target[:, 1]
            + 0.114 * target[:, 2]
        )
        weight = torch.where(
            brightness > self.brightness_threshold,
            torch.full_like(brightness, self.weight_factor),
            torch.ones_like(brightness),
        ).unsqueeze(1)
        return (self.base_loss(prediction, target) * weight).mean()


class VGGPerceptualLoss(nn.Module):
    """VGG-19 features through layer 16."""

    def __init__(
        self,
        pretrained: bool = True,
        weights_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        if weights_path is not None:
            path = Path(weights_path)
            if not path.is_file():
                raise FileNotFoundError(f"VGG19 weights not found: {path}")
            vgg = vgg19(weights=None)
            vgg.load_state_dict(
                torch.load(path, map_location="cpu", weights_only=True),
                strict=True,
            )
        else:
            weights = VGG19_Weights.IMAGENET1K_V1 if pretrained else None
            vgg = vgg19(weights=weights)
        vgg = vgg.features[:16].eval()
        for parameter in vgg.parameters():
            parameter.requires_grad = False
        self.vgg = vgg
        self.mse = nn.MSELoss()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.mse(self.vgg(prediction), self.vgg(target))


class OriginalFDPLoss(nn.Module):
    """Original image-spectrum FDPL with a fixed radial low/high mask."""

    def __init__(
        self,
        low_weight: float = 0.2,
        high_weight: float = 0.8,
        radial_cutoff: float = 0.15,
    ) -> None:
        super().__init__()
        self.low_weight = low_weight
        self.high_weight = high_weight
        self.radial_cutoff = radial_cutoff

    def _radial_mask(
        self,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        y, x = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )
        center_y = height // 2
        center_x = width // 2
        distance = torch.sqrt(
            (y - center_y).float().square()
            + (x - center_x).float().square()
        )
        max_distance = math.sqrt(center_y**2 + center_x**2)
        return (distance < max_distance * self.radial_cutoff).to(dtype)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_amplitude = torch.abs(torch.fft.fft2(prediction, dim=(-2, -1)))
        target_amplitude = torch.abs(torch.fft.fft2(target, dim=(-2, -1)))
        low_mask = self._radial_mask(
            prediction.shape[-2],
            prediction.shape[-1],
            prediction.device,
            prediction.dtype,
        )
        high_mask = 1.0 - low_mask
        low_loss = F.mse_loss(
            pred_amplitude * low_mask,
            target_amplitude * low_mask,
        )
        high_loss = F.mse_loss(
            pred_amplitude * high_mask,
            target_amplitude * high_mask,
        )
        return self.low_weight * low_loss + self.high_weight * high_loss


class PerChannelLogFDPLoss(nn.Module):
    """Residual-spectrum FDPL with fixed per-channel frequency maps."""

    def __init__(
        self,
        weight_map_64: torch.Tensor,
        weight_map_256: torch.Tensor,
        target_ratio: float = 0.30,
        rest_perceptual_weight: float = 0.01,
    ) -> None:
        super().__init__()
        if not 0.0 < target_ratio < 1.0:
            raise ValueError("target_ratio must be between 0 and 1")
        self.register_buffer("weight_map_64", weight_map_64.unsqueeze(0))
        self.register_buffer("weight_map_256", weight_map_256.unsqueeze(0))
        self.target_ratio = target_ratio
        self.rest_perceptual_weight = rest_perceptual_weight
        self.scale_factor = 1.0
        self.calibrated = False

    def _select_map(self, height: int, width: int) -> torch.Tensor:
        if height == 64 and width == 64:
            return self.weight_map_64
        if height == 256 and width == 256:
            return self.weight_map_256
        return F.interpolate(
            self.weight_map_256,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

    def calibrate_weight(
        self,
        mse_value: float,
        perceptual_value: float,
        fdpl_value: float,
        fdpl_lambda: float,
    ) -> float:
        """Set the scale that targets ``target_ratio`` at calibration time."""

        rest = mse_value + self.rest_perceptual_weight * perceptual_value
        if fdpl_value <= 0 or rest <= 0 or fdpl_lambda <= 0:
            raise ValueError("calibration values must be positive")
        self.scale_factor = (
            self.target_ratio / (1.0 - self.target_ratio)
        ) * (rest / (fdpl_lambda * fdpl_value))
        self.calibrated = True
        return self.scale_factor

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        noisy_input: torch.Tensor,
    ) -> torch.Tensor:
        predicted_noise = noisy_input - prediction
        target_noise = noisy_input - target
        predicted_fft = torch.fft.fftshift(
            torch.fft.fft2(predicted_noise, dim=(-2, -1)),
            dim=(-2, -1),
        )
        target_fft = torch.fft.fftshift(
            torch.fft.fft2(target_noise, dim=(-2, -1)),
            dim=(-2, -1),
        )
        predicted_log = torch.log1p(torch.abs(predicted_fft))
        target_log = torch.log1p(torch.abs(target_fft))
        weight_map = self._select_map(prediction.shape[-2], prediction.shape[-1])
        return ((predicted_log - target_log).square() * weight_map).mean() * (
            self.scale_factor
        )
