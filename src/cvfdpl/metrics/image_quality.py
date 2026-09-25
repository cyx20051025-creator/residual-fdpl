"""PSNR and Gaussian-window SSIM."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def calculate_psnr(
    prediction: torch.Tensor,
    target: torch.Tensor,
    max_value: float = 1.0,
) -> float:
    """Compute mean-squared-error PSNR for tensors in ``[0, max_value]``."""

    mse = F.mse_loss(prediction, target)
    if mse.item() == 0:
        return float("inf")
    maximum = torch.tensor(max_value, device=mse.device, dtype=mse.dtype)
    return float(20 * torch.log10(maximum / torch.sqrt(mse)))


def _gaussian_kernel(window_size: int, sigma: float) -> torch.Tensor:
    coordinates = torch.arange(window_size, dtype=torch.float32)
    gaussian = torch.exp(
        -(coordinates - window_size // 2).square() / (2 * sigma**2)
    )
    return gaussian / gaussian.sum()


def _create_window(window_size: int, channels: int) -> torch.Tensor:
    window_1d = _gaussian_kernel(window_size, sigma=1.5).unsqueeze(1)
    window_2d = window_1d.mm(window_1d.t()).unsqueeze(0).unsqueeze(0)
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


def calculate_ssim(
    prediction: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    data_range: float = 1.0,
) -> float:
    """Compute Gaussian-window SSIM with the v3.1 settings."""

    channels = prediction.shape[1]
    window = _create_window(window_size, channels).to(
        device=prediction.device,
        dtype=prediction.dtype,
    )
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mean_prediction = F.conv2d(
        prediction,
        window,
        padding=window_size // 2,
        groups=channels,
    )
    mean_target = F.conv2d(
        target,
        window,
        padding=window_size // 2,
        groups=channels,
    )
    mean_prediction_sq = mean_prediction.square()
    mean_target_sq = mean_target.square()
    mean_product = mean_prediction * mean_target

    variance_prediction = (
        F.conv2d(
            prediction.square(),
            window,
            padding=window_size // 2,
            groups=channels,
        )
        - mean_prediction_sq
    )
    variance_target = (
        F.conv2d(
            target.square(),
            window,
            padding=window_size // 2,
            groups=channels,
        )
        - mean_target_sq
    )
    covariance = (
        F.conv2d(
            prediction * target,
            window,
            padding=window_size // 2,
            groups=channels,
        )
        - mean_product
    )

    ssim_map = (
        (2 * mean_product + c1)
        * (2 * covariance + c2)
        / (
            (mean_prediction_sq + mean_target_sq + c1)
            * (variance_prediction + variance_target + c2)
        )
    )
    return float(ssim_map.mean().item())
