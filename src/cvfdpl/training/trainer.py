"""Core training and validation loops for the v3.1 protocol."""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset, random_split

from cvfdpl.data import HDF5PairDataset
from cvfdpl.losses import PerChannelLogFDPLoss


@dataclass(frozen=True)
class StageResult:
    """Final validation metrics and best Stage PSNR."""

    final_psnr: float
    final_ssim: float
    best_psnr: float


def build_sidd_loaders(
    h5_path: str | Path,
    *,
    patch_size: int,
    batch_size: int,
    seed: int = 42,
    num_workers: int = 0,
    limit: int = 0,
    data_chain: str = "bilinear256",
) -> tuple[DataLoader, DataLoader]:
    """Build the SIDD loaders for the selected paper data chain."""

    if data_chain == "bilinear256":
        train_full = HDF5PairDataset(h5_path, resize_size=patch_size)
        validation_full = train_full
    elif data_chain == "crop512":
        train_full = HDF5PairDataset(
            h5_path,
            crop_size=patch_size,
            crop_mode="random",
        )
        validation_full = HDF5PairDataset(
            h5_path,
            crop_size=patch_size,
            crop_mode="center",
        )
    else:
        raise ValueError(f"unknown data chain: {data_chain}")
    if limit > 0:
        train_full = Subset(train_full, range(min(limit, len(train_full))))
        validation_full = Subset(validation_full, range(min(limit, len(validation_full))))
    if len(train_full) < 2:
        raise ValueError("at least two HDF5 pairs are required for a train/validation split")

    train_size = int(0.9 * len(train_full))
    train_size = min(max(train_size, 1), len(train_full) - 1)
    train_split, _ = random_split(
        train_full,
        [train_size, len(train_full) - train_size],
        generator=torch.Generator().manual_seed(seed),
    )
    _, validation_split = random_split(
        validation_full,
        [train_size, len(validation_full) - train_size],
        generator=torch.Generator().manual_seed(seed),
    )
    loader_kwargs = {
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True, **loader_kwargs)
    validation_loader = DataLoader(
        validation_split,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, validation_loader


def _global_ssim(prediction: torch.Tensor, target: torch.Tensor) -> float:
    c1 = 0.01**2
    c2 = 0.03**2
    mean_prediction = prediction.mean()
    mean_target = target.mean()
    variance_prediction = prediction.var(unbiased=False)
    variance_target = target.var(unbiased=False)
    covariance = ((prediction - mean_prediction) * (target - mean_target)).mean()
    score = ((2 * mean_prediction * mean_target + c1) * (2 * covariance + c2)) / (
        (mean_prediction.square() + mean_target.square() + c1)
        * (variance_prediction + variance_target + c2)
    )
    return float(score.item())


def validate_model(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[float, float]:
    """Return the legacy per-image PSNR and global SSIM used during training."""

    model.eval()
    psnr_sum = 0.0
    ssim_sum = 0.0
    count = 0
    with torch.no_grad():
        for noisy, clean in loader:
            noisy = noisy.to(device)
            clean = clean.to(device)
            prediction = model(noisy)
            prediction_01 = (prediction * 0.5 + 0.5).clamp(0, 1)
            clean_01 = (clean * 0.5 + 0.5).clamp(0, 1)
            for index in range(prediction.shape[0]):
                prediction_image = prediction_01[index : index + 1]
                target_image = clean_01[index : index + 1]
                mse = F.mse_loss(prediction_image, target_image)
                psnr = (
                    20 * math.log10(1.0 / math.sqrt(mse.item()))
                    if mse.item() > 0
                    else 100.0
                )
                psnr_sum += psnr
                ssim_sum += _global_ssim(prediction_image, target_image)
                count += 1
    model.train()
    if count == 0:
        raise RuntimeError("validation loader returned no samples")
    return psnr_sum / count, ssim_sum / count


def estimate_loss_terms(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    mse_fn: nn.Module,
    perceptual_fn: nn.Module,
    fdpl_fn: PerChannelLogFDPLoss | None,
    device: torch.device,
    batches: int,
) -> tuple[float, float, float]:
    """Estimate calibration terms on a bounded number of training batches."""

    model.eval()
    mse_sum = 0.0
    perceptual_sum = 0.0
    fdpl_sum = 0.0
    count = 0
    with torch.no_grad():
        for noisy, clean in itertools.islice(loader, batches):
            noisy = noisy.to(device)
            clean = clean.to(device)
            prediction = model(noisy)
            mse_sum += float(mse_fn(prediction, clean).item())
            prediction_01 = (prediction * 0.5 + 0.5).clamp(0, 1)
            clean_01 = (clean * 0.5 + 0.5).clamp(0, 1)
            perceptual_sum += float(perceptual_fn(prediction_01, clean_01).item())
            if fdpl_fn is not None:
                fdpl_sum += float(fdpl_fn(prediction, clean, noisy).item()) / max(
                    fdpl_fn.scale_factor,
                    1e-8,
                )
            count += 1
    model.train()
    if count == 0:
        raise RuntimeError("calibration loader returned no batches")
    return mse_sum / count, perceptual_sum / count, fdpl_sum / count


def train_stage(
    model: nn.Module,
    ema: nn.Module | object,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    *,
    mse_fn: nn.Module,
    perceptual_fn: nn.Module,
    fdpl_fn: PerChannelLogFDPLoss | None,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    warmup_epochs: int,
    fdpl_weight: float,
    perceptual_weight: float,
    stage_name: str,
    stage_id: int,
    output_path: str | Path,
    best_path: str | Path,
    weight_decay: float = 1e-4,
    calibration_batches: int = 32,
    recalibrate_every: int = 0,
    history: list[dict[str, object]] | None = None,
) -> StageResult:
    """Run one 64 x 64 or 256 x 256 fine-tuning stage."""

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if warmup_epochs <= 0:
        raise ValueError("warmup_epochs must be positive")
    mse_value, perceptual_value, fdpl_value = estimate_loss_terms(
        model,
        train_loader,
        mse_fn=mse_fn,
        perceptual_fn=perceptual_fn,
        fdpl_fn=fdpl_fn,
        device=device,
        batches=calibration_batches,
    )
    if fdpl_fn is not None:
        fdpl_fn.calibrate_weight(
            mse_value,
            perceptual_value,
            fdpl_value,
            fdpl_lambda=fdpl_weight,
        )

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    best_psnr = -1.0
    final_psnr = 0.0
    final_ssim = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_mse = 0.0
        total_perceptual = 0.0
        total_fdpl = 0.0
        schedule_weight = min(
            fdpl_weight,
            fdpl_weight * epoch / warmup_epochs,
        )

        for noisy, clean in train_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)
            prediction = model(noisy)
            loss_mse = mse_fn(prediction, clean)
            prediction_01 = (prediction * 0.5 + 0.5).clamp(0, 1)
            clean_01 = (clean * 0.5 + 0.5).clamp(0, 1)
            loss_perceptual = perceptual_fn(prediction_01, clean_01)
            loss_fdpl = (
                fdpl_fn(prediction, clean, noisy) if fdpl_fn is not None else None
            )
            loss = loss_mse + perceptual_weight * loss_perceptual
            if loss_fdpl is not None:
                loss = loss + schedule_weight * loss_fdpl

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            update = getattr(ema, "update", None)
            if update is None:
                raise TypeError("ema must provide an update(model) method")
            update(model)

            total_loss += float(loss.item())
            total_mse += float(loss_mse.item())
            total_perceptual += float(loss_perceptual.item())
            total_fdpl += 0.0 if loss_fdpl is None else float(loss_fdpl.item())

        scheduler.step()
        final_psnr, final_ssim = validate_model(model, validation_loader, device)
        batch_count = len(train_loader)
        if history is not None:
            history.append(
                {
                    "stage": stage_id,
                    "epoch": epoch,
                    "loss": round(total_loss / batch_count, 6),
                    "mse": round(total_mse / batch_count, 6),
                    "perc": round(total_perceptual / batch_count, 6),
                    "fdpl_raw": round(total_fdpl / batch_count, 6),
                    "lam": round(schedule_weight, 6),
                    "scale_factor": (
                        None if fdpl_fn is None else round(fdpl_fn.scale_factor, 6)
                    ),
                    "val_psnr": round(final_psnr, 6),
                    "val_ssim": round(final_ssim, 6),
                }
            )

        if final_psnr > best_psnr:
            best_psnr = final_psnr
            best_file = Path(best_path)
            best_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), best_file)

        if (
            fdpl_fn is not None
            and recalibrate_every > 0
            and epoch >= warmup_epochs
            and epoch % recalibrate_every == 0
        ):
            mse_value, perceptual_value, fdpl_value = estimate_loss_terms(
                model,
                train_loader,
                mse_fn=mse_fn,
                perceptual_fn=perceptual_fn,
                fdpl_fn=fdpl_fn,
                device=device,
                batches=calibration_batches,
            )
            fdpl_fn.calibrate_weight(
                mse_value,
                perceptual_value,
                fdpl_value,
                fdpl_lambda=schedule_weight,
            )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_file)
    return StageResult(
        final_psnr=final_psnr,
        final_ssim=final_ssim,
        best_psnr=best_psnr,
    )
