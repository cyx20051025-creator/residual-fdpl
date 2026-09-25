"""Build fixed per-channel residual-noise frequency maps."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


def weight_map_name(
    image_size: int,
    *,
    form: str = "log_inv",
    use_power: bool = False,
    clip_quantile: float = 0.0,
    shuffle: bool = True,
    seed: int = 42,
) -> str:
    """Return the canonical map filename."""

    parts = [f"weight_map_c3_{image_size}_{form}"]
    if use_power:
        parts.append("power")
    if clip_quantile > 0:
        parts.append(f"q{clip_quantile}")
    if shuffle:
        parts.append(f"s{seed}")
    return "_".join(parts) + ".pt"


def compute_weight_map(
    dataset: Dataset,
    *,
    num_pairs: int = 15000,
    batch_size: int = 8,
    shuffle: bool = True,
    seed: int = 42,
    form: str = "log_inv",
    use_power: bool = False,
    clip_quantile: float = 0.0,
    num_workers: int = 0,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Estimate a mean-one frequency map from paired noise spectra."""

    if len(dataset) == 0:
        raise ValueError("cannot build a weight map from an empty dataset")
    if shuffle:
        torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator if shuffle else None,
        persistent_workers=num_workers > 0,
    )
    map_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    accumulated: torch.Tensor | None = None
    count = 0
    for noisy, clean in loader:
        if count >= num_pairs:
            break
        noisy = noisy.to(map_device)
        clean = clean.to(map_device)
        noise = noisy - clean
        amplitude = torch.abs(
            torch.fft.fftshift(
                torch.fft.fft2(noise, dim=(-2, -1)),
                dim=(-2, -1),
            )
        )
        if use_power:
            amplitude = amplitude.square()
        batch_sum = amplitude.sum(dim=0)
        accumulated = batch_sum if accumulated is None else accumulated + batch_sum
        count += amplitude.shape[0]

    if accumulated is None or count == 0:
        raise RuntimeError("the dataset produced no batches")
    average = accumulated / count
    average = average.cpu()
    epsilon = average[average > 0].min().item()

    if form == "inv":
        weight_map = 1.0 / (average + epsilon)
    elif form == "inv_sq":
        weight_map = 1.0 / (average + epsilon).square()
    elif form == "log_inv":
        weight_map = torch.log1p(1.0 / (average + epsilon))
    else:
        raise ValueError(f"unknown weight-map form: {form}")

    if clip_quantile > 0:
        lower = torch.quantile(weight_map, clip_quantile)
        upper = torch.quantile(weight_map, 1.0 - clip_quantile)
        weight_map = weight_map.clamp(lower, upper)
    return weight_map / weight_map.mean()


def get_or_compute_weight_map(
    dataset: Dataset,
    image_size: int,
    output_dir: str | Path,
    *,
    num_pairs: int = 15000,
    batch_size: int = 8,
    shuffle: bool = True,
    seed: int = 42,
    form: str = "log_inv",
    use_power: bool = False,
    clip_quantile: float = 0.0,
    num_workers: int = 0,
    force: bool = False,
) -> torch.Tensor:
    """Load a cached map or compute and save it under ``output_dir``."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    map_path = output_path / weight_map_name(
        image_size,
        form=form,
        use_power=use_power,
        clip_quantile=clip_quantile,
        shuffle=shuffle,
        seed=seed,
    )
    if map_path.exists() and not force:
        weight_map = torch.load(map_path, map_location="cpu", weights_only=True)
        expected_shape = (3, image_size, image_size)
        if tuple(weight_map.shape) != expected_shape:
            raise ValueError(
                f"cached weight map {map_path} has shape {tuple(weight_map.shape)}, "
                f"expected {expected_shape}"
            )
        return weight_map

    weight_map = compute_weight_map(
        dataset,
        num_pairs=num_pairs,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        form=form,
        use_power=use_power,
        clip_quantile=clip_quantile,
        num_workers=num_workers,
    )
    torch.save(weight_map.cpu(), map_path)
    return weight_map
