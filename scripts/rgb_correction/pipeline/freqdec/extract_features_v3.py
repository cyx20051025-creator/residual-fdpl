#!/usr/bin/env python3
"""Extract richer source features for frozen-A RGB correction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path

import h5py
import numpy as np
import torch

from .model import FixedBandResidualNet, load_exact_state, load_initial_state
from .nafnet_adapter import load_frozen_nafnet
from .train_protected_correction import EXT_FEATURE_NAMES, extended_features


QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scene_ids(
    metadata: Path,
    source_indices: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    with metadata.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    scenes = []
    names = []
    for source_index in source_indices:
        name = Path(rows[int(source_index)]["Noisy Image:FILE"]).stem
        match = re.match(r"^(\d+)_SRGB_", name)
        if match is None:
            raise ValueError(f"cannot parse SIDD scene from {name}")
        scenes.append(match.group(1))
        names.append(name)
    return np.asarray(scenes), names


def richer_features(
    noisy: torch.Tensor,
    base: torch.Tensor,
) -> torch.Tensor:
    original = extended_features(noisy, base)
    residual = noisy - base
    extras = []
    names = list(EXT_FEATURE_NAMES)
    for image_name, image in (("noisy", noisy), ("base", base)):
        flat = image.flatten(2)
        for channel, channel_name in enumerate(("r", "g", "b")):
            for quantile in QUANTILES:
                extras.append(
                    torch.quantile(
                        flat[:, channel],
                        quantile,
                        dim=1,
                    )
                )
                names.append(
                    f"{image_name}_{channel_name}_q{int(quantile * 100):02d}"
                )
        for channel, channel_name in enumerate(("r", "g", "b")):
            extras.extend(
                (
                    (image[:, channel] < 0.0)
                    .float()
                    .mean(dim=(-2, -1)),
                    (image[:, channel] > 1.0)
                    .float()
                    .mean(dim=(-2, -1)),
                )
            )
            names.extend(
                (
                    f"{image_name}_{channel_name}_below0_fraction",
                    f"{image_name}_{channel_name}_above1_fraction",
                )
            )

    residual_flat = residual.flatten(2)
    for channel, channel_name in enumerate(("r", "g", "b")):
        for quantile in QUANTILES:
            extras.append(
                torch.quantile(
                    residual_flat[:, channel],
                    quantile,
                    dim=1,
                )
            )
            names.append(
                f"residual_{channel_name}_q{int(quantile * 100):02d}"
            )
        dx = residual[:, channel, :, 1:] - residual[:, channel, :, :-1]
        dy = residual[:, channel, 1:, :] - residual[:, channel, :-1, :]
        extras.append(
            torch.sqrt(
                dx.square().mean(dim=(1, 2))
                + dy.square().mean(dim=(1, 2))
            )
        )
        names.append(f"residual_{channel_name}_gradient_rms")

    for first, second in ((0, 1), (0, 2), (1, 2)):
        first_r = residual[:, first].flatten(1)
        second_r = residual[:, second].flatten(1)
        first_center = first_r - first_r.mean(dim=1, keepdim=True)
        second_center = second_r - second_r.mean(dim=1, keepdim=True)
        covariance = (first_center * second_center).mean(dim=1)
        extras.append(covariance)
        names.append(f"residual_cov_{first}{second}")

    return torch.cat((original, *[value[:, None] for value in extras]), dim=1), names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train-h5", type=Path, required=True)
    parser.add_argument("--source-metadata", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--architecture", default="A")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if args.architecture == "NAF":
        model = load_frozen_nafnet(args.base_checkpoint, device)
    else:
        model = FixedBandResidualNet(args.architecture).to(device)
        if args.architecture == "A":
            load_initial_state(model, args.base_checkpoint)
        else:
            load_exact_state(model, args.base_checkpoint)
        model.eval()
    feature_batches = []
    target_batches = []
    base_mse_batches = []
    feature_names = None
    with h5py.File(args.train_h5, "r") as handle:
        total = int(handle["noisy"].shape[0])
        source_indices = np.asarray(handle["source_index"][:], dtype=np.int64)
        scenes, crop_names = scene_ids(args.source_metadata, source_indices)
        for start in range(0, total, args.batch_size):
            end = min(total, start + args.batch_size)
            noisy = (
                torch.from_numpy(
                    np.asarray(handle["noisy"][start:end], dtype=np.float32)
                ).permute(0, 3, 1, 2)
                / 255.0
            ).to(device)
            clean = (
                torch.from_numpy(
                    np.asarray(handle["gt"][start:end], dtype=np.float32)
                ).permute(0, 3, 1, 2)
                / 255.0
            ).to(device)
            with torch.inference_mode():
                if args.architecture == "NAF":
                    base_01 = model(noisy)
                    base = base_01 * 2.0 - 1.0
                    target_coordinates = (clean * 2.0 - 1.0) - base
                else:
                    base = model(noisy * 2.0 - 1.0)
                    base_01 = base * 0.5 + 0.5
                    target_coordinates = (clean * 2.0 - 1.0) - base
                features, names = richer_features(noisy, base_01)
                if feature_names is None:
                    feature_names = names
                elif feature_names != names:
                    raise RuntimeError("feature-name drift")
                targets = target_coordinates.mean(dim=(-2, -1))
                base_mse = (base_01 - clean).square().mean(dim=(1, 2, 3))
            feature_batches.append(
                features.cpu().numpy().astype(np.float32)
            )
            target_batches.append(targets.cpu().numpy().astype(np.float32))
            base_mse_batches.append(
                base_mse.cpu().numpy().astype(np.float32)
            )
            if (start // args.batch_size) % 25 == 0:
                print(f"extracted {end}/{total}", flush=True)

    feature_matrix = np.concatenate(feature_batches)
    np.savez_compressed(
        args.output,
        features=feature_matrix,
        targets=np.concatenate(target_batches),
        base_mse=np.concatenate(base_mse_batches),
        source_indices=source_indices,
        scenes=scenes,
        crop_names=np.asarray(crop_names),
        feature_names=np.asarray(feature_names),
    )
    print(
        {
            "seed": args.seed,
            "architecture": args.architecture,
            "target_coordinate_scale": 2.0,
            "shape": feature_matrix.shape,
            "feature_names": len(feature_names),
            "sha256": sha256(args.output),
        }
    )


if __name__ == "__main__":
    main()
