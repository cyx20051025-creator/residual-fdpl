#!/usr/bin/env python3
"""Extract frozen-A correction features and ensemble predictions on eval sets."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import h5py
import numpy as np
import torch

from .data import file_sha256
from .model import FixedBandResidualNet, load_initial_state
from .screen_direction_targets import invert_target
from .train_protected_correction import (
    RAW_FEATURE_NAMES,
    extended_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-artifact", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    artifact = pickle.loads(args.model_artifact.read_bytes())
    if artifact.get("schema_version") != 3:
        raise ValueError("expected ensemble schema version 3")
    model = FixedBandResidualNet("A").to(device)
    load_initial_state(model, args.base_checkpoint)
    model.eval()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for population in manifest["populations"]:
        data_path = args.data_root / population["filename"]
        if file_sha256(data_path) != population["sha256"]:
            raise ValueError(f"hash mismatch for {population['id']}")
        feature_batches = []
        target_batches = []
        base_mse_batches = []
        member_batches = {
            member["name"]: [] for member in artifact["ensemble_members"]
        }
        row_ids = []
        with h5py.File(data_path, "r") as handle:
            total = int(handle["noisy"].shape[0])
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
                    base = model(noisy * 2.0 - 1.0)
                    base_01 = base * 0.5 + 0.5
                    features = extended_features(
                        noisy, base_01
                    ).cpu().numpy()
                    target = (
                        (clean * 2.0 - 1.0) - base
                    ).mean(dim=(-2, -1)).cpu().numpy()
                    base_mse = (
                        base_01 - clean
                    ).square().mean(dim=(1, 2, 3)).cpu().numpy()
                feature_batches.append(features)
                target_batches.append(target)
                base_mse_batches.append(base_mse)
                row_ids.extend(range(start, end))
                for member in artifact["ensemble_members"]:
                    matrix = (
                        features[:, : len(RAW_FEATURE_NAMES)]
                        if member["feature_family"] == "raw12"
                        else features
                    )
                    prediction = member["model"].predict(matrix)
                    member_batches[member["name"]].append(
                        invert_target(
                            member["target_transform"],
                            np.asarray(prediction, dtype=np.float64),
                        )
                    )
                print(
                    f"{population['id']} {end}/{total}",
                    flush=True,
                )
        arrays = {
            "features": np.concatenate(feature_batches),
            "targets": np.concatenate(target_batches),
            "base_mse": np.concatenate(base_mse_batches),
            "rows": np.asarray(row_ids),
        }
        arrays.update(
            {
                f"prediction__{name}": np.concatenate(values)
                for name, values in member_batches.items()
            }
        )
        np.savez_compressed(
            args.output_dir / f"{population['id']}.npz",
            **arrays,
        )


if __name__ == "__main__":
    main()
