#!/usr/bin/env python3
"""Evaluate simple source-constant or noisy-mean zero-luma baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from .data import file_sha256
from .evaluate_direction_correction import (
    add_metric_records,
    grouped_summary,
    metric_values,
)
from .luma_weights import luma_weights
from .model import FixedBandResidualNet, load_exact_state, load_initial_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=(
            "source_constant",
            "noisy_mean_match",
            "noisy_mean_deadzone",
        ),
    )
    parser.add_argument("--architecture", choices=("A", "C", "E"), default="A")
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-features", type=Path)
    parser.add_argument("--deadzone-threshold", type=float)
    parser.add_argument("--deadzone-scale", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def project_zero_luma(values: np.ndarray) -> np.ndarray:
    weight = luma_weights()
    luma = values @ weight
    return values - luma[:, None] * weight[None, :] / float(weight @ weight)


def main() -> None:
    args = parse_args()
    import lpips
    import piq

    device = torch.device(args.device)
    model = FixedBandResidualNet(args.architecture).to(device)
    if args.architecture == "A":
        load_initial_state(model, args.base_checkpoint)
    else:
        load_exact_state(model, args.base_checkpoint)
    model.eval()

    source_constant = None
    if args.mode == "source_constant":
        if args.source_features is None:
            raise ValueError("--source-features is required for source_constant")
        payload = np.load(args.source_features, allow_pickle=False)
        source_targets = np.asarray(payload["targets"], dtype=np.float64)
        source_constant = project_zero_luma(
            source_targets.mean(axis=0, keepdims=True)
        )[0].astype(np.float32)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    lpips_model = lpips.LPIPS(net="alex", pretrained=True, verbose=False).to(device)
    lpips_model.eval()
    dists_model = piq.DISTS(reduction="none").to(device).eval()
    outputs = []

    for population in manifest["populations"]:
        data_path = args.data_root / population["filename"]
        if file_sha256(data_path) != population["sha256"]:
            raise ValueError(f"hash mismatch for {population['id']}")
        records = []
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
                    base_coord = model(noisy * 2.0 - 1.0)
                    base_01 = base_coord * 0.5 + 0.5
                    if source_constant is None:
                        base_mean = base_coord.mean(dim=(-2, -1)).cpu().numpy()
                        noisy_mean = noisy.mean(dim=(-2, -1)).cpu().numpy()
                        offset = project_zero_luma(
                            2.0 * noisy_mean - 1.0 - base_mean
                        ).astype(np.float32)
                        if args.mode == "noisy_mean_deadzone":
                            if args.deadzone_threshold is None:
                                raise ValueError(
                                    "--deadzone-threshold is required"
                                )
                            active = (
                                np.linalg.norm(offset, axis=1)
                                >= args.deadzone_threshold
                            )
                            offset = (
                                offset
                                * active[:, None]
                                * args.deadzone_scale
                            ).astype(np.float32)
                            alpha_values = active.astype(np.float32) * float(
                                args.deadzone_scale
                            )
                        else:
                            alpha_values = np.ones(
                                end - start,
                                dtype=np.float32,
                            )
                    else:
                        offset = np.broadcast_to(
                            source_constant,
                            (end - start, 3),
                        ).copy()
                        alpha_values = np.ones(
                            end - start,
                            dtype=np.float32,
                        )
                    offset_tensor = (
                        torch.from_numpy(offset)
                        .to(device)
                        .view(-1, 3, 1, 1)
                    )
                    corrected_coord = base_coord + offset_tensor
                    corrected_01 = (corrected_coord * 0.5 + 0.5).clamp(0.0, 1.0)
                    base_prediction = base_01.clamp(0.0, 1.0)
                    base_metrics = metric_values(
                        base_prediction,
                        clean,
                        lpips_model,
                        dists_model,
                        model.low_pass,
                    )
                    candidate_metrics = metric_values(
                        corrected_01,
                        clean,
                        lpips_model,
                        dists_model,
                        model.low_pass,
                    )
                    base_preclip_mse = (
                        base_01 - clean
                    ).square().mean(dim=(1, 2, 3))
                    candidate_preclip_mse = (
                        corrected_coord * 0.5 + 0.5 - clean
                    ).square().mean(dim=(1, 2, 3))
                    clip_fraction = (
                        (corrected_coord * 0.5 + 0.5 < 0.0)
                        | (corrected_coord * 0.5 + 0.5 > 1.0)
                    ).float().mean(dim=(1, 2, 3))

                for index in range(end - start):
                    entry = population["entries"][start + index]
                    record = {
                        "row": start + index,
                        "base_preclip_mse": float(base_preclip_mse[index]),
                        "candidate_preclip_mse": float(
                            candidate_preclip_mse[index]
                        ),
                        "candidate_clip_fraction": float(clip_fraction[index]),
                        "ridge_offset_rgb": offset[index].tolist(),
                            "alpha": float(alpha_values[index]),
                        "effective_offset_rgb": offset[index].tolist(),
                    }
                    for key, value in entry.items():
                        if key != "row":
                            record[key] = value
                    records.append(record)
                add_metric_records(records[-len(offset) :], base_metrics, "base")
                add_metric_records(
                    records[-len(offset) :],
                    candidate_metrics,
                    "candidate",
                )
                print(f"{population['id']} {end}/{total}", flush=True)

        outputs.append(
            {
                "population": population["id"],
                "role": population["role"],
                "base_summary": grouped_summary(
                    records, population["primary_unit"], "base"
                ),
                "candidate_summary": grouped_summary(
                    records, population["primary_unit"], "candidate"
                ),
                "alpha_stats": {
                    "mean": float(
                        np.mean([float(row["alpha"]) for row in records])
                    ),
                    "zero_fraction": float(
                        np.mean([float(row["alpha"]) == 0.0 for row in records])
                    ),
                    "one_fraction": float(
                        np.mean([float(row["alpha"]) == 1.0 for row in records])
                    ),
                    "quantiles": {
                        key: float(np.quantile(
                            [float(row["alpha"]) for row in records],
                            quantile,
                        ))
                        for key, quantile in (
                            ("min", 0.0),
                            ("p05", 0.05),
                            ("p25", 0.25),
                            ("p50", 0.50),
                            ("p75", 0.75),
                            ("p95", 0.95),
                            ("max", 1.0),
                        )
                    },
                },
                "records": records,
            }
        )

    payload = {
        "schema_version": 2,
        "architecture": args.architecture,
        "selected_candidate": args.mode,
        "direction_target_transform": args.mode,
        "direction_feature_family": args.mode,
        "correction_subspace": (
            "rec601_zero_luma"
            if args.mode != "source_constant"
            else "rec601_zero_luma_constant"
        ),
        "global_calibration": (
            float(args.deadzone_scale)
            if args.mode == "noisy_mean_deadzone"
            else 1.0
        ),
        "deadzone_threshold": args.deadzone_threshold,
        "source_constant_zero_luma": (
            None if source_constant is None else source_constant.tolist()
        ),
        "base_checkpoint": str(args.base_checkpoint),
        "base_checkpoint_sha256": file_sha256(args.base_checkpoint),
        "manifest": str(args.manifest),
        "manifest_sha256": file_sha256(args.manifest),
        "populations": outputs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                row["population"]: row["candidate_summary"]["metric_means"][
                    "psnr"
                ]
                - row["base_summary"]["metric_means"]["psnr"]
                for row in outputs
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
