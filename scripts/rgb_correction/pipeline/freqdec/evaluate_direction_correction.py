#!/usr/bin/env python3
"""Evaluate a source-frozen RGB-direction candidate on the fixed manifest."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import h5py
import numpy as np
import torch

from .data import file_sha256
from .evaluate import component_diagnostics, ssim_per_image
from .extract_features_v3 import richer_features
from .luma_weights import luma_weights
from .model import FixedBandResidualNet, load_exact_state, load_initial_state
from .nafnet_adapter import load_frozen_nafnet
from .screen_direction_ensembles import (
    capped_candidates,
    ensemble_candidates,
    subspace_candidates,
)
from .screen_direction_targets import invert_target
from .screen_source_gate import gate_matrix
from .train_protected_correction import (
    EXT_FEATURE_NAMES,
    RAW_FEATURE_NAMES,
    extended_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-artifact", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--architecture", default="A")
    parser.add_argument("--population", action="append")
    parser.add_argument(
        "--shuffle-offsets-within-batch",
        action="store_true",
        help="Post-hoc control: permute predicted offsets among rows in each batch.",
    )
    parser.add_argument("--shuffle-offset-seed", type=int, default=20261115)
    return parser.parse_args()


def metric_values(
    prediction: torch.Tensor,
    clean: torch.Tensor,
    lpips_model: object,
    dists_model: object,
    low_pass: object,
) -> dict[str, torch.Tensor]:
    mse = (prediction - clean).square().mean(dim=(1, 2, 3))
    return {
        "psnr": -10.0 * torch.log10(mse),
        "ssim": ssim_per_image(prediction, clean),
        "lpips": lpips_model(
            prediction * 2.0 - 1.0,
            clean * 2.0 - 1.0,
        ).flatten(),
        "dists": dists_model(prediction, clean).flatten(),
        **component_diagnostics(prediction, clean, low_pass),
    }


def add_metric_records(
    records: list[dict],
    metrics: dict[str, torch.Tensor],
    prefix: str,
) -> None:
    for index, record in enumerate(records):
        for key in ("psnr", "ssim", "lpips", "dists"):
            record[f"{prefix}_{key}"] = float(metrics[key][index])
        record[f"{prefix}_radial_band_mse"] = [
            float(value[index]) for value in metrics["radial_band_mse"].T
        ]


def grouped_summary(
    records: list[dict],
    primary_unit: str,
    prefix: str,
) -> dict[str, object]:
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(str(record[primary_unit]), []).append(record)
    metric_names = ("psnr", "ssim", "lpips", "dists")
    units = []
    for unit, rows in sorted(grouped.items()):
        summary = {"unit": unit, "count": len(rows)}
        for metric in metric_names:
            summary[metric] = float(
                np.mean([float(row[f"{prefix}_{metric}"]) for row in rows])
            )
        summary["preclip_mse"] = float(
            np.mean([float(row[f"{prefix}_preclip_mse"]) for row in rows])
        )
        summary["radial_band_mse"] = np.mean(
            [row[f"{prefix}_radial_band_mse"] for row in rows],
            axis=0,
        ).tolist()
        units.append(summary)
    metric_means = {
        metric: float(np.mean([unit[metric] for unit in units]))
        for metric in (*metric_names, "preclip_mse")
    }
    metric_means["radial_band_mse"] = np.mean(
        [unit["radial_band_mse"] for unit in units], axis=0
    ).tolist()
    return {
        "primary_unit": primary_unit,
        "primary_unit_count": len(units),
        "row_count": len(records),
        "metric_means": metric_means,
        "units": units,
    }


def main() -> None:
    args = parse_args()
    import lpips
    import piq

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

    artifact = pickle.loads(args.model_artifact.read_bytes())
    schema_version = artifact.get("schema_version")
    if schema_version not in {2, 3, 4}:
        raise ValueError(
            "expected direction-candidate schema version 2, 3, or 4"
        )
    if tuple(artifact["raw_feature_names"]) != RAW_FEATURE_NAMES:
        raise ValueError("raw feature schema mismatch")
    extended_feature_names = tuple(artifact["extended_feature_names"])
    if extended_feature_names == EXT_FEATURE_NAMES:
        feature_schema = str(
            artifact.get("feature_schema", "v1_extended36")
        )
    elif len(extended_feature_names) == 117:
        feature_schema = str(
            artifact.get("feature_schema", "v3_extended117")
        )
    else:
        raise ValueError(
            f"unsupported extended feature schema with "
            f"{len(extended_feature_names)} names"
        )
    if feature_schema not in {"v1_extended36", "v3_extended117"}:
        raise ValueError(f"unsupported feature schema: {feature_schema}")
    direction_model = artifact.get("direction_model")
    target_transform = str(
        artifact.get(
            "direction_target_transform",
            artifact.get("selected_ensemble_rule", ""),
        )
    )
    feature_family = str(
        artifact.get("direction_feature_family", "ensemble")
    )
    ensemble_members = artifact.get("ensemble_members")
    correction_subspace = str(
        artifact.get("correction_subspace", "raw")
    )
    gate_model = artifact.get("gate_model")
    calibration = float(
        artifact.get(
            "source_gate_scale",
            artifact.get("source_selected_scale"),
        )
    )

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected_populations = set(args.population or [])
    lpips_model = lpips.LPIPS(net="alex", pretrained=True, verbose=False).to(device)
    lpips_model.eval()
    dists_model = piq.DISTS(reduction="none").to(device).eval()
    outputs = []
    for population_index, population in enumerate(manifest["populations"]):
        if selected_populations and population["id"] not in selected_populations:
            continue
        data_path = args.data_root / population["filename"]
        if file_sha256(data_path) != population["sha256"]:
            raise ValueError(f"hash mismatch for {population['id']}")
        records = []
        shuffle_rng = np.random.default_rng(
            args.shuffle_offset_seed + population_index
        )
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
                    if args.architecture == "NAF":
                        base_01 = model(noisy)
                        base = base_01 * 2.0 - 1.0
                    else:
                        base = model(noisy * 2.0 - 1.0)
                        base_01 = base * 0.5 + 0.5
                    if feature_schema == "v3_extended117":
                        feature_tensor, extracted_names = richer_features(
                            noisy,
                            base_01,
                        )
                        if tuple(extracted_names) != extended_feature_names:
                            raise RuntimeError("v3 feature-name drift")
                    else:
                        feature_tensor = extended_features(noisy, base_01)
                    features = feature_tensor.cpu().numpy()
                    matrix = (
                        features[:, : len(RAW_FEATURE_NAMES)]
                        if feature_family == "raw12"
                        else features
                    )
                    if schema_version == 2:
                        predicted_transformed = direction_model.predict(matrix)
                        direction = invert_target(
                            target_transform,
                            np.asarray(
                                predicted_transformed, dtype=np.float64
                            ),
                        ).astype(np.float32)
                        alpha = np.full(
                            (end - start,),
                            calibration,
                            dtype=np.float32,
                        )
                    else:
                        member_predictions = {}
                        for member in ensemble_members:
                            member_matrix = (
                                features[:, : len(RAW_FEATURE_NAMES)]
                                if member["feature_family"] == "raw12"
                                else features
                            )
                            prediction = member["model"].predict(member_matrix)
                            member_predictions[member["name"]] = invert_target(
                                member["target_transform"],
                                np.asarray(prediction, dtype=np.float64),
                            )
                        candidates = subspace_candidates(
                            capped_candidates(
                                {
                                    **member_predictions,
                                    **ensemble_candidates(member_predictions),
                                }
                            )
                        )
                        direction64 = candidates[target_transform]
                        direction = np.asarray(
                            direction64, dtype=np.float32
                        )
                        if schema_version == 4:
                            gate_x, _ = gate_matrix(
                                features,
                                member_predictions,
                                direction64,
                            )
                            alpha = (
                                calibration
                                * np.clip(
                                    gate_model.predict(gate_x),
                                    0.0,
                                    1.0,
                                )
                            ).astype(np.float32)
                        else:
                            alpha = np.full(
                                (end - start,),
                                calibration,
                                dtype=np.float32,
                            )
                    if correction_subspace == "chroma":
                        luma = luma_weights().astype(np.float32)
                        luma_norm_sq = float(luma @ luma)
                        luma_component = (direction * luma).sum(
                            axis=1, keepdims=True
                        )
                        direction = (
                            direction
                            - luma_component * luma / luma_norm_sq
                        )
                    elif correction_subspace != "raw":
                        raise ValueError(
                            f"unknown correction subspace {correction_subspace}"
                        )
                    if args.shuffle_offsets_within_batch and end - start > 1:
                        permutation = shuffle_rng.permutation(end - start)
                        direction = direction[permutation]
                        alpha = alpha[permutation]
                    b_tensor = (
                        torch.from_numpy(direction)
                        .to(device)
                        .view(-1, 3, 1, 1)
                    )
                    alpha_tensor = (
                        torch.from_numpy(alpha)
                        .to(device)
                        .view(-1, 1, 1, 1)
                    )
                    corrected = base + alpha_tensor * b_tensor
                    corrected_unclipped = corrected * 0.5 + 0.5
                    base_prediction = base_01.clamp(0.0, 1.0)
                    corrected_prediction = corrected_unclipped.clamp(0.0, 1.0)
                    base_metrics = metric_values(
                        base_prediction,
                        clean,
                        lpips_model,
                        dists_model,
                        model.low_pass,
                    )
                    candidate_metrics = metric_values(
                        corrected_prediction,
                        clean,
                        lpips_model,
                        dists_model,
                        model.low_pass,
                    )
                    base_preclip_mse = (
                        base_01 - clean
                    ).square().mean(dim=(1, 2, 3))
                    candidate_preclip_mse = (
                        corrected_unclipped - clean
                    ).square().mean(dim=(1, 2, 3))
                    candidate_clip_fraction = (
                        (corrected_unclipped < 0.0)
                        | (corrected_unclipped > 1.0)
                    ).float().mean(dim=(1, 2, 3))

                for offset in range(end - start):
                    entry = population["entries"][start + offset]
                    record = {
                        "row": start + offset,
                        "base_preclip_mse": float(base_preclip_mse[offset]),
                        "candidate_preclip_mse": float(
                            candidate_preclip_mse[offset]
                        ),
                        "candidate_clip_fraction": float(
                            candidate_clip_fraction[offset]
                        ),
                        "ridge_offset_rgb": direction[offset].tolist(),
                        "alpha": float(alpha[offset]),
                        "effective_offset_rgb": (
                            alpha[offset] * direction[offset]
                        ).tolist(),
                    }
                    for key, value in entry.items():
                        if key != "row":
                            record[key] = value
                    records.append(record)
                add_metric_records(records[-len(direction) :], base_metrics, "base")
                add_metric_records(
                    records[-len(direction) :], candidate_metrics, "candidate"
                )
                print(f"{population['id']} {end}/{total}", flush=True)

        alpha_values = np.asarray([record["alpha"] for record in records])
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
                    "mean": float(alpha_values.mean()),
                    "zero_fraction": float((alpha_values == 0.0).mean()),
                    "one_fraction": float((alpha_values == 1.0).mean()),
                    "quantiles": {
                        "min": float(alpha_values.min()),
                        "p05": float(np.quantile(alpha_values, 0.05)),
                        "p25": float(np.quantile(alpha_values, 0.25)),
                        "p50": float(np.median(alpha_values)),
                        "p75": float(np.quantile(alpha_values, 0.75)),
                        "p95": float(np.quantile(alpha_values, 0.95)),
                        "max": float(alpha_values.max()),
                    },
                },
                "records": records,
            }
        )

    selected_candidate = artifact.get(
        "direction_model_name",
        artifact.get("selected_ensemble_rule"),
    )
    if schema_version == 4:
        selected_candidate = (
            f"{artifact['selected_ensemble_rule']}"
            f"+{artifact['gate_model_name']}"
        )
    output = {
        "schema_version": 2,
        "architecture": args.architecture,
        "model_artifact": str(args.model_artifact),
        "model_artifact_sha256": file_sha256(args.model_artifact),
        "selected_candidate": selected_candidate,
        "direction_target_transform": target_transform,
        "direction_feature_family": feature_family,
        "correction_subspace": correction_subspace,
        "global_calibration": calibration,
        "base_checkpoint": str(args.base_checkpoint),
        "base_checkpoint_sha256": file_sha256(args.base_checkpoint),
        "manifest": str(args.manifest),
        "manifest_sha256": file_sha256(args.manifest),
        "shuffle_offsets_within_batch": bool(
            args.shuffle_offsets_within_batch
        ),
        "shuffle_offset_seed": int(args.shuffle_offset_seed),
        "shuffle_unit": (
            "within_batch"
            if args.shuffle_offsets_within_batch
            else None
        ),
        "populations": outputs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                population["population"]: population["alpha_stats"]
                for population in outputs
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
