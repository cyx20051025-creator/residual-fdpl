#!/usr/bin/env python3
"""Screen a source-support gate for the frozen noisy-mean direction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .luma_weights import luma_weights


QUANTILES = (0.90, 0.95)
SCALES = (0.0, 0.25, 0.50, 0.75, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def project_zero_luma(values: np.ndarray) -> np.ndarray:
    weight = luma_weights()
    luma = values @ weight
    return values - luma[:, None] * weight[None, :] / float(weight @ weight)


def robust_support_score(features: np.ndarray) -> np.ndarray:
    center = np.median(features, axis=0)
    absolute_deviation = np.abs(features - center)
    mad = 1.4826 * np.median(absolute_deviation, axis=0)
    q25, q75 = np.quantile(features, (0.25, 0.75), axis=0)
    iqr_scale = (q75 - q25) / 1.349
    scale = np.maximum.reduce(
        (mad, iqr_scale, np.full(features.shape[1], 1e-8))
    )
    standardized = (features - center) / scale
    return np.sqrt(np.mean(standardized * standardized, axis=1))


def scene_delta(
    scenes: np.ndarray,
    base_mse: np.ndarray,
    offset: np.ndarray,
    target: np.ndarray,
    scale: float,
) -> np.ndarray:
    applied = scale * offset
    dot = np.einsum("ij,ij->i", applied, target)
    norm_sq = np.einsum("ij,ij->i", applied, applied)
    candidate_mse = np.maximum(
        base_mse - dot / 6.0 + norm_sq / 12.0,
        1e-12,
    )
    delta = -10.0 * np.log10(candidate_mse) + 10.0 * np.log10(base_mse)
    return np.asarray(
        [delta[scenes == scene].mean() for scene in np.unique(scenes)]
    )


def selection_key(record: dict[str, float | int]) -> tuple[float, ...]:
    protected_tail = (
        record["pass_fraction_scene_ge_minus_0_02"] >= 0.95
        and record["p05_scene_delta"] >= 0.0
    )
    return (
        float(protected_tail),
        float(record["p05_scene_delta"]),
        float(record["mean_scene_delta"]),
        float(record["pass_fraction_scene_ge_minus_0_02"]),
        -float(record["scale"]),
    )


def score(
    scenes: np.ndarray,
    base_mse: np.ndarray,
    offset: np.ndarray,
    target: np.ndarray,
) -> dict:
    scale_records = []
    for scale in SCALES:
        deltas = scene_delta(scenes, base_mse, offset, target, scale)
        scale_records.append(
            {
                "scale": scale,
                "pass_fraction_scene_ge_minus_0_02": float(
                    (deltas >= -0.02).mean()
                ),
                "worst_scene_delta": float(deltas.min()),
                "p05_scene_delta": float(np.quantile(deltas, 0.05)),
                "mean_scene_delta": float(deltas.mean()),
                "median_scene_delta": float(np.median(deltas)),
            }
        )
    selected = max(scale_records, key=selection_key)
    dot = np.einsum("ij,ij->i", offset, target)
    norm_sq = np.einsum("ij,ij->i", offset, offset)
    target_sq = np.einsum("ij,ij->i", target, target)
    cosine = np.divide(
        dot,
        np.sqrt(norm_sq * target_sq),
        out=np.zeros_like(dot),
        where=(norm_sq > 0.0) & (target_sq > 0.0),
    )
    return {
        "selected_scale": selected["scale"],
        "selected_scale_record": selected,
        "scale_records": scale_records,
        "direction_positive_fraction": float((dot > 0.0).mean()),
        "signed_cosine_mean": float(cosine.mean()),
        "signed_cosine_median": float(np.median(cosine)),
        "signed_cosine_p05": float(np.quantile(cosine, 0.05)),
    }


def main() -> None:
    args = parse_args()
    payload = np.load(args.oof, allow_pickle=False)
    features = np.asarray(payload["features"], dtype=np.float64)
    targets = np.asarray(payload["targets"], dtype=np.float64)
    base_mse = np.asarray(payload["base_mse"], dtype=np.float64)
    scenes = np.asarray(payload["scenes"]).astype(str)
    raw_direction = 2.0 * (features[:, (0, 2, 4)] - features[:, (6, 8, 10)])
    direction = project_zero_luma(raw_direction)

    support_sets = {
        "raw12": features[:, :12],
        "extended36": features[:, :36],
    }
    results = {}
    thresholds_by_feature_set = {}
    for feature_name, matrix in support_sets.items():
        support = robust_support_score(matrix)
        thresholds = {
            f"q{int(quantile * 100):03d}": float(
                np.quantile(support, quantile)
            )
            for quantile in QUANTILES
        }
        thresholds_by_feature_set[feature_name] = thresholds
        for threshold_name, threshold in thresholds.items():
            active = support >= threshold
            offset = direction * active[:, None]
            record = score(scenes, base_mse, offset, targets)
            record["feature_set"] = feature_name
            record["threshold_name"] = threshold_name
            record["threshold"] = threshold
            record["retained_fraction"] = float(active.mean())
            record["active_direction_positive_fraction"] = float(
                (
                    np.einsum(
                        "ij,ij->i",
                        direction[active],
                        targets[active],
                    )
                    > 0.0
                ).mean()
            )
            record["eligible"] = bool(
                record["selected_scale_record"][
                    "pass_fraction_scene_ge_minus_0_02"
                ]
                >= 0.95
                and record["selected_scale_record"]["p05_scene_delta"] >= 0.0
                and record["selected_scale"] > 0.0
            )
            results[f"{feature_name}__{threshold_name}"] = record

    eligible = [
        (name, record)
        for name, record in results.items()
        if record["eligible"]
    ]

    def rank(item: tuple[str, dict]) -> tuple[float, ...]:
        _, record = item
        return (
            *selection_key(record["selected_scale_record"]),
            record["retained_fraction"],
        )

    selected_name, selected_record = (
        max(eligible, key=rank) if eligible else (None, None)
    )
    output = {
        "schema_version": 1,
        "seed": args.seed,
        "oof": str(args.oof),
        "direction": "rec601_zero_luma_noisy_mean",
        "thresholds": thresholds_by_feature_set,
        "source_rows": int(len(features)),
        "source_scenes": int(np.unique(scenes).size),
        "selected_candidate": selected_name,
        "selected_candidate_record": selected_record,
        "results": results,
        "source_only": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "seed": args.seed,
                "selected_candidate": selected_name,
                "thresholds": thresholds_by_feature_set,
                "results": {
                    name: {
                        "feature_set": record["feature_set"],
                        "threshold": record["threshold"],
                        "retained_fraction": record["retained_fraction"],
                        "active_positive_fraction": (
                            record["active_direction_positive_fraction"]
                        ),
                        "selected_scale": record["selected_scale"],
                        **record["selected_scale_record"],
                        "eligible": record["eligible"],
                    }
                    for name, record in results.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
