#!/usr/bin/env python3
"""Screen source-trained linear maps of the noisy-mean RGB direction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .luma_weights import luma_weights


SCALES = (0.0, 0.25, 0.50, 0.75, 1.0)
RIDGE_ALPHA = 0.1


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


def fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    kind: str,
) -> np.ndarray:
    if kind == "scalar":
        denominator = float(np.sum(train_x * train_x))
        scale = float(np.sum(train_x * train_y)) / max(denominator, 1e-12)
        return test_x * scale
    if kind == "diagonal":
        numerator = np.sum(train_x * train_y, axis=0)
        denominator = np.sum(train_x * train_x, axis=0)
        scale = numerator / np.maximum(denominator, 1e-12)
        return test_x * scale[None, :]
    center = train_x.mean(axis=0)
    scale = train_x.std(axis=0, ddof=0)
    scale = np.maximum(scale, 1e-8)
    standardized = (train_x - center) / scale
    target_mean = train_y.mean(axis=0)
    design_train = standardized
    design_test = (test_x - center) / scale
    if kind == "affine":
        design_train = np.column_stack((standardized, np.ones(len(standardized))))
        design_test = np.column_stack((design_test, np.ones(len(design_test))))
    regularizer = np.eye(design_train.shape[1]) * RIDGE_ALPHA
    if kind == "affine":
        regularizer[-1, -1] = 0.0
    coefficients = np.linalg.solve(
        design_train.T @ design_train + regularizer,
        design_train.T @ (train_y - target_mean),
    )
    return design_test @ coefficients + target_mean


def scene_folds(scenes: np.ndarray, n_splits: int = 5):
    unique_scenes = np.unique(scenes)
    scene_to_fold = {
        scene: index % n_splits
        for index, scene in enumerate(unique_scenes)
    }
    row_folds = np.asarray([scene_to_fold[scene] for scene in scenes])
    for fold in range(n_splits):
        validation = row_folds == fold
        yield np.flatnonzero(~validation), np.flatnonzero(validation)


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
    direction = 2.0 * (features[:, (0, 2, 4)] - features[:, (6, 8, 10)])
    results = {}
    for kind in ("scalar", "diagonal", "full", "affine"):
        prediction = np.full_like(targets, np.nan, dtype=np.float64)
        for train_index, validation_index in scene_folds(scenes):
            prediction[validation_index] = fit_predict(
                direction[train_index],
                targets[train_index],
                direction[validation_index],
                kind,
            )
        if np.isnan(prediction).any():
            raise RuntimeError(f"incomplete OOF predictions for {kind}")
        for subspace, values in (
            ("raw", prediction),
            ("zero_luma", project_zero_luma(prediction)),
        ):
            record = score(scenes, base_mse, values, targets)
            record["map_kind"] = kind
            record["subspace"] = subspace
            record["eligible"] = bool(
                record["selected_scale_record"][
                    "pass_fraction_scene_ge_minus_0_02"
                ]
                >= 0.95
                and record["selected_scale_record"]["p05_scene_delta"] >= 0.0
                and record["selected_scale"] > 0.0
            )
            results[f"{kind}__{subspace}"] = record

    eligible = [
        (name, record)
        for name, record in results.items()
        if record["eligible"]
    ]

    def rank(item: tuple[str, dict]) -> tuple[float, ...]:
        _, record = item
        return (
            *selection_key(record["selected_scale_record"]),
            record["direction_positive_fraction"],
        )

    selected_name, selected_record = (
        max(eligible, key=rank) if eligible else (None, None)
    )
    output = {
        "schema_version": 1,
        "seed": args.seed,
        "oof": str(args.oof),
        "ridge_alpha": RIDGE_ALPHA,
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
                "results": {
                    name: {
                        "selected_scale": record["selected_scale"],
                        **record["selected_scale_record"],
                        "direction_positive_fraction": record[
                            "direction_positive_fraction"
                        ],
                        "signed_cosine_mean": record["signed_cosine_mean"],
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
