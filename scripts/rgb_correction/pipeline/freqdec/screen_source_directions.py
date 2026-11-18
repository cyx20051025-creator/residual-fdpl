#!/usr/bin/env python3
"""Compare source-group-disjoint RGB correction directions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SCALES = (0.0, 0.25, 0.5, 0.75, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


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


def score_direction(
    scenes: np.ndarray,
    base_mse: np.ndarray,
    offset: np.ndarray,
    target: np.ndarray,
) -> dict:
    scale_records = []
    for scale in SCALES:
        deltas = scene_delta(
            scenes, base_mse, offset, target, scale
        )
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

    def key(record: dict) -> tuple[float, ...]:
        protected_tail = (
            record["pass_fraction_scene_ge_minus_0_02"] >= 0.95
            and record["p05_scene_delta"] >= 0.0
        )
        return (
            float(protected_tail),
            record["pass_fraction_scene_ge_minus_0_02"],
            record["p05_scene_delta"],
            record["mean_scene_delta"],
            -record["scale"],
        )

    selected = max(scale_records, key=key)
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


def fixed_direction(name: str, features: np.ndarray) -> np.ndarray | None:
    noisy_mean = features[:, (0, 2, 4)]
    base_mean = features[:, (6, 8, 10)]
    residual_mean = features[:, (12, 13, 14)]
    luma_noisy = features[:, 18:19]
    luma_base = features[:, 20:21]
    if name == "noisy_mean_preservation":
        return 2.0 * (noisy_mean - base_mean)
    if name == "residual_mean":
        return 2.0 * residual_mean
    if name == "negative_residual_mean":
        return -2.0 * residual_mean
    if name == "luma_preservation":
        return 2.0 * (luma_noisy - luma_base) @ np.asarray(
            [[0.299, 0.587, 0.114]]
        )
    if name == "gray_world":
        luma = base_mean.mean(axis=1, keepdims=True)
        return 2.0 * (luma - base_mean)
    if name == "zero":
        return np.zeros_like(base_mean)
    return None


def make_model(name: str, seed: int):
    if name.startswith("ridge_"):
        alpha = float(name.removeprefix("ridge_"))
        return make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    if name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=250,
            max_depth=8,
            min_samples_leaf=20,
            max_features=0.6,
            random_state=seed,
            n_jobs=-1,
        )
    if name == "hist_gradient":
        return MultiOutputRegressor(
            HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=160,
                max_leaf_nodes=7,
                min_samples_leaf=40,
                l2_regularization=10.0,
                max_bins=128,
                random_state=seed,
            ),
            n_jobs=-1,
        )
    raise ValueError(name)


def main() -> None:
    args = parse_args()
    payload = np.load(args.features, allow_pickle=False)
    features = np.asarray(payload["features"], dtype=np.float64)
    targets = np.asarray(payload["targets"], dtype=np.float64)
    base_mse = np.asarray(payload["base_mse"], dtype=np.float64)
    scenes = np.asarray(payload["scenes"]).astype(str)

    feature_sets = {
        "means6": features[:, (0, 2, 4, 6, 8, 10)],
        "means_residual9": features[:, (0, 2, 4, 6, 8, 10, 12, 13, 14)],
        "raw12": features[:, :12],
        "extended36": features,
    }
    model_names = (
        "ridge_0.001",
        "ridge_0.01",
        "ridge_0.1",
        "ridge_1.0",
        "ridge_10.0",
        "ridge_100.0",
        "hist_gradient",
        "extra_trees",
    )
    splitter = GroupKFold(n_splits=5)
    results = {}

    for fixed_name in (
        "zero",
        "noisy_mean_preservation",
        "residual_mean",
        "negative_residual_mean",
        "luma_preservation",
        "gray_world",
    ):
        results[fixed_name] = score_direction(
            scenes,
            base_mse,
            fixed_direction(fixed_name, features),
            targets,
        )

    for feature_name, matrix in feature_sets.items():
        for model_name in model_names:
            if model_name == "extra_trees" and feature_name != "raw12":
                continue
            name = f"{model_name}__{feature_name}"
            prediction = np.full_like(targets, np.nan)
            for fold, (train_index, validation_index) in enumerate(
                splitter.split(matrix, targets, scenes)
            ):
                model = make_model(model_name, args.seed + fold)
                model.fit(matrix[train_index], targets[train_index])
                prediction[validation_index] = model.predict(
                    matrix[validation_index]
                )
            if np.isnan(prediction).any():
                raise RuntimeError(f"incomplete prediction for {name}")
            results[name] = score_direction(
                scenes, base_mse, prediction, targets
            )

    def ranking(item):
        name, record = item
        selected = record["selected_scale_record"]
        return (
            selected["pass_fraction_scene_ge_minus_0_02"] >= 0.95,
            selected["p05_scene_delta"] >= 0.0,
            selected["pass_fraction_scene_ge_minus_0_02"],
            selected["p05_scene_delta"],
            selected["mean_scene_delta"],
            record["direction_positive_fraction"],
        )

    order = sorted(results.items(), key=ranking, reverse=True)
    output = {
        "seed": args.seed,
        "features": str(args.features),
        "selected_direction": order[0][0],
        "ranking": [name for name, _ in order],
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "seed": args.seed,
                "selected_direction": order[0][0],
                "top": [
                    {
                        "name": name,
                        "direction_positive_fraction": record[
                            "direction_positive_fraction"
                        ],
                        **record["selected_scale_record"],
                    }
                    for name, record in order[:10]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
