#!/usr/bin/env python3
"""Screen source-only RGB correction directions and target parameterizations.

The screen uses scene-disjoint GroupKFold predictions and the same protection
objective as the main protocol. It deliberately has no external-data inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .luma_weights import luma_weights


SCALES = (0.0, 0.25, 0.5, 0.75, 1.0)
LUMA = luma_weights()


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
    ideal_alpha = np.clip(
        dot / np.maximum(norm_sq, 1e-12),
        0.0,
        1.0,
    )
    return {
        "selected_scale": selected["scale"],
        "selected_scale_record": selected,
        "scale_records": scale_records,
        "direction_positive_fraction": float((dot > 0.0).mean()),
        "signed_cosine_mean": float(cosine.mean()),
        "signed_cosine_median": float(np.median(cosine)),
        "signed_cosine_p05": float(np.quantile(cosine, 0.05)),
        "ideal_alpha_mean": float(ideal_alpha.mean()),
        "ideal_alpha_zero_fraction": float((ideal_alpha == 0.0).mean()),
    }


def selection_key(record: dict) -> tuple[float, ...]:
    protected_tail = (
        record["pass_fraction_scene_ge_minus_0_02"] >= 0.95
        and record["p05_scene_delta"] >= 0.0
    )
    return (
        float(protected_tail),
        record["p05_scene_delta"],
        record["mean_scene_delta"],
        record["pass_fraction_scene_ge_minus_0_02"],
        -record["scale"],
    )


def transform_target(name: str, target: np.ndarray) -> np.ndarray | None:
    if name == "raw_rgb":
        return target
    if name == "clip01":
        lower, upper = np.quantile(target, (0.01, 0.99), axis=0)
        return np.clip(target, lower, upper)
    if name == "direction_unit":
        norm = np.linalg.norm(target, axis=1, keepdims=True)
        return np.divide(
            target,
            norm,
            out=np.zeros_like(target),
            where=norm > 1e-12,
        )
    if name == "luma_chroma":
        luma = target @ LUMA
        rg = target[:, 0] - target[:, 1]
        bg = target[:, 2] - target[:, 1]
        return np.stack((luma, rg, bg), axis=1)
    return None


def invert_target(name: str, prediction: np.ndarray) -> np.ndarray:
    if name != "luma_chroma":
        return prediction
    luma, rg, bg = prediction.T
    # Invert [L, R-G, B-G] while preserving the green channel identity
    # implied by Y = 0.299R + 0.587G + 0.114B.
    chroma_weight = LUMA[0] * rg + LUMA[2] * bg
    green = (luma - chroma_weight) / LUMA.sum()
    return np.stack((green + rg, green, green + bg), axis=1)


def make_model(name: str, seed: int):
    if name.startswith("ridge_"):
        alpha = float(name.removeprefix("ridge_"))
        return make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    if name.startswith("huber_"):
        alpha = float(name.removeprefix("huber_"))
        return make_pipeline(
            StandardScaler(),
            MultiOutputRegressor(
                HuberRegressor(
                    alpha=alpha,
                    epsilon=1.35,
                    max_iter=2000,
                    tol=1e-6,
                ),
                n_jobs=-1,
            ),
        )
    if name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=20,
            max_features=0.65,
            random_state=seed,
            n_jobs=-1,
        )
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=20,
            max_features=0.65,
            random_state=seed,
            n_jobs=-1,
        )
    if name == "hist_gradient":
        return MultiOutputRegressor(
            HistGradientBoostingRegressor(
                learning_rate=0.04,
                max_iter=180,
                max_leaf_nodes=7,
                min_samples_leaf=40,
                l2_regularization=10.0,
                max_bins=128,
                random_state=seed,
            ),
            n_jobs=-1,
        )
    if name == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(32, 16),
                activation="relu",
                alpha=1.0,
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=25,
                random_state=seed,
            ),
        )
    raise ValueError(name)


def main() -> None:
    args = parse_args()
    payload = np.load(args.features, allow_pickle=False)
    features = np.asarray(payload["features"], dtype=np.float64)
    targets = np.asarray(payload["targets"], dtype=np.float64)
    base_mse = np.asarray(payload["base_mse"], dtype=np.float64)
    scenes = np.asarray(payload["scenes"]).astype(str)

    extended_name = f"extended{features.shape[1]}"
    feature_sets = {
        "raw12": features[:, :12],
        extended_name: features,
    }
    target_names = ("raw_rgb", "clip01", "direction_unit", "luma_chroma")
    model_names = (
        "ridge_0.01",
        "ridge_0.1",
        "ridge_1.0",
        "ridge_10.0",
        "huber_0.1",
        "huber_1.0",
        "hist_gradient",
        "extra_trees",
        "random_forest",
        "mlp",
    )
    splitter = GroupKFold(n_splits=5)
    results = {}

    for feature_name, matrix in feature_sets.items():
        for target_name in target_names:
            transformed = transform_target(target_name, targets)
            if transformed is None:
                raise ValueError(target_name)
            for model_name in model_names:
                if target_name == "luma_chroma" and model_name == "mlp":
                    # Nonlinear decomposition is retained below; skip only the
                    # most overflexible combination in this broad screen.
                    continue
                prediction_transformed = np.full_like(
                    transformed, np.nan, dtype=np.float64
                )
                for fold, (train_index, validation_index) in enumerate(
                    splitter.split(matrix, transformed, scenes)
                ):
                    model = make_model(model_name, args.seed + fold)
                    model.fit(matrix[train_index], transformed[train_index])
                    prediction_transformed[validation_index] = model.predict(
                        matrix[validation_index]
                    )
                if np.isnan(prediction_transformed).any():
                    raise RuntimeError(
                        f"incomplete prediction for {feature_name}/{target_name}/{model_name}"
                    )
                prediction = invert_target(
                    target_name, prediction_transformed
                )
                name = f"{feature_name}__{target_name}__{model_name}"
                results[name] = score(
                    scenes, base_mse, prediction, targets
                )

    order = sorted(
        results.items(),
        key=lambda item: selection_key(item[1]["selected_scale_record"]),
        reverse=True,
    )
    output = {
        "schema_version": 1,
        "seed": args.seed,
        "features": str(args.features),
        "selection_rule": (
            "scene-disjoint GroupKFold; protect >=95% of scenes at >= -0.02 dB "
            "and p05 >= 0; then maximize p05, mean, pass fraction, smaller scale"
        ),
        "selected_direction": order[0][0],
        "ranking": [name for name, _ in order],
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2) + "\n",
        encoding="utf-8",
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
                        "signed_cosine_mean": record["signed_cosine_mean"],
                        **record["selected_scale_record"],
                    }
                    for name, record in order[:15]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
