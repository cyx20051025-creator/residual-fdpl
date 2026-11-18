#!/usr/bin/env python3
"""Screen source-only gates for an ensemble RGB correction."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .luma_weights import luma_weights
from .screen_direction_ensembles import (
    capped_candidates,
    ensemble_candidates,
    subspace_candidates,
)
from .screen_direction_targets import score, selection_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--rule", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def vector_summary(values: np.ndarray) -> np.ndarray:
    luma = values @ luma_weights()
    rg = values[:, 0] - values[:, 1]
    bg = values[:, 2] - values[:, 1]
    return np.column_stack(
        (values, np.linalg.norm(values, axis=1), luma, rg, bg)
    )


def gate_matrix(
    features: np.ndarray,
    member_predictions: dict[str, np.ndarray],
    direction: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    blocks = [features, vector_summary(direction)]
    names = list(range(features.shape[1]))
    for name, values in member_predictions.items():
        blocks.append(vector_summary(values))
    pairwise = []
    members = list(member_predictions)
    for first, second in itertools.combinations(members, 2):
        first_values = member_predictions[first]
        second_values = member_predictions[second]
        numerator = np.einsum("ij,ij->i", first_values, second_values)
        denominator = (
            np.linalg.norm(first_values, axis=1)
            * np.linalg.norm(second_values, axis=1)
        )
        cosine = np.divide(
            numerator,
            denominator,
            out=np.zeros(len(features)),
            where=denominator > 1e-12,
        )
        pairwise.append(cosine)
    pairwise_matrix = np.stack(pairwise, axis=1)
    blocks.extend(
        (
            pairwise_matrix,
            pairwise_matrix.mean(axis=1, keepdims=True),
            pairwise_matrix.min(axis=1, keepdims=True),
            pairwise_matrix.std(axis=1, keepdims=True),
        )
    )
    return np.concatenate(blocks, axis=1), names


def model(name: str, seed: int):
    if name.startswith("ridge_"):
        return make_pipeline(
            StandardScaler(),
            Ridge(alpha=float(name.removeprefix("ridge_"))),
        )
    if name == "hist_gradient":
        return HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.04,
            max_iter=180,
            max_leaf_nodes=7,
            min_samples_leaf=40,
            l2_regularization=10.0,
            max_bins=128,
            random_state=seed,
        )
    raise ValueError(name)


def main() -> None:
    args = parse_args()
    payload = np.load(args.oof, allow_pickle=False)
    features = np.asarray(payload["features"], dtype=np.float64)
    targets = np.asarray(payload["targets"], dtype=np.float64)
    base_mse = np.asarray(payload["base_mse"], dtype=np.float64)
    scenes = np.asarray(payload["scenes"]).astype(str)
    member_predictions = {
        key.removeprefix("prediction__"): np.asarray(
            payload[key], dtype=np.float64
        )
        for key in payload.files
        if key.startswith("prediction__")
    }
    candidates = subspace_candidates(
        capped_candidates(
            {
                **member_predictions,
                **ensemble_candidates(member_predictions),
            }
        )
    )
    if args.rule not in candidates:
        raise ValueError(f"unknown direction rule: {args.rule}")
    direction = candidates[args.rule]
    target_alpha = np.clip(
        np.einsum("ij,ij->i", direction, targets)
        / np.maximum(np.einsum("ij,ij->i", direction, direction), 1e-12),
        0.0,
        1.0,
    )
    matrix, _ = gate_matrix(
        features, member_predictions, direction
    )
    splitter = GroupKFold(n_splits=5)
    results = {}
    for model_name in (
        "ridge_0.1",
        "ridge_1.0",
        "ridge_10.0",
        "ridge_100.0",
        "hist_gradient",
    ):
        prediction = np.full(len(targets), np.nan)
        fold = 0
        for train_index, validation_index in splitter.split(
            matrix, target_alpha, scenes
        ):
            candidate = model(model_name, 700 + fold)
            candidate.fit(matrix[train_index], target_alpha[train_index])
            prediction[validation_index] = np.clip(
                candidate.predict(matrix[validation_index]),
                0.0,
                1.0,
            )
            fold += 1
        if np.isnan(prediction).any():
            raise RuntimeError("gate OOF predictions incomplete")
        # Re-score with per-row gate magnitudes. The generic score helper
        # accepts an effective offset, so scale the direction first and set
        # the standalone scale to one.
        gated_offset = prediction[:, None] * direction
        record = score(
            scenes,
            base_mse,
            gated_offset,
            targets,
        )
        calibration = float(record["selected_scale"])
        record["alpha_mean"] = float(
            calibration * prediction.mean()
        )
        record["alpha_zero_fraction"] = float((prediction == 0.0).mean())
        results[model_name] = record

    gated_offset = direction
    global_record = score(
        scenes, base_mse, gated_offset, targets
    )
    global_record["alpha_mean"] = 1.0
    global_record["alpha_zero_fraction"] = 0.0
    results["constant_gate"] = global_record
    order = sorted(
        results.items(),
        key=lambda item: selection_key(item[1]["selected_scale_record"]),
        reverse=True,
    )
    output = {
        "schema_version": 1,
        "oof": str(args.oof),
        "rule": args.rule,
        "target_alpha_mean": float(target_alpha.mean()),
        "target_alpha_zero_fraction": float(
            (target_alpha == 0.0).mean()
        ),
        "selected_gate": order[0][0],
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
                "rule": args.rule,
                "selected_gate": order[0][0],
                "top": [
                    {
                        "name": name,
                        "alpha_mean": record.get("alpha_mean"),
                        **record["selected_scale_record"],
                    }
                    for name, record in order
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
