#!/usr/bin/env python3
"""Screen source-only ensembles of RGB direction predictors."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from .luma_weights import luma_weights
from .screen_direction_targets import (
    invert_target,
    make_model,
    score,
    selection_key,
    transform_target,
)


CONFIGS = (
    ("extended36", "raw_rgb", "ridge_1.0"),
    ("extended36", "clip01", "huber_1.0"),
    ("extended36", "clip01", "hist_gradient"),
    ("extended36", "luma_chroma", "huber_1.0"),
    ("extended36", "luma_chroma", "hist_gradient"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--oof-output", type=Path)
    return parser.parse_args()


def unit_rows(values: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(
        values,
        norm,
        out=np.zeros_like(values),
        where=norm > 1e-12,
    )


def ensemble_candidates(
    predictions: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    names = tuple(predictions)
    matrices = np.stack([predictions[name] for name in names], axis=0)
    units = np.stack([unit_rows(matrix) for matrix in matrices], axis=0)
    norms = np.linalg.norm(matrices, axis=2)
    candidates = {
        "mean_rgb": matrices.mean(axis=0),
        "median_rgb": np.median(matrices, axis=0),
        "mean_unit_x_mean_norm": unit_rows(units.mean(axis=0))
        * norms.mean(axis=0)[:, None],
        "mean_unit_x_median_norm": unit_rows(units.mean(axis=0))
        * np.median(norms, axis=0)[:, None],
        "median_unit_x_mean_norm": unit_rows(
            np.median(units, axis=0)
        )
        * norms.mean(axis=0)[:, None],
    }

    # Pairwise unit-direction agreement is a direct source-side uncertainty
    # signal. Candidate offsets shrink rows on which model families disagree.
    pairwise_cosines = []
    for first, second in itertools.combinations(range(len(names)), 2):
        denominator = (
            np.linalg.norm(matrices[first], axis=1)
            * np.linalg.norm(matrices[second], axis=1)
        )
        cosine = np.divide(
            np.einsum("ij,ij->i", matrices[first], matrices[second]),
            denominator,
            out=np.zeros(len(matrices[first])),
            where=denominator > 1e-12,
        )
        pairwise_cosines.append(cosine)
    pairwise_cosines = np.stack(pairwise_cosines, axis=1)
    agreement = pairwise_cosines.mean(axis=1)
    consensus = unit_rows(units.mean(axis=0)) * np.median(
        norms, axis=0
    )[:, None]
    for threshold in (0.0, 0.25, 0.50, 0.75, 0.90):
        confidence = np.clip(
            (agreement - threshold) / max(1e-6, 1.0 - threshold),
            0.0,
            1.0,
        )
        candidates[f"consensus_x_agreement_ge_{threshold:.2f}"] = (
            consensus * confidence[:, None]
        )
    return candidates


def capped_candidates(
    candidates: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    output = dict(candidates)
    for name, values in candidates.items():
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        for cap in (0.020, 0.030, 0.040, 0.050, 0.060):
            shrink = np.minimum(
                1.0,
                cap / np.maximum(norms, 1e-12),
            )
            output[f"cap_{cap:.3f}__{name}"] = values * shrink
    return output


def subspace_candidates(
    candidates: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    luma = luma_weights()
    luma_norm_sq = float(luma @ luma)
    output = dict(candidates)
    for name, values in candidates.items():
        luma_component = (values @ luma)[:, None]
        output[f"chroma__{name}"] = (
            values - luma_component * luma / luma_norm_sq
        )
        output[f"luma__{name}"] = luma_component * luma
    return output


def main() -> None:
    args = parse_args()
    payload = np.load(args.features, allow_pickle=False)
    features = np.asarray(payload["features"], dtype=np.float64)
    targets = np.asarray(payload["targets"], dtype=np.float64)
    base_mse = np.asarray(payload["base_mse"], dtype=np.float64)
    scenes = np.asarray(payload["scenes"]).astype(str)
    splitter = GroupKFold(n_splits=5)

    individual: dict[str, np.ndarray] = {}
    for feature_name, target_name, model_name in CONFIGS:
        matrix = features if feature_name == "extended36" else features[:, :12]
        transformed = transform_target(target_name, targets)
        if transformed is None:
            raise ValueError(target_name)
        prediction_transformed = np.full_like(transformed, np.nan)
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
                f"incomplete OOF predictions for {feature_name}/{target_name}/{model_name}"
            )
        name = f"{feature_name}__{target_name}__{model_name}"
        individual[name] = invert_target(
            target_name, prediction_transformed
        )

    individual_results = {
        name: score(scenes, base_mse, prediction, targets)
        for name, prediction in individual.items()
    }
    candidates = subspace_candidates(
        capped_candidates(
            {**individual, **ensemble_candidates(individual)}
        )
    )
    results = dict(individual_results)
    for name, prediction in candidates.items():
        results[name] = score(scenes, base_mse, prediction, targets)

    order = sorted(
        results.items(),
        key=lambda item: selection_key(item[1]["selected_scale_record"]),
        reverse=True,
    )
    payload_out = {
        "schema_version": 1,
        "seed": args.seed,
        "features": str(args.features),
        "individual_members": list(individual),
        "selected_direction": order[0][0],
        "ranking": [name for name, _ in order],
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload_out, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.oof_output is not None:
        arrays = {
            f"prediction__{name}": values
            for name, values in individual.items()
        }
        args.oof_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.oof_output,
            features=features,
            targets=targets,
            base_mse=base_mse,
            scenes=scenes,
            **arrays,
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
