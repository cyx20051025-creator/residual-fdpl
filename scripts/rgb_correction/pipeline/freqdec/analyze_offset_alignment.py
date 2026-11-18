#!/usr/bin/env python3
"""Compare predicted effective offsets with a label-aware zero-luma oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--population", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def population(payload: dict, population_id: str) -> dict:
    return next(
        item
        for item in payload["populations"]
        if item["population"] == population_id
    )


def cosine_matrix(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    numerator = (first * second).sum(axis=1)
    denominator = (
        np.linalg.norm(first, axis=1)
        * np.linalg.norm(second, axis=1)
    )
    return numerator / np.maximum(denominator, 1e-12)


def main() -> None:
    args = parse_args()
    candidate_payload = json.loads(args.candidate.read_text(encoding="utf-8"))
    oracle_payload = json.loads(args.oracle.read_text(encoding="utf-8"))
    candidate = population(candidate_payload, args.population)
    oracle = population(oracle_payload, args.population)

    candidate_rows = {
        int(row["row"]): row for row in candidate["records"]
    }
    oracle_rows = {
        int(row["row"]): row for row in oracle["records"]
    }
    rows = sorted(set(candidate_rows) & set(oracle_rows))
    if not rows:
        raise ValueError("no aligned rows")
    predicted = np.asarray(
        [
            candidate_rows[row]["effective_offset_rgb"]
            for row in rows
        ],
        dtype=np.float64,
    )
    target = np.asarray(
        [
            oracle_rows[row]["oracle_offset_rgb"]
            for row in rows
        ],
        dtype=np.float64,
    )
    row_cosine = cosine_matrix(predicted, target)
    row_delta = np.asarray(
        [
            float(candidate_rows[row]["candidate_psnr"])
            - float(candidate_rows[row]["base_psnr"])
            for row in rows
        ]
    )
    oracle_delta = np.asarray(
        [float(oracle_rows[row]["psnr_delta"]) for row in rows]
    )

    scene_to_values: dict[str, dict[str, list[float]]] = {}
    for row, cosine, delta, headroom in zip(
        rows,
        row_cosine,
        row_delta,
        oracle_delta,
        strict=True,
    ):
        scene = str(
            oracle_rows[row].get(
                "scene",
                oracle_rows[row].get("unit_id", row),
            )
        )
        scene_to_values.setdefault(
            scene,
            {
                "row_cosine": [],
                "candidate_delta": [],
                "oracle_delta": [],
            },
        )
        scene_to_values[scene]["row_cosine"].append(float(cosine))
        scene_to_values[scene]["candidate_delta"].append(float(delta))
        scene_to_values[scene]["oracle_delta"].append(float(headroom))
    scene_values = {
        scene: {
            key: float(np.mean(values))
            for key, values in metrics.items()
        }
        for scene, metrics in scene_to_values.items()
    }
    scene_cosine = np.asarray(
        [value["row_cosine"] for value in scene_values.values()]
    )
    candidate_scene_delta = np.asarray(
        [value["candidate_delta"] for value in scene_values.values()]
    )
    oracle_scene_delta = np.asarray(
        [value["oracle_delta"] for value in scene_values.values()]
    )

    predicted_mean = predicted.mean(axis=0)
    target_mean = target.mean(axis=0)
    mean_cosine = float(
        predicted_mean @ target_mean
        / max(
            np.linalg.norm(predicted_mean) * np.linalg.norm(target_mean),
            1e-12,
        )
    )
    output = {
        "schema_version": 1,
        "population": args.population,
        "candidate": str(args.candidate),
        "oracle": str(args.oracle),
        "row_count": len(rows),
        "scene_count": len(scene_values),
        "predicted_offset_mean_rgb": predicted_mean.tolist(),
        "oracle_offset_mean_rgb": target_mean.tolist(),
        "predicted_mean_norm": float(np.linalg.norm(predicted_mean)),
        "oracle_mean_norm": float(np.linalg.norm(target_mean)),
        "mean_vector_cosine": mean_cosine,
        "row_cosine": {
            "mean": float(row_cosine.mean()),
            "median": float(np.median(row_cosine)),
            "positive_fraction": float((row_cosine > 0.0).mean()),
            "p05": float(np.quantile(row_cosine, 0.05)),
            "p95": float(np.quantile(row_cosine, 0.95)),
        },
        "scene_cosine": {
            "mean": float(scene_cosine.mean()),
            "median": float(np.median(scene_cosine)),
            "positive_fraction": float((scene_cosine > 0.0).mean()),
        },
        "candidate_scene_delta_psnr": {
            "mean": float(candidate_scene_delta.mean()),
            "positive_fraction": float(
                (candidate_scene_delta > 0.0).mean()
            ),
        },
        "oracle_scene_delta_psnr": {
            "mean": float(oracle_scene_delta.mean()),
            "positive_fraction": float((oracle_scene_delta > 0.0).mean()),
        },
        "scene_values": scene_values,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
