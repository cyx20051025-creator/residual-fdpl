#!/usr/bin/env python3
"""Screen a standard Gray-World color-constancy baseline on source crops."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .screen_noisy_mean_deadzone import score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def project_zero_luma(values: np.ndarray) -> np.ndarray:
    weight = np.asarray([0.299, 0.587, 0.114], dtype=np.float64)
    luma = values @ weight
    return values - luma[:, None] * weight[None, :] / float(weight @ weight)


def main() -> None:
    args = parse_args()
    payload = np.load(args.oof, allow_pickle=False)
    features = np.asarray(payload["features"], dtype=np.float64)
    targets = np.asarray(payload["targets"], dtype=np.float64)
    base_mse = np.asarray(payload["base_mse"], dtype=np.float64)
    scenes = np.asarray(payload["scenes"]).astype(str)

    # The base RGB means are stored in 0-1 image coordinates.
    base_mean = features[:, (6, 8, 10)]
    gray_mean = base_mean.mean(axis=1, keepdims=True)
    raw_direction = 2.0 * (gray_mean - base_mean)

    results = {}
    for name, direction in (
        ("gray_world_raw", raw_direction),
        ("gray_world_zero_luma", project_zero_luma(raw_direction)),
    ):
        record = score(scenes, base_mse, direction, targets)
        record["eligible"] = bool(
            record["selected_scale"] > 0.0
            and record["selected_scale_record"][
                "pass_fraction_scene_ge_minus_0_02"
            ]
            >= 0.95
            and record["selected_scale_record"]["p05_scene_delta"] >= 0.0
        )
        results[name] = record

    eligible = [name for name, record in results.items() if record["eligible"]]
    selected = None
    if eligible:
        selected = max(
            eligible,
            key=lambda name: (
                results[name]["selected_scale_record"]["p05_scene_delta"],
                results[name]["selected_scale_record"]["mean_scene_delta"],
                results[name]["selected_scale_record"][
                    "pass_fraction_scene_ge_minus_0_02"
                ],
                -results[name]["selected_scale_record"]["scale"],
            ),
        )

    output = {
        "schema_version": 1,
        "seed": args.seed,
        "oof": str(args.oof),
        "baseline": "gray_world",
        "selected_candidate": selected,
        "selected_candidate_record": results[selected] if selected else None,
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
                "selected_candidate": selected,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
