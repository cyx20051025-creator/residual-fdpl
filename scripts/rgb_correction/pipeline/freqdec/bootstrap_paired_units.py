#!/usr/bin/env python3
"""Bootstrap paired candidate-minus-base deltas over primary evaluator units."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = ("psnr", "ssim", "lpips", "dists")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261114)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    rng = np.random.default_rng(args.seed)
    populations = {}
    for population in payload["populations"]:
        base = {
            str(unit["unit"]): unit
            for unit in population["base_summary"]["units"]
        }
        candidate = {
            str(unit["unit"]): unit
            for unit in population["candidate_summary"]["units"]
        }
        if set(base) != set(candidate):
            raise ValueError(
                f"unit mismatch for {population['population']}"
            )
        metrics = {}
        for metric in METRICS:
            deltas = np.asarray(
                [
                    float(candidate[unit][metric])
                    - float(base[unit][metric])
                    for unit in sorted(base)
                ],
                dtype=np.float64,
            )
            samples = rng.choice(
                deltas,
                size=(args.bootstrap, deltas.size),
                replace=True,
            ).mean(axis=1)
            metrics[metric] = {
                "count": int(deltas.size),
                "mean": float(deltas.mean()),
                "median": float(np.median(deltas)),
                "positive_fraction": float((deltas > 0.0).mean()),
                "ci95_low": float(np.quantile(samples, 0.025)),
                "ci95_high": float(np.quantile(samples, 0.975)),
            }
        populations[population["population"]] = {
            "primary_unit": population["base_summary"]["primary_unit"],
            "metrics": metrics,
        }
    output = {
        "schema_version": 1,
        "result": str(args.result),
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_seed": args.seed,
        "populations": populations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
