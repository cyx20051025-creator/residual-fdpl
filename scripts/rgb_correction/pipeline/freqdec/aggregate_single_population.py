#!/usr/bin/env python3
"""Aggregate one population across paired seed result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", action="append", type=Path, required=True)
    parser.add_argument("--population", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_values: dict[str, dict[str, float]] = {}
    for path in args.result:
        payload = json.loads(path.read_text(encoding="utf-8"))
        population = next(
            item
            for item in payload["populations"]
            if item["population"] == args.population
        )
        base = population["base_summary"]["metric_means"]
        candidate = population["candidate_summary"]["metric_means"]
        seed = path.name.split("_", 1)[0].removeprefix("seed")
        seed_values[seed] = {
            metric: float(candidate[metric] - base[metric])
            for metric in ("psnr", "ssim", "lpips", "dists")
        }
    metrics = {}
    for metric in ("psnr", "ssim", "lpips", "dists"):
        values = [seed_values[seed][metric] for seed in sorted(seed_values)]
        metrics[metric] = {
            "seed_deltas": {
                seed: seed_values[seed][metric] for seed in sorted(seed_values)
            },
            "seed_mean": float(np.mean(values)),
            "worst_seed_delta": float(np.min(values)),
            "positive_seed_count": int(sum(value > 0.0 for value in values)),
            "seed_count": len(values),
        }
    checks = {
        "psnr_all_seeds_positive": all(
            value["psnr"] > 0.0 for value in seed_values.values()
        ),
        "lpips_guardrail": metrics["lpips"]["worst_seed_delta"] <= 0.001,
        "dists_guardrail": metrics["dists"]["worst_seed_delta"] <= 0.001,
    }
    checks["consistent_with_benefit"] = all(checks.values())
    output = {
        "schema_version": 1,
        "result_files": [str(path) for path in args.result],
        "population": args.population,
        "metrics": metrics,
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
