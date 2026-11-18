#!/usr/bin/env python3
"""Scene-aggregated intervals for calibration-budget selector outputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def bootstrap_interval(
    values: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = np.asarray(
        [
            rng.choice(values, size=len(values), replace=True).mean()
            for _ in range(samples)
        ],
        dtype=np.float64,
    )
    low, high = np.quantile(draws, (0.025, 0.975))
    return float(low), float(high)


def summarize(path: Path, samples: int) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    endpoint_seed = payload.get("endpoint_seed")
    if endpoint_seed is None:
        match = re.search(r"_seed(\d+)_budget\d+\.json$", path.name)
        if match:
            endpoint_seed = int(match.group(1))
    per_scene: dict[str, list[float]] = defaultdict(list)
    per_scene_minus_fixed: dict[str, list[float]] = defaultdict(list)
    for record in payload["records"]:
        selected = record["decision"]["selected"]
        summary = record["evaluation_by_candidate"][selected]
        fixed = record["evaluation_by_candidate"]["fixed_target"]
        for scene, metrics in summary["scene_metrics"].items():
            per_scene[scene].append(float(metrics["psnr"]))
            per_scene_minus_fixed[scene].append(
                float(metrics["psnr"])
                - float(fixed["scene_metrics"][scene]["psnr"])
            )
    scene_means = np.asarray(
        [np.mean(per_scene[scene]) for scene in sorted(per_scene)],
        dtype=np.float64,
    )
    scene_minus_fixed_means = np.asarray(
        [
            np.mean(per_scene_minus_fixed[scene])
            for scene in sorted(per_scene_minus_fixed)
        ],
        dtype=np.float64,
    )
    low, high = bootstrap_interval(
        scene_means,
        samples=samples,
        seed=20261117,
    )
    minus_low, minus_high = bootstrap_interval(
        scene_minus_fixed_means,
        samples=samples,
        seed=20261118,
    )
    return {
        "file": path.name,
        "architecture": payload["architecture"],
        "endpoint_seed": endpoint_seed,
        "source_seed": payload["source_seed"],
        "population_id": payload["population_id"],
        "budget": int(payload["calibration_scenes"]),
        "unique_scenes": len(scene_means),
        "scene_psnr_mean": float(scene_means.mean()),
        "scene_psnr_low": low,
        "scene_psnr_high": high,
        "scene_minus_fixed_mean": float(scene_minus_fixed_means.mean()),
        "scene_minus_fixed_low": minus_low,
        "scene_minus_fixed_high": minus_high,
        "scene_minus_fixed_positive_fraction": float(
            (scene_minus_fixed_means > 1e-6).mean()
        ),
        "positive_scene_fraction": float((scene_means > 1e-6).mean()),
        "worst_scene": float(scene_means.min()),
        "best_scene": float(scene_means.max()),
        "selected_counts": " / ".join(
            f"{name}:{count}"
            for name, count in payload["selected_counts"].items()
            if count
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    args = parser.parse_args()

    paths = []
    for root in args.results:
        paths.extend(root.glob("**/risk_selector_no_source_*_budget*.json"))
    rows = sorted(
        (summarize(path, args.bootstrap_samples) for path in set(paths)),
        key=lambda row: (
            row["population_id"],
            row["architecture"],
            row["endpoint_seed"] or 0,
            row["budget"],
        ),
    )
    if not rows:
        raise SystemExit("no budget result files found")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Scene-aggregated calibration-budget outcomes",
        "",
        "Repeated visits to a scene are averaged before the bootstrap.",
        "These intervals therefore describe the reused scene population and",
        "do not turn repeated splits into independent populations.",
        "",
        "| Population | Backbone | Endpoint | Scenes | ΔPSNR [95%] | "
        "Selected−fixed [95%] | Positive | Worst | Actions |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['population_id']} | {row['architecture']} | "
            f"{row['endpoint_seed'] or '—'} | {row['budget']} | "
            f"{row['scene_psnr_mean']:+.4f} "
            f"[{row['scene_psnr_low']:+.4f}, "
            f"{row['scene_psnr_high']:+.4f}] | "
            f"{row['scene_minus_fixed_mean']:+.4f} "
            f"[{row['scene_minus_fixed_low']:+.4f}, "
            f"{row['scene_minus_fixed_high']:+.4f}] | "
            f"{row['positive_scene_fraction']:.0%} | "
            f"{row['worst_scene']:+.4f} | {row['selected_counts']} |"
        )
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
