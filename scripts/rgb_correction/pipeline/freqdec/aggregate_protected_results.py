#!/usr/bin/env python3
"""Aggregate protected-gate evaluation and apply the frozen external gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


POPULATIONS = (
    "sidd_valid_1024",
    "sidd_train_crop_holdout_1024",
    "nind_reserve2_confirmation",
    "nind_reserve1_independent",
    "nind_remaining_56_scenes",
    "ccnoise_15",
    "polyu_30_scenes",
)
GAIN_GATES = {"ccnoise_15": 0.460, "polyu_30_scenes": 0.182}
INTERNAL_GATES = (
    "sidd_valid_1024",
    "sidd_train_crop_holdout_1024",
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def units(payload: dict) -> dict[str, dict[str, dict]]:
    return {
        population["population"]: {
            str(unit["unit"]): unit
            for unit in population["candidate_summary"]["units"]
        }
        for population in payload["populations"]
    }


def base_units(payload: dict) -> dict[str, dict[str, dict]]:
    return {
        population["population"]: {
            str(unit["unit"]): unit
            for unit in population["base_summary"]["units"]
        }
        for population in payload["populations"]
    }


def paired(unit_map: dict[str, dict[str, dict]], metric: str) -> dict:
    values = np.asarray(
        [
            float(unit[metric])
            for unit in unit_map.values()
        ],
        dtype=np.float64,
    )
    return {
        "n": int(values.size),
        "mean_delta": float(values.mean()),
        "median_delta": float(np.median(values)),
        "positive_fraction": float((values > 0.0).mean()),
        "p05": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95)),
        "min": float(values.min()),
        "max": float(values.max()),
        "values": values.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="seed*_tree.json")
    parser.add_argument("--global-pattern", default="seed*_global.json")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    paths = sorted(args.result_dir.glob(args.pattern))
    if not paths:
        raise FileNotFoundError("no result files")
    candidate_payloads = {}
    for path in paths:
        seed = int(path.stem.split("_")[0].removeprefix("seed"))
        candidate_payloads[seed] = load(path)

    global_paths = sorted(args.result_dir.glob(args.global_pattern))
    global_payloads = {}
    for path in global_paths:
        seed = int(path.stem.split("_")[0].removeprefix("seed"))
        global_payloads[seed] = load(path)

    comparisons: dict = {}
    for seed, payload in candidate_payloads.items():
        candidate = units(payload)
        base = base_units(payload)
        comparisons[seed] = {}
        for population in POPULATIONS:
            if population not in candidate:
                continue
            comparisons[seed][population] = {}
            for metric in ("psnr", "ssim", "lpips", "dists"):
                delta_map = {
                    unit: {
                        metric: candidate[population][unit][metric]
                        - base[population][unit][metric]
                    }
                    for unit in candidate[population]
                }
                comparisons[seed][population][metric] = paired(
                    delta_map, metric
                )

    aggregate = {}
    for population in POPULATIONS:
        if not all(
            population in comparisons[seed]
            for seed in comparisons
        ):
            continue
        aggregate[population] = {}
        for metric in ("psnr", "ssim", "lpips", "dists"):
            seed_means = {
                str(seed): comparisons[seed][population][metric]["mean_delta"]
                for seed in sorted(comparisons)
            }
            aggregate[population][metric] = {
                "seed_deltas": seed_means,
                "seed_count": len(seed_means),
                "seed_mean": float(np.mean(list(seed_means.values()))),
                "positive_seed_count": int(
                    sum(value > 0.0 for value in seed_means.values())
                ),
                "worst_seed_delta": float(min(seed_means.values())),
            }

    checks = {}
    for seed in comparisons:
        per_seed = comparisons[seed]
        gain_pass = all(
            per_seed[population]["psnr"]["mean_delta"] >= threshold
            and per_seed[population]["psnr"]["mean_delta"] > 0.0
            for population, threshold in GAIN_GATES.items()
        )
        internal_psnr = all(
            per_seed[population]["psnr"]["mean_delta"] >= -0.020
            for population in INTERNAL_GATES
        )
        nind2 = (
            per_seed["nind_reserve2_confirmation"]["psnr"]["mean_delta"]
            >= -0.020
            and per_seed["nind_reserve2_confirmation"]["ssim"]["mean_delta"]
            >= -0.0005
        )
        nind1 = (
            per_seed["nind_reserve1_independent"]["psnr"]["mean_delta"]
            >= -0.020
        )
        perceptual = all(
            per_seed[population][metric]["mean_delta"] <= 0.001
            for population in POPULATIONS
            if population in per_seed
            for metric in ("lpips", "dists")
        )
        checks[str(seed)] = {
            "gain_gate": gain_pass,
            "internal_psnr_gate": internal_psnr,
            "nind_reserve2_gate": nind2,
            "nind_reserve1_gate": nind1,
            "perceptual_gate": perceptual,
            "all_gates": bool(
                gain_pass and internal_psnr and nind2 and nind1 and perceptual
            ),
        }

    gain_all_seeds = all(
        aggregate[population]["psnr"]["worst_seed_delta"] >= threshold
        for population, threshold in GAIN_GATES.items()
    )
    protection_all_seeds = all(
        aggregate[population]["psnr"]["worst_seed_delta"] >= -0.020
        for population in INTERNAL_GATES
    )
    nind2_all = (
        aggregate["nind_reserve2_confirmation"]["psnr"]["worst_seed_delta"]
        >= -0.020
        and aggregate["nind_reserve2_confirmation"]["ssim"]["worst_seed_delta"]
        >= -0.0005
    )
    nind1_all = (
        aggregate["nind_reserve1_independent"]["psnr"]["worst_seed_delta"]
        >= -0.020
    )
    perceptual_all = all(
        per_seed[population][metric]["mean_delta"] <= 0.001
        for per_seed in comparisons.values()
        for population in POPULATIONS
        if population in per_seed
        for metric in ("lpips", "dists")
    )
    overall = bool(all(check["all_gates"] for check in checks.values()))

    incremental = {}
    common_seeds = sorted(set(candidate_payloads) & set(global_payloads))
    for population in ("ccnoise_15", "polyu_30_scenes"):
        if not common_seeds:
            break
        values = []
        for seed in common_seeds:
            candidate = units(candidate_payloads[seed])
            comparator = units(global_payloads[seed])
            population_ids = candidate[population].keys()
            delta = np.asarray(
                [
                    candidate[population][unit]["psnr"]
                    - comparator[population][unit]["psnr"]
                    for unit in population_ids
                ]
            )
            values.append(float(delta.mean()))
        incremental[population] = {
            "seed_deltas": {
                str(seed): value
                for seed, value in zip(common_seeds, values, strict=True)
            },
            "seed_count": len(values),
            "seed_mean": float(np.mean(values)),
            "positive_seed_count": int(sum(value > 0 for value in values)),
        }

    adaptive_gate_increment = False
    if common_seeds and len(common_seeds) == len(candidate_payloads):
        one_gain = any(
            incremental[population]["seed_mean"] >= 0.030
            and incremental[population]["positive_seed_count"] >= 2
            for population in incremental
        )
        other_not_worse = all(
            incremental[population]["seed_mean"] >= -0.020
            for population in incremental
        )
        adaptive_gate_increment = bool(one_gain and other_not_worse)

    output = {
        "schema_version": 1,
        "result_files": [str(path) for path in paths],
        "global_result_files": [str(path) for path in global_paths],
        "per_seed": {
            str(seed): comparisons[seed] for seed in sorted(comparisons)
        },
        "aggregate": aggregate,
        "checks": checks,
        "overall_checks": {
            "gain_all_seeds": gain_all_seeds,
            "protection_all_seeds": protection_all_seeds,
            "nind_reserve2_all_seeds": nind2_all,
            "nind_reserve1_all_seeds": nind1_all,
            "perceptual_all_seeds": perceptual_all,
            "candidate_passes_existing_gates": overall,
        },
        "adaptive_vs_global": {
            "comparisons": incremental,
            "increment_gate": adaptive_gate_increment,
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Protected correction: external screen",
        "",
        "Seed means use all result files supplied to this aggregation.",
        "",
        "| Population | PSNR | SSIM | LPIPS | DISTS |",
        "|---|---:|---:|---:|---:|",
    ]
    for population in POPULATIONS:
        values = aggregate[population]
        lines.append(
            f"| {population} | "
            f"{values['psnr']['seed_mean']:+.6f} | "
            f"{values['ssim']['seed_mean']:+.6f} | "
            f"{values['lpips']['seed_mean']:+.6f} | "
            f"{values['dists']['seed_mean']:+.6f} |"
        )
    seed_ids = sorted(comparisons)
    lines += [
        "",
        "## Per-seed PSNR",
        "",
        "| Population | " + " | ".join(f"Seed {seed}" for seed in seed_ids) + " |",
        "|---|" + "---:|" * len(seed_ids),
    ]
    for population in POPULATIONS:
        deltas = aggregate[population]["psnr"]["seed_deltas"]
        lines.append(
            f"| {population} | "
            + " | ".join(
                f"{deltas[str(seed)]:+.6f}" for seed in seed_ids
            )
            + " |"
        )
    lines += [
        "",
        "## Gates",
        "",
        "```json",
        json.dumps(output["overall_checks"], indent=2),
        "```",
    ]
    if incremental:
        lines += [
            "",
            "## Adaptive increment over source-selected global scale",
            "",
            "```json",
            json.dumps(output["adaptive_vs_global"], indent=2),
            "```",
        ]
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "overall_checks": output["overall_checks"],
        "aggregate": aggregate,
        "adaptive_vs_global": output["adaptive_vs_global"],
    }, indent=2))


if __name__ == "__main__":
    main()
