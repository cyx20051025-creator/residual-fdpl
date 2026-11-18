"""Aggregate the pre-registered three-seed Stage-2 replication gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SEEDS = (56, 57, 58)
EXTERNAL_GATES = ("nind_reserve2_confirmation", "ccnoise_15")
INTERNAL_GATES = ("sidd_valid_1024", "sidd_train_crop_holdout_1024")
PERCEPTUAL_LIMITS = (("ssim", 0.0005), ("lpips", 0.002), ("dists", 0.002))


def guardrail_ok(mean_delta: float, metric: str, threshold: float) -> bool:
    if metric == "ssim":
        return mean_delta >= -threshold
    return mean_delta <= threshold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payloads = {
        seed: json.loads(
            (args.root / f"stage1_comparison_seed{seed}.json").read_text(
                encoding="utf-8"
            )
        )
        for seed in SEEDS
    }

    external = {}
    for population in EXTERNAL_GATES:
        seed_deltas = {
            seed: payloads[seed]["comparisons"]["C_minus_E"][population]["psnr"][
                "mean_delta"
            ]
            for seed in SEEDS
        }
        external[population] = {
            "seed_mean_delta": seed_deltas,
            "three_seed_mean_delta": sum(seed_deltas.values()) / len(SEEDS),
            "positive_seed_count": sum(value > 0 for value in seed_deltas.values()),
            "mean_at_least_0_02": (
                sum(seed_deltas.values()) / len(SEEDS) >= 0.02
            ),
            "positive_in_at_least_two_seeds": (
                sum(value > 0 for value in seed_deltas.values()) >= 2
            ),
        }

    internal = {}
    for seed in SEEDS:
        internal[seed] = {}
        for population in INTERNAL_GATES:
            deltas = {
                comparator: payloads[seed]["comparisons"][f"C_minus_{comparator}"][
                    population
                ]["psnr"]["mean_delta"]
                for comparator in ("A", "E")
            }
            internal[seed][population] = {
                "mean_delta": deltas,
                "within_0_02_db": all(value >= -0.02 for value in deltas.values()),
            }

    perceptual = {}
    for seed in SEEDS:
        perceptual[seed] = {}
        for population in EXTERNAL_GATES:
            payload = payloads[seed]["comparisons"]["C_minus_E"][population]
            perceptual[seed][population] = {
                metric: {
                    "mean_delta": payload[metric]["mean_delta"],
                    "within_documented_limit": guardrail_ok(
                        payload[metric]["mean_delta"], metric, threshold
                    ),
                }
                for metric, threshold in PERCEPTUAL_LIMITS
            }

    checks = {
        "external_three_seed_means": all(
            result["mean_at_least_0_02"] for result in external.values()
        ),
        "external_positive_seeds": all(
            result["positive_in_at_least_two_seeds"] for result in external.values()
        ),
        "internal_guardrails_every_seed": all(
            result["within_0_02_db"]
            for seed_result in internal.values()
            for result in seed_result.values()
        ),
        "perceptual_guardrails_every_seed": all(
            result["within_documented_limit"]
            for population_result in perceptual.values()
            for metric_result in population_result.values()
            for result in metric_result.values()
        ),
    }
    checks["stage2_pass"] = all(checks.values())

    output = {
        "seeds": list(SEEDS),
        "external_gates": external,
        "internal_guardrails": internal,
        "perceptual_guardrails": perceptual,
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
