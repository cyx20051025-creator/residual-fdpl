"""Apply the pre-registered A/C/E advancement gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


EXTERNAL_GATES = ("nind_reserve2_confirmation", "ccnoise_15")
INTERNAL_GATES = ("sidd_valid_1024", "sidd_train_crop_holdout_1024")


def units_by_population(payload: dict) -> dict[str, dict[str, dict]]:
    result = {}
    for population in payload["populations"]:
        result[population["population"]] = {
            str(unit["unit"]): unit for unit in population["summary"]["units"]
        }
    return result


def paired_delta(
    candidate: dict[str, dict],
    comparator: dict[str, dict],
    metric: str,
    *,
    seed: int,
    draws: int = 10_000,
) -> dict[str, object]:
    common = sorted(set(candidate) & set(comparator))
    if not common:
        raise ValueError("no common primary units")
    delta = np.asarray(
        [float(candidate[unit][metric]) - float(comparator[unit][metric]) for unit in common]
    )
    rng = np.random.RandomState(seed)
    indices = rng.randint(0, len(delta), size=(draws, len(delta)))
    bootstrap = delta[indices].mean(axis=1)
    return {
        "metric": metric,
        "unit_count": len(common),
        "mean_delta": float(delta.mean()),
        "median_delta": float(np.median(delta)),
        "positive_fraction": float((delta > 0).mean()),
        "bootstrap_95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "units": common,
        "deltas": delta.tolist(),
    }


def perceptual_guardrail_ok(mean_delta: float, metric: str, threshold: float) -> bool:
    """Apply the documented directional limit: SSIM may lose, not gain, at most threshold."""
    if metric == "ssim":
        return mean_delta >= -threshold
    return mean_delta <= threshold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--c", type=Path, required=True)
    parser.add_argument("--e", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261114)
    args = parser.parse_args()
    payloads = {
        "A": json.loads(args.a.read_text(encoding="utf-8")),
        "C": json.loads(args.c.read_text(encoding="utf-8")),
        "E": json.loads(args.e.read_text(encoding="utf-8")),
    }
    units = {arm: units_by_population(payload) for arm, payload in payloads.items()}
    comparisons: dict[str, dict[str, dict[str, object]]] = {}
    for name, candidate, comparator in (
        ("C_minus_A", "C", "A"),
        ("C_minus_E", "C", "E"),
        ("E_minus_A", "E", "A"),
    ):
        comparisons[name] = {}
        for population in EXTERNAL_GATES + INTERNAL_GATES + (
            "nind_remaining_56_scenes",
            "polyu_30_scenes",
            "nind_reserve1_independent",
        ):
            if population not in units[candidate]:
                continue
            comparisons[name][population] = {
                metric: paired_delta(
                    units[candidate][population],
                    units[comparator][population],
                    metric,
                    seed=args.seed + len(name) + len(population) + index,
                )
                for index, metric in enumerate(
                    ("psnr", "ssim", "lpips", "dists", "dc_mse", "lf_mse", "hf_mse")
                )
            }

    ce = comparisons["C_minus_E"]
    ca = comparisons["C_minus_A"]
    checks = {
        "c_minus_e_psnr_at_least_0_03_on_both_external": all(
            ce[population]["psnr"]["mean_delta"] >= 0.03
            for population in EXTERNAL_GATES
        ),
        "c_minus_e_positive_fraction_at_least_0_60_on_both_external": all(
            ce[population]["psnr"]["positive_fraction"] >= 0.60
            for population in EXTERNAL_GATES
        ),
        "c_within_0_02_db_of_a_on_both_internal": all(
            ca[population]["psnr"]["mean_delta"] >= -0.02
            for population in INTERNAL_GATES
        ),
        "c_within_0_02_db_of_e_on_both_internal": all(
            ce[population]["psnr"]["mean_delta"] >= -0.02
            for population in INTERNAL_GATES
        ),
        "perceptual_guardrails": all(
            perceptual_guardrail_ok(
                ce[population][metric]["mean_delta"], metric, threshold
            )
            for population in EXTERNAL_GATES
            for metric, threshold in (("lpips", 0.002), ("dists", 0.002), ("ssim", 0.0005))
        ),
        "c_minus_a_positive_internal_and_external": (
            any(ca[population]["psnr"]["mean_delta"] >= 0.02 for population in INTERNAL_GATES)
            and any(ca[population]["psnr"]["mean_delta"] > 0 for population in EXTERNAL_GATES)
        ),
    }
    checks["stage1_pass"] = all(checks.values())
    result = {
        "a": str(args.a),
        "c": str(args.c),
        "e": str(args.e),
        "comparisons": comparisons,
        "gate_checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
