#!/usr/bin/env python3
"""Decompose frozen-A RGB correction errors into direction and scale effects."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-rows", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[56, 57, 58])
    return parser.parse_args()


def load_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def records_by_population(payload: dict) -> dict[str, list[dict]]:
    return {
        population["population"]: population["records"]
        for population in payload["populations"]
    }


def quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {}
    points = np.quantile(values, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    labels = ("min", "p05", "p25", "p50", "p75", "p95", "max")
    return {
        label: float(value) for label, value in zip(labels, points, strict=True)
    }


def rows_equal(first: dict, second: dict) -> bool:
    return (
        int(first["row"]) == int(second["row"])
        and str(first.get("unit_id", "")) == str(second.get("unit_id", ""))
        and str(first.get("image_id", "")) == str(second.get("image_id", ""))
    )


def main() -> None:
    args = parse_args()
    output = {
        "schema_version": 1,
        "definition": {
            "base": "frozen A output in normalized [-1,1] coordinates",
            "ridge_offset": "per-image predicted RGB offset b",
            "oracle_offset": "per-image mean(clean - A) = t",
            "mse_identity": (
                "mean_01((clean01-(A01+x/2))^2) = base01_mse "
                "- dot(x,t)/6 + sum(x^2)/12"
            ),
            "note": (
                "Pre-clipping diagnostic only. Final deployment scores use "
                "the existing clamped evaluation pipeline."
            ),
        },
        "seeds": {},
    }
    csv_rows: list[dict[str, object]] = []

    for seed in args.seeds:
        stats = load_payload(args.eval_dir / f"seed{seed}_stats.json")
        oracle = load_payload(args.eval_dir / f"seed{seed}_oracle.json")
        stats_populations = records_by_population(stats)
        oracle_populations = records_by_population(oracle)
        if stats_populations.keys() != oracle_populations.keys():
            raise ValueError(f"population mismatch for seed {seed}")

        seed_output: dict[str, object] = {"populations": {}}
        for population_name, stats_records in stats_populations.items():
            oracle_records = oracle_populations[population_name]
            if len(stats_records) != len(oracle_records):
                raise ValueError(
                    f"row count mismatch for {seed}/{population_name}"
                )

            direction_dot = []
            ridge_norm_sq = []
            oracle_norm_sq = []
            alpha_unconstrained = []
            alpha_clipped = []
            ridge_gain = []
            scaled_gain = []
            oracle_gain = []
            base_mse = []
            formula_error = []

            for stats_record, oracle_record in zip(
                stats_records, oracle_records, strict=True
            ):
                if not rows_equal(stats_record, oracle_record):
                    raise ValueError(
                        f"unaligned row in {seed}/{population_name}: "
                        f"{stats_record['row']} vs {oracle_record['row']}"
                    )
                ridge = np.asarray(
                    stats_record["predicted_offset_rgb"], dtype=np.float64
                )
                target = np.asarray(
                    oracle_record["predicted_offset_rgb"], dtype=np.float64
                )
                dot = float(np.dot(ridge, target))
                ridge_sq = float(np.dot(ridge, ridge))
                target_sq = float(np.dot(target, target))
                if ridge_sq > 0.0:
                    alpha = dot / ridge_sq
                else:
                    alpha = 0.0
                clipped = float(np.clip(alpha, 0.0, 1.0))

                actual_gain_value = dot / 6.0 - ridge_sq / 12.0
                scaled_gain_value = (
                    dot * dot / (12.0 * ridge_sq)
                    if ridge_sq > 0.0 and dot > 0.0
                    else 0.0
                )
                constrained_gain = (
                    clipped * dot / 6.0
                    - clipped * clipped * ridge_sq / 12.0
                )
                oracle_gain_value = target_sq / 12.0
                oracle_preclip_mse = float(oracle_record["preclip_mse"])
                base_value = oracle_preclip_mse + oracle_gain_value
                predicted_stats_mse = base_value - actual_gain_value
                error = predicted_stats_mse - float(
                    stats_record["preclip_mse"]
                )

                direction_dot.append(dot)
                ridge_norm_sq.append(ridge_sq)
                oracle_norm_sq.append(target_sq)
                alpha_unconstrained.append(alpha)
                alpha_clipped.append(clipped)
                ridge_gain.append(actual_gain_value)
                scaled_gain.append(constrained_gain)
                oracle_gain.append(oracle_gain_value)
                base_mse.append(base_value)
                formula_error.append(error)

                csv_rows.append(
                    {
                        "seed": seed,
                        "population": population_name,
                        "row": int(stats_record["row"]),
                        "unit_id": stats_record.get(
                            "unit_id", stats_record.get("image_id", "")
                        ),
                        "ridge_r": ridge[0],
                        "ridge_g": ridge[1],
                        "ridge_b": ridge[2],
                        "target_r": target[0],
                        "target_g": target[1],
                        "target_b": target[2],
                        "direction_dot": dot,
                        "ridge_norm_sq": ridge_sq,
                        "target_norm_sq": target_sq,
                        "alpha_unconstrained": alpha,
                        "alpha_clipped": clipped,
                        "ridge_preclip_gain": actual_gain_value,
                        "direction_scaled_preclip_gain": constrained_gain,
                        "oracle_preclip_gain": oracle_gain_value,
                        "base_preclip_mse": base_value,
                        "identity_error": error,
                    }
                )

            dot = np.asarray(direction_dot)
            ridge_sq = np.asarray(ridge_norm_sq)
            target_sq = np.asarray(oracle_norm_sq)
            alpha = np.asarray(alpha_unconstrained)
            clipped = np.asarray(alpha_clipped)
            actual = np.asarray(ridge_gain)
            scaled = np.asarray(scaled_gain)
            oracle = np.asarray(oracle_gain)
            errors = np.asarray(formula_error)
            positive = dot > 0.0
            cosine = np.divide(
                dot,
                np.sqrt(ridge_sq * target_sq),
                out=np.zeros_like(dot),
                where=(ridge_sq > 0.0) & (target_sq > 0.0),
            )
            positive_oracle = oracle[oracle > 0.0]
            oracle_total = float(positive_oracle.mean()) if positive_oracle.size else 0.0
            scaled_total = float(scaled[oracle > 0.0].mean()) if oracle_total else 0.0
            actual_total = float(actual[oracle > 0.0].mean()) if oracle_total else 0.0
            direction_ratio = (
                scaled_total / oracle_total if oracle_total > 0.0 else math.nan
            )
            actual_ratio = (
                actual_total / oracle_total if oracle_total > 0.0 else math.nan
            )

            population_output = {
                "rows": len(stats_records),
                "positive_direction_rows": int(positive.sum()),
                "positive_direction_fraction": float(positive.mean()),
                "wrong_or_zero_direction_fraction": float((~positive).mean()),
                "overshoot_fraction_among_positive": float(
                    (alpha[positive] > 1.0).mean()
                )
                if positive.any()
                else 0.0,
                "alpha_unconstrained": quantiles(alpha),
                "alpha_clipped": quantiles(clipped),
                "signed_cosine": quantiles(cosine),
                "cosine_squared_positive": quantiles(np.maximum(cosine, 0.0) ** 2),
                "rows_with_positive_oracle_gain": int((oracle > 0.0).sum()),
                "base_preclip_mse_mean": float(np.mean(base_mse)),
                "oracle_preclip_gain_mean_all_rows": float(oracle.mean()),
                "ridge_preclip_gain_mean_all_rows": float(actual.mean()),
                "direction_scaled_preclip_gain_mean_all_rows": float(scaled.mean()),
                "oracle_preclip_gain_mean_positive_oracle": oracle_total,
                "ridge_preclip_gain_mean_positive_oracle": actual_total,
                "direction_scaled_gain_mean_positive_oracle": scaled_total,
                "actual_ridge_recovery_of_oracle_headroom": actual_ratio,
                "direction_scaled_recovery_of_oracle_headroom": direction_ratio,
                "direction_limitation": max(0.0, 1.0 - direction_ratio)
                if not math.isnan(direction_ratio)
                else math.nan,
                "scale_and_sign_limitation": max(0.0, direction_ratio - actual_ratio)
                if not math.isnan(actual_ratio)
                else math.nan,
                "preclip_identity_max_abs_error": float(np.abs(errors).max()),
            }
            seed_output["populations"][population_name] = population_output

        output["seeds"][str(seed)] = seed_output

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.output_rows.parent.mkdir(parents=True, exist_ok=True)
    with args.output_rows.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
