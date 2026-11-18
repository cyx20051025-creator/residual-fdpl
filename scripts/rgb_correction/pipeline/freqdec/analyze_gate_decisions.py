#!/usr/bin/env python3
"""Measure whether the frozen gate identifies unsafe RGB corrections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-rows", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[56, 57, 58])
    return parser.parse_args()


def rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def correlation(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    first_rank = rank(first)
    second_rank = rank(second)
    first_rank -= first_rank.mean()
    second_rank -= second_rank.mean()
    denominator = np.sqrt(
        np.sum(first_rank * first_rank)
        * np.sum(second_rank * second_rank)
    )
    spearman = float(np.sum(first_rank * second_rank) / denominator) if denominator else 0.0
    pearson = float(np.corrcoef(first, second)[0, 1]) if first.std() > 0 and second.std() > 0 else 0.0
    return spearman, pearson


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = scores[labels]
    negatives = scores[~labels]
    if positives.size == 0 or negatives.size == 0:
        return float("nan")
    ranks = rank(scores) + 1.0
    rank_sum = ranks[labels].sum()
    return float(
        (
            rank_sum
            - positives.size * (positives.size + 1) / 2.0
        )
        / (positives.size * negatives.size)
    )


def population_records(payload: dict) -> dict[str, list[dict]]:
    return {
        item["population"]: item["records"]
        for item in payload["populations"]
    }


def main() -> None:
    args = parse_args()
    output = {"seeds": {}}
    csv_rows: list[dict] = []
    for seed in args.seeds:
        candidate = json.loads(
            (args.result_dir / f"seed{seed}_tree.json").read_text(
                encoding="utf-8"
            )
        )
        oracle = json.loads(
            (args.oracle_dir / f"seed{seed}_oracle.json").read_text(
                encoding="utf-8"
            )
        )
        candidate_pops = population_records(candidate)
        oracle_pops = population_records(oracle)
        seed_output = {}
        for population_name, records in candidate_pops.items():
            oracle_records = oracle_pops[population_name]
            if len(records) != len(oracle_records):
                raise ValueError(f"row mismatch {seed}/{population_name}")
            gate_alpha = []
            ridge_norm = []
            dot = []
            ideal_alpha = []
            correct_direction = []
            base_psnr = []
            candidate_psnr = []
            for row, oracle_row in zip(records, oracle_records, strict=True):
                if int(row["row"]) != int(oracle_row["row"]):
                    raise ValueError("unaligned result rows")
                offset = np.asarray(row["ridge_offset_rgb"], dtype=np.float64)
                target = np.asarray(
                    oracle_row["predicted_offset_rgb"], dtype=np.float64
                )
                dot_value = float(np.dot(offset, target))
                norm_sq = float(np.dot(offset, offset))
                ideal = float(np.clip(dot_value / max(norm_sq, 1e-12), 0.0, 1.0))
                gate = float(row["alpha"])
                gate_alpha.append(gate)
                ridge_norm.append(float(np.linalg.norm(offset)))
                dot.append(dot_value)
                ideal_alpha.append(ideal)
                correct_direction.append(dot_value > 0.0)
                base_psnr.append(float(row["base_psnr"]))
                candidate_psnr.append(float(row["candidate_psnr"]))
                csv_rows.append(
                    {
                        "seed": seed,
                        "population": population_name,
                        "row": int(row["row"]),
                        "unit_id": row.get("unit_id", row.get("image_id", "")),
                        "gate_alpha": gate,
                        "ideal_alpha": ideal,
                        "direction_dot": dot_value,
                        "ridge_norm": ridge_norm[-1],
                        "correct_direction": correct_direction[-1],
                        "base_psnr": base_psnr[-1],
                        "candidate_psnr": candidate_psnr[-1],
                        "psnr_delta": candidate_psnr[-1] - base_psnr[-1],
                    }
                )
            gate = np.asarray(gate_alpha)
            ideal = np.asarray(ideal_alpha)
            norm = np.asarray(ridge_norm)
            correct = np.asarray(correct_direction)
            delta = np.asarray(candidate_psnr) - np.asarray(base_psnr)
            ideal_spearman, ideal_pearson = correlation(gate, ideal)
            norm_spearman, norm_pearson = correlation(gate, norm)
            seed_output[population_name] = {
                "rows": len(gate),
                "correct_direction_fraction": float(correct.mean()),
                "gate_alpha_all": {
                    "mean": float(gate.mean()),
                    "median": float(np.median(gate)),
                    "p05": float(np.quantile(gate, 0.05)),
                    "p95": float(np.quantile(gate, 0.95)),
                },
                "gate_alpha_correct_direction": {
                    "mean": float(gate[correct].mean())
                    if correct.any()
                    else 0.0,
                    "median": float(np.median(gate[correct]))
                    if correct.any()
                    else 0.0,
                },
                "gate_alpha_wrong_direction": {
                    "mean": float(gate[~correct].mean())
                    if (~correct).any()
                    else 0.0,
                    "median": float(np.median(gate[~correct]))
                    if (~correct).any()
                    else 0.0,
                },
                "correct_direction_auc_from_gate_alpha": auc(correct, gate),
                "gate_vs_ideal_alpha_spearman": ideal_spearman,
                "gate_vs_ideal_alpha_pearson": ideal_pearson,
                "gate_vs_ridge_norm_spearman": norm_spearman,
                "gate_vs_ridge_norm_pearson": norm_pearson,
                "alpha_zero_thresholds": {
                    str(threshold): float((gate <= threshold).mean())
                    for threshold in (0.05, 0.10, 0.20, 0.30, 0.50)
                },
                "ideal_alpha_zero_fraction": float((ideal == 0.0).mean()),
                "wrong_direction_gate_le_0_10_fraction": float(
                    (gate[~correct] <= 0.10).mean()
                )
                if (~correct).any()
                else 0.0,
                "wrong_direction_gate_le_0_20_fraction": float(
                    (gate[~correct] <= 0.20).mean()
                )
                if (~correct).any()
                else 0.0,
                "psnr_delta": {
                    "mean": float(delta.mean()),
                    "positive_fraction": float((delta > 0.0).mean()),
                    "worst": float(delta.min()),
                    "p05": float(np.quantile(delta, 0.05)),
                    "best": float(delta.max()),
                },
            }
        output["seeds"][str(seed)] = seed_output
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    import csv

    args.output_rows.parent.mkdir(parents=True, exist_ok=True)
    with args.output_rows.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
