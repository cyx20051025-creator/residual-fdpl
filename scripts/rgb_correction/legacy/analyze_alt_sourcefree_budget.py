#!/usr/bin/env python3
"""Validate and summarize the alternate-partition SIDD+ budget sweep."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


BUDGETS = (16, 64, 128, 256, 512)
SEEDS = (62, 63)
METRICS = ("psnr", "ssim", "lpips", "dists")


def payload_path(root: Path, seed: int, budget: int) -> Path:
    return root / f"sourcefree_alt20261227_seed{seed}" / (
        f"risk_selector_no_source_siddplus_seed{seed}_budget{budget}.json"
    )


def primary_path(root: Path, seed: int, budget: int) -> Path:
    return root / f"sourcefree_core_seed{seed}" / (
        f"risk_selector_no_source_siddplus_seed{seed}_budget{budget}.json"
    )


def validate(payload: dict, path: Path, seed: int, budget: int) -> None:
    if payload["endpoint_seed"] != seed:
        raise ValueError(f"endpoint seed mismatch in {path}")
    if int(payload["calibration_scenes"]) != budget:
        raise ValueError(f"calibration budget mismatch in {path}")
    if payload["splits"] != 10 or len(payload["records"]) != 10:
        raise ValueError(f"expected ten records in {path}")
    for record in payload["records"]:
        calibration = set(record["calibration_scenes"])
        evaluation = set(record["evaluation_scenes"])
        if calibration & evaluation:
            raise ValueError(f"calibration/evaluation overlap in {path}")
        if len(calibration) != budget:
            raise ValueError(f"calibration partition mismatch in {path}")
        if len(calibration) + len(evaluation) != 1024:
            raise ValueError(f"partition coverage mismatch in {path}")


def metric_mean(payload: dict, metric: str) -> float:
    return float(
        np.mean(
            [
                record["evaluation"][f"mean_{metric}_delta"]
                for record in payload["records"]
            ]
        )
    )


def summarize(path: Path, seed: int, budget: int) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate(payload, path, seed, budget)
    selected = np.asarray(
        [
            record["evaluation"]["mean_psnr_delta"]
            for record in payload["records"]
        ],
        dtype=np.float64,
    )
    fixed = np.asarray(
        [
            record["evaluation_by_candidate"]["fixed_target"][
                "mean_psnr_delta"
            ]
            for record in payload["records"]
        ],
        dtype=np.float64,
    )
    return {
        "seed": seed,
        "budget": budget,
        "selected_psnr": float(selected.mean()),
        "fixed_target": float(fixed.mean()),
        "selected_minus_fixed": float((selected - fixed).mean()),
        "ssim": metric_mean(payload, "ssim"),
        "lpips": metric_mean(payload, "lpips"),
        "dists": metric_mean(payload, "dists"),
        "positive_splits": int((selected > 0.0).sum()),
        "worst_split": float(selected.min()),
        "actions": " / ".join(
            f"{name}:{count}"
            for name, count in payload["selected_counts"].items()
            if count
        ),
    }


def partition_comparison(alt: dict, primary: dict) -> dict:
    if len(alt["records"]) != len(primary["records"]):
        raise ValueError("split-count mismatch")
    calibration_overlap = []
    evaluation_overlap = []
    for alt_record, primary_record in zip(
        alt["records"],
        primary["records"],
        strict=True,
    ):
        calibration_overlap.append(
            len(
                set(alt_record["calibration_scenes"])
                & set(primary_record["calibration_scenes"])
            )
            / len(alt_record["calibration_scenes"])
        )
        evaluation_overlap.append(
            len(
                set(alt_record["evaluation_scenes"])
                & set(primary_record["evaluation_scenes"])
            )
            / len(alt_record["evaluation_scenes"])
        )
    return {
        "mean_calibration_overlap_with_primary": float(
            np.mean(calibration_overlap)
        ),
        "mean_evaluation_overlap_with_primary": float(
            np.mean(evaluation_overlap)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "siddplus_calibration_budget_20261117",
    )
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    rows = []
    missing = []
    for seed in SEEDS:
        for budget in BUDGETS:
            alt_path = payload_path(args.root, seed, budget)
            if not alt_path.is_file():
                missing.append((seed, budget))
                continue
            row = summarize(alt_path, seed, budget)
            primary_path_value = primary_path(args.root, seed, budget)
            if primary_path_value.is_file():
                alt = json.loads(alt_path.read_text(encoding="utf-8"))
                primary = json.loads(
                    primary_path_value.read_text(encoding="utf-8")
                )
                validate(primary, primary_path_value, seed, budget)
                row["primary_selected_psnr"] = metric_mean(primary, "psnr")
                row.update(partition_comparison(alt, primary))
            rows.append(row)

    if missing and not args.allow_partial:
        raise SystemExit(f"missing alternate-split results: {missing}")
    if not rows:
        raise SystemExit("no alternate-split results found")

    csv_path = args.root / "ALT_SOURCEFREE_BUDGET_SUMMARY.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=sorted({key for row in rows for key in row}),
        )
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Alternate-partition SIDD+ budget sweep",
        "",
        "All rows use split seed `20261227`, the 117-statistic Ridge input,",
        "source predictor 56, ten fixed image-level leaves and frozen selector",
        "thresholds. The overlap columns compare each alternate split with the",
        "same-index primary split and confirm that the partitions are not",
        "being treated as independent populations.",
        "",
        "| Seed | Images | ΔPSNR | Fixed target | Selected−fixed | "
        "Positive | Worst | ΔSSIM | ΔLPIPS | ΔDISTS | Primary overlap | Actions |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        overlap = (
            f"{row['mean_evaluation_overlap_with_primary']:.0%}"
            if "mean_evaluation_overlap_with_primary" in row
            else "—"
        )
        lines.append(
            f"| {row['seed']} | {row['budget']} | "
            f"{row['selected_psnr']:+.4f} | {row['fixed_target']:+.4f} | "
            f"{row['selected_minus_fixed']:+.4f} | "
            f"{row['positive_splits']}/10 | {row['worst_split']:+.4f} | "
            f"{row['ssim']:+.6f} | {row['lpips']:+.6f} | "
            f"{row['dists']:+.6f} | {overlap} | {row['actions']} |"
        )
    if missing:
        lines.extend(["", f"Not yet available: `{missing}`."])
    lines.extend(
        [
            "",
            "Interpretation guardrail: these rows test partition stability,",
            "not a new population. They reuse the same 1,024 SIDD+ images.",
        ]
    )
    output_path = args.root / "ALT_SOURCEFREE_BUDGET_SUMMARY.md"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
