#!/usr/bin/env python3
"""Summarize source-free calibration-budget curves on NIND remaining-56."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


BUDGETS = (4, 8, 16, 28, 40)
SEEDS = (62, 63)
METRICS = ("psnr", "ssim", "lpips", "dists")


def mean_metric(records: list[dict], metric: str) -> float:
    return float(
        np.mean(
            [
                record["evaluation"][f"mean_{metric}_delta"]
                for record in records
            ]
        )
    )


def summarize(path: Path, seed: int, budget: int) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["splits"] != 10 or len(payload["records"]) != 10:
        raise ValueError(f"expected ten splits in {path}")
    if int(payload["calibration_scenes"]) != budget:
        raise ValueError(f"budget mismatch in {path}")
    for record in payload["records"]:
        calibration = set(record["calibration_scenes"])
        evaluation = set(record["evaluation_scenes"])
        if calibration & evaluation:
            raise ValueError(f"calibration/evaluation overlap in {path}")
        if len(calibration) != budget:
            raise ValueError(f"calibration-size mismatch in {path}")
        if len(calibration) + len(evaluation) != 56:
            raise ValueError(f"scene coverage mismatch in {path}")

    records = payload["records"]
    selected = np.asarray(
        [record["evaluation"]["mean_psnr_delta"] for record in records],
        dtype=np.float64,
    )
    fixed = np.asarray(
        [
            record["evaluation_by_candidate"]["fixed_target"][
                "mean_psnr_delta"
            ]
            for record in records
        ],
        dtype=np.float64,
    )
    return {
        "seed": seed,
        "budget": budget,
        "selected_psnr": float(selected.mean()),
        "fixed_target": float(fixed.mean()),
        "selected_minus_fixed": float((selected - fixed).mean()),
        "ssim": mean_metric(records, "ssim"),
        "lpips": mean_metric(records, "lpips"),
        "dists": mean_metric(records, "dists"),
        "positive_splits": int((selected > 0.0).sum()),
        "worst_split": float(selected.min()),
        "selected_counts": " / ".join(
            f"{name}:{count}"
            for name, count in payload["selected_counts"].items()
            if count
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "nind_budget_20261117",
    )
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    rows = []
    missing = []
    for seed in SEEDS:
        for budget in BUDGETS:
            path = args.root / f"seed{seed}" / (
                f"risk_selector_no_source_nind_seed{seed}_budget{budget}.json"
            )
            if not path.is_file():
                missing.append((seed, budget))
                continue
            rows.append(summarize(path, seed, budget))

    if missing and not args.allow_partial:
        raise SystemExit(f"missing NIND budget results: {missing}")
    if not rows:
        raise SystemExit("no NIND budget results found")

    args.root.mkdir(parents=True, exist_ok=True)
    csv_path = args.root / "NIND_BUDGET_SUMMARY.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# NIND remaining-56 source-free calibration-budget results",
        "",
        "All rows use the 117-statistic Ridge input, frozen selector",
        "thresholds, ten scene partitions and split seed `20261219`.",
        "",
        "| Seed | Scenes | ΔPSNR | Fixed target | Selected−fixed | "
        "ΔSSIM | ΔLPIPS | ΔDISTS | Positive | Worst split | Actions |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['budget']} | "
            f"{row['selected_psnr']:+.4f} | {row['fixed_target']:+.4f} | "
            f"{row['selected_minus_fixed']:+.4f} | {row['ssim']:+.6f} | "
            f"{row['lpips']:+.6f} | {row['dists']:+.6f} | "
            f"{row['positive_splits']}/10 | {row['worst_split']:+.4f} | "
            f"{row['selected_counts']} |"
        )
    if missing:
        lines.extend(["", f"Not yet available: `{missing}`."])
    lines.extend(
        [
            "",
            "Interpretation guardrail: NIND remaining-56 appeared in earlier",
            "project work and is a pressure test, not a blind population.",
            "Repeated splits reuse the same scenes. At 40 calibration scenes",
            "only 16 outer-evaluation scenes remain, so that point is a",
            "sensitivity diagnostic rather than a new operating point.",
        ]
    )
    (args.root / "NIND_BUDGET_SUMMARY.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
