#!/usr/bin/env python3
"""Summarize the frozen cross-backbone calibration-budget curves."""

from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np


BUDGETS = (4, 8, 12, 16, 20, 24)


def mean_metric(records: list[dict], metric: str) -> float:
    return float(
        np.mean(
            [
                record["evaluation"][f"mean_{metric}_delta"]
                for record in records
            ]
        )
    )


def summarize(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["splits"] != 10 or len(payload["records"]) != 10:
        raise ValueError(f"expected ten splits in {path}")
    budget = int(payload["calibration_scenes"])
    for record in payload["records"]:
        calibration = set(record["calibration_scenes"])
        evaluation = set(record["evaluation_scenes"])
        if calibration & evaluation:
            raise ValueError(f"calibration/evaluation overlap in {path}")
        if len(calibration) != budget:
            raise ValueError(f"calibration-size mismatch in {path}")
        if len(calibration) + len(evaluation) != 29:
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
    difference = selected - fixed
    return {
        "endpoint": path.parent.name,
        "budget": budget,
        "psnr": float(selected.mean()),
        "ssim": mean_metric(records, "ssim"),
        "lpips": mean_metric(records, "lpips"),
        "dists": mean_metric(records, "dists"),
        "fixed_target": float(fixed.mean()),
        "selected_minus_fixed": float(difference.mean()),
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
        / "crossbackbone_budget_20261117",
    )
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    paths = [
        Path(path)
        for path in glob.glob(
            str(args.root / "*" / "risk_selector_no_source_*_budget*.json")
        )
    ]
    rows = sorted(
        (summarize(path) for path in paths),
        key=lambda row: (row["endpoint"], row["budget"]),
    )
    present = {(row["endpoint"], row["budget"]) for row in rows}
    expected = {
        ("swinir_seed62", budget) for budget in BUDGETS
    } | {
        ("restormer", budget) for budget in BUDGETS
    }
    missing = sorted(expected - present)
    if missing and not args.allow_partial:
        raise SystemExit(f"missing cross-backbone results: {missing}")
    if not rows:
        raise SystemExit("no cross-backbone results found")

    csv_path = args.root / "CROSSBACKBONE_BUDGET_SUMMARY.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Cross-backbone SIDD-CC calibration-budget results",
        "",
        "All rows use the 117-feature-only Ridge input, the frozen selector",
        "thresholds, ten scene partitions and split seed `20261216`.",
        "",
        "| Endpoint | Scenes | ΔPSNR | Fixed target | Selected−fixed | "
        "ΔSSIM | ΔLPIPS | ΔDISTS | Positive | Worst split | Actions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['endpoint']} | {row['budget']} | "
            f"{row['psnr']:+.4f} | {row['fixed_target']:+.4f} | "
            f"{row['selected_minus_fixed']:+.4f} | "
            f"{row['ssim']:+.6f} | {row['lpips']:+.6f} | "
            f"{row['dists']:+.6f} | {row['positive_splits']}/10 | "
            f"{row['worst_split']:+.4f} | {row['selected_counts']} |"
        )
    if missing:
        lines.extend(["", f"Not yet available: `{missing}`."])
    lines.extend(
        [
            "",
            "Interpretation guardrail: both architectures are evaluated on the",
            "same 29 reused scenes. Repeated splits and calibration-budget draws",
            "are not independent populations, and the two backbones are not",
            "independent training seeds of one architecture.",
        ]
    )
    (args.root / "CROSSBACKBONE_BUDGET_SUMMARY.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
