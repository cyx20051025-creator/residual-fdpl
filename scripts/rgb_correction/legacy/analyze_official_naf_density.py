#!/usr/bin/env python3
"""Summarize the released-NAFNet calibration-density boundary."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_siddcc_budget_curves import summarize, validate


BUDGETS = (4, 8, 12, 16)


def bootstrap_interval(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = np.asarray(
        [
            rng.choice(values, size=len(values), replace=True).mean()
            for _ in range(5000)
        ]
    )
    low, high = np.quantile(draws, (0.025, 0.975))
    return float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "sourcefree_official_naf_density_20261117",
    )
    parser.add_argument(
        "--primary",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "sourcefree_official_naf_20261117"
        / "risk_selector_no_source_naf_official_source56.json",
    )
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    rows = []
    missing = []
    for budget in BUDGETS:
        path = (
            args.primary
            if budget == 16
            else args.root / (
                f"risk_selector_no_source_naf_official_budget{budget}.json"
            )
        )
        if not path.is_file():
            missing.append((budget, str(path)))
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate(payload, path, budget)
        summary = summarize(payload)
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
        low, high = bootstrap_interval(
            selected - fixed,
            seed=20261117 + budget,
        )
        rows.append(
            {
                "budget": budget,
                **summary,
                "selected_minus_fixed_low": low,
                "selected_minus_fixed_high": high,
            }
        )

    if missing and not args.allow_partial:
        raise SystemExit(f"missing released-NAFNet budgets: {missing}")
    if not rows:
        raise SystemExit("no released-NAFNet density outputs found")

    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / "OFFICIAL_NAF_DENSITY.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Released-NAFNet calibration-density boundary",
        "",
        "All rows use the source-free 117-statistic Ridge input, source rule",
        "56, ten fixed 29-scene partitions and the frozen selector.",
        "",
        "| Scenes | Selected ΔPSNR | Fixed target | Selected−fixed [95%] | "
        "Positive | Worst | Actions |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['budget']} | {row['psnr']:+.4f} | "
            f"{row['fixed_target']:+.4f} | "
            f"{row['selected_minus_fixed']:+.4f} "
            f"[{row['selected_minus_fixed_low']:+.4f}, "
            f"{row['selected_minus_fixed_high']:+.4f}] | "
            f"{row['positive_splits']}/10 | {row['worst_split']:+.4f} | "
            f"{row['actions']} |"
        )
    if missing:
        lines.extend(["", f"Not yet available: `{missing}`."])
    lines.extend(
        [
            "",
            "These are exploratory calibration-support diagnostics, not new",
            "operating points. Repeated partitions reuse the same 29 scenes.",
        ]
    )
    (args.root / "OFFICIAL_NAF_DENSITY.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
