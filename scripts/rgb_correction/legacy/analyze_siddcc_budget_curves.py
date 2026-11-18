#!/usr/bin/env python3
"""Summarize SIDD-CC source-free calibration-budget curves."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


BUDGETS = (4, 8, 12, 16, 20, 24)
SEEDS = (62, 63, 64)
METRICS = ("psnr", "ssim", "lpips", "dists")


def sourcefree_path(
    root: Path,
    prior_root: Path,
    seed: int,
    budget: int,
) -> Path:
    if budget == 16:
        return prior_root / (
            f"risk_selector_no_source_naf_finetuned_seed{seed}_source56.json"
        )
    return root / f"seed{seed}" / (
        f"risk_selector_no_source_siddcc_seed{seed}_budget{budget}.json"
    )


def original_path(
    release: Path,
    seed: int,
    budget: int,
) -> Path:
    filename = (
        f"risk_selector_naf_finetuned_seed{seed}_source56_"
        "fresh20261216_full.json"
    )
    if budget == 16:
        return release / "supplementary_results" / filename
    return (
        release
        / "calibration_budget_finetuned"
        / f"calibration{budget}"
        / filename
    )


def validate(payload: dict, path: Path, budget: int) -> None:
    if payload["splits"] != 10 or len(payload["records"]) != 10:
        raise ValueError(f"expected ten splits in {path}")
    if payload["calibration_scenes"] != budget:
        raise ValueError(f"budget mismatch in {path}")
    for record in payload["records"]:
        calibration = set(record["calibration_scenes"])
        evaluation = set(record["evaluation_scenes"])
        if calibration & evaluation:
            raise ValueError(f"partition overlap in {path}")
        if len(calibration) != budget or len(calibration) + len(evaluation) != 29:
            raise ValueError(f"partition-size mismatch in {path}")


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


def metric_mean(records: list[dict], metric: str) -> float:
    return float(
        np.mean(
            [
                record["evaluation"][f"mean_{metric}_delta"]
                for record in records
            ]
        )
    )


def summarize(payload: dict) -> dict:
    records = payload["records"]
    selected = np.asarray(
        [record["evaluation"]["mean_psnr_delta"] for record in records]
    )
    fixed = np.asarray(
        [
            record["evaluation_by_candidate"]["fixed_target"][
                "mean_psnr_delta"
            ]
            for record in records
        ]
    )
    return {
        "psnr": float(selected.mean()),
        "fixed_target": float(fixed.mean()),
        "selected_minus_fixed": float((selected - fixed).mean()),
        "ssim": metric_mean(records, "ssim"),
        "lpips": metric_mean(records, "lpips"),
        "dists": metric_mean(records, "dists"),
        "positive_splits": int((selected > 0.0).sum()),
        "worst_split": float(selected.min()),
        "actions": " / ".join(
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
        / "selector_no_source_siddcc_budget_20261117",
    )
    parser.add_argument(
        "--prior-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "selector_no_source_20261117"
        / "results",
    )
    parser.add_argument(
        "--release",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "target_endpoint_calibration_20261116"
        / "risk_selector_release_full_20261117",
    )
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    rows = []
    missing = []
    for seed in SEEDS:
        for budget in BUDGETS:
            sf_path = sourcefree_path(args.root, args.prior_root, seed, budget)
            if not sf_path.is_file():
                missing.append((seed, budget))
                continue
            sourcefree = json.loads(sf_path.read_text(encoding="utf-8"))
            validate(sourcefree, sf_path, budget)
            row = {
                "seed": seed,
                "budget": budget,
                **{
                    f"sourcefree_{key}": value
                    for key, value in summarize(sourcefree).items()
                },
            }
            orig_path = original_path(args.release, seed, budget)
            if orig_path.is_file():
                original = json.loads(orig_path.read_text(encoding="utf-8"))
                validate(original, orig_path, budget)
                original_summary = summarize(original)
                differences = np.asarray(
                    [
                        rb["evaluation"]["mean_psnr_delta"]
                        - ra["evaluation"]["mean_psnr_delta"]
                        for ra, rb in zip(
                            original["records"],
                            sourcefree["records"],
                            strict=True,
                        )
                    ]
                )
                low, high = bootstrap_interval(
                    differences,
                    seed=20261216 + 1000 * seed + budget,
                )
                changed = []
                for index, (ra, rb) in enumerate(
                    zip(
                        original["records"],
                        sourcefree["records"],
                        strict=True,
                    ),
                    start=1,
                ):
                    before = ra["decision"]["selected"]
                    after = rb["decision"]["selected"]
                    if before != after:
                        changed.append(f"{index}:{before}->{after}")
                row.update(
                    {
                        "original_psnr": original_summary["psnr"],
                        "paired_difference": float(differences.mean()),
                        "paired_ci_low": low,
                        "paired_ci_high": high,
                        "identical_actions": int(
                            sum(
                                ra["decision"]["selected"]
                                == rb["decision"]["selected"]
                                for ra, rb in zip(
                                    original["records"],
                                    sourcefree["records"],
                                    strict=True,
                                )
                            )
                        ),
                        "changed_actions": "; ".join(changed),
                    }
                )
            rows.append(row)

    if missing and not args.allow_partial:
        raise SystemExit(f"missing SIDD-CC source-free budgets: {missing}")
    if not rows:
        raise SystemExit("no SIDD-CC source-free results found")

    fields = sorted({key for row in rows for key in row})
    with (args.root / "SIDDCC_BUDGET_CURVES.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# SIDD-CC source-free calibration-budget results",
        "",
        "The selector and data are unchanged; only the three source columns in",
        "the Ridge candidates are zero. Matched original-input results exist for",
        "4/8/12/16 scenes.",
        "",
        "| Seed | Scenes | ΔPSNR | Fixed target | Selected−fixed | "
        "ΔSSIM | ΔLPIPS | ΔDISTS | Positive | Worst | Actions | "
        "Original | Paired diff [95%] | Action agreement |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        if "original_psnr" in row:
            comparison = (
                f"{row['original_psnr']:+.4f} | "
                f"{row['paired_difference']:+.4f} "
                f"[{row['paired_ci_low']:+.4f}, "
                f"{row['paired_ci_high']:+.4f}] | "
                f"{row['identical_actions']}/10"
            )
        else:
            comparison = "— | — | —"
        lines.append(
            f"| {row['seed']} | {row['budget']} | "
            f"{row['sourcefree_psnr']:+.4f} | "
            f"{row['sourcefree_fixed_target']:+.4f} | "
            f"{row['sourcefree_selected_minus_fixed']:+.4f} | "
            f"{row['sourcefree_ssim']:+.6f} | "
            f"{row['sourcefree_lpips']:+.6f} | "
            f"{row['sourcefree_dists']:+.6f} | "
            f"{row['sourcefree_positive_splits']}/10 | "
            f"{row['sourcefree_worst_split']:+.4f} | "
            f"{row['sourcefree_actions']} | {comparison} |"
        )
    if missing:
        lines.extend(["", f"Not yet available: `{missing}`."])
    lines.extend(
        [
            "",
            "The 20/24-scene points have no matched original-input selector",
            "runs and therefore extend only the source-free curve. Repeated",
            "splits reuse the 29 scenes and are not independent populations.",
        ]
    )
    (args.root / "SIDDCC_BUDGET_CURVES.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
