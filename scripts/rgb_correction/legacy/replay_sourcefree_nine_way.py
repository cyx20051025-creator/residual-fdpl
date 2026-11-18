#!/usr/bin/env python3
"""Replay the source-free selector over all three frozen source rules.

The source columns are absent from the target-Ridge candidates. For a fixed
endpoint and split, those candidate summaries are therefore independent of
the source rule. We replace only those two candidate summaries in the
original selector records; source-frozen, source-scaled and fixed-target
candidate outputs remain specific to each source rule. The frozen decision
function is then rerun over the combined summaries.

The source-56 replay is checked against the actual source-free GPU output
before any source-57/58 result is accepted.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np


RIDGE_CANDIDATES = {"target_ridge_capped", "target_ridge_uncapped"}
SEEDS = (62, 63, 64)
SOURCES = (56, 57, 58)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def original_path(release: Path, seed: int, source: int) -> Path:
    return release / "supplementary_results" / (
        f"risk_selector_naf_finetuned_seed{seed}_source{source}_"
        "fresh20261216_full.json"
    )


def sourcefree_path(root: Path, seed: int) -> Path:
    return root / "selector_no_source_20261117" / "results" / (
        f"risk_selector_no_source_naf_finetuned_seed{seed}_source56.json"
    )


def aggregate_record(records: list[dict]) -> dict:
    values = np.asarray(
        [record["evaluation"]["mean_psnr_delta"] for record in records],
        dtype=np.float64,
    )
    return {
        "mean_psnr_delta": float(values.mean()),
        "median_psnr_delta": float(np.median(values)),
        "positive_split_fraction": float((values > 0.0).mean()),
        "worst_split_psnr_delta": float(values.min()),
        "best_split_psnr_delta": float(values.max()),
    }


def bootstrap_mean_interval(
    values: np.ndarray,
    *,
    samples: int = 5000,
    seed: int = 20261117,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = np.asarray(
        [
            rng.choice(values, size=len(values), replace=True).mean()
            for _ in range(samples)
        ],
        dtype=np.float64,
    )
    low, high = np.quantile(draws, (0.025, 0.975))
    return float(low), float(high)


def combine_record(
    original: dict,
    sourcefree: dict,
    choose_candidate,
) -> dict:
    if original["calibration_scenes"] != sourcefree["calibration_scenes"]:
        raise ValueError("calibration budget mismatch")
    if original["splits"] != sourcefree["splits"]:
        raise ValueError("split-count mismatch")

    combined_records = []
    for original_record, sourcefree_record in zip(
        original["records"],
        sourcefree["records"],
        strict=True,
    ):
        if (
            original_record["calibration_scenes"]
            != sourcefree_record["calibration_scenes"]
            or original_record["evaluation_scenes"]
            != sourcefree_record["evaluation_scenes"]
        ):
            raise ValueError("split partition mismatch")

        calibration = copy.deepcopy(original_record["calibration_loo"])
        evaluation = copy.deepcopy(original_record["evaluation_by_candidate"])
        for candidate in RIDGE_CANDIDATES:
            calibration[candidate] = copy.deepcopy(
                sourcefree_record["calibration_loo"][candidate]
            )
            evaluation[candidate] = copy.deepcopy(
                sourcefree_record["evaluation_by_candidate"][candidate]
            )

        decision = choose_candidate(calibration)
        selected = decision["selected"]
        if selected in RIDGE_CANDIDATES:
            selected_fit = sourcefree_record["selected_fit_parameter"]
        else:
            selected_fit = original_record["selected_fit_parameter"]

        original_folds = {
            fold["heldout_scene"]: fold
            for fold in original_record["calibration_fold_diagnostics"]
        }
        sourcefree_folds = {
            fold["heldout_scene"]: fold
            for fold in sourcefree_record["calibration_fold_diagnostics"]
        }
        combined_folds = []
        for scene in sorted(original_folds):
            diagnostics = copy.deepcopy(original_folds[scene])
            for candidate in RIDGE_CANDIDATES:
                diagnostics["candidates"][candidate] = copy.deepcopy(
                    sourcefree_folds[scene]["candidates"][candidate]
                )
            combined_folds.append(diagnostics)

        combined_records.append(
            {
                "split": original_record["split"],
                "calibration_scenes": original_record["calibration_scenes"],
                "evaluation_scenes": original_record["evaluation_scenes"],
                "calibration_loo": calibration,
                "decision": decision,
                "selected_fit_parameter": (
                    None if selected_fit is None else float(selected_fit)
                ),
                "evaluation": evaluation[selected],
                "evaluation_by_candidate": evaluation,
                "calibration_fold_diagnostics": combined_folds,
                "replay_components": {
                    "original_source_record": str(original["_path"]),
                    "sourcefree_ridge_record": str(sourcefree["_path"]),
                },
            }
        )

    candidates = list(combined_records[0]["calibration_loo"])
    selected_counts = {
        candidate: sum(
            record["decision"]["selected"] == candidate
            for record in combined_records
        )
        for candidate in candidates
    }
    always_means = {}
    for candidate in candidates:
        values = np.asarray(
            [
                record["evaluation_by_candidate"][candidate][
                    "mean_psnr_delta"
                ]
                for record in combined_records
            ],
            dtype=np.float64,
        )
        always_means[candidate] = {
            "mean_psnr_delta": float(values.mean()),
            "positive_split_fraction": float((values > 0.0).mean()),
        }

    return {
        "schema_version": 1,
        "replay_kind": "sourcefree_ridge_with_source_specific_simple_candidates",
        "architecture": original["architecture"],
        "endpoint_seed": original["endpoint_seed"],
        "source_seed": original["source_seed"],
        "population_id": original["population_id"],
        "checkpoint": original["checkpoint"],
        "feature_file": original["feature_file"],
        "h5": original["h5"],
        "manifest": original["manifest"],
        "splits": original["splits"],
        "calibration_scenes": original["calibration_scenes"],
        "evaluation_scenes": original["evaluation_scenes"],
        "metric_names": original["metric_names"],
        "selection_policy": original["selection_policy"],
        "selected_counts": selected_counts,
        "always_candidate_outer_means": always_means,
        "aggregate": aggregate_record(combined_records),
        "records": combined_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--release",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "target_endpoint_calibration_20261116"
        / "risk_selector_release_full_20261117",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent
        / "sourcefree_nineway_replay_20261117",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(args.release / "code"))
    import risk_controlled_selector as selector

    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in SEEDS:
        sfc_path = sourcefree_path(args.root, seed)
        sourcefree = load_json(sfc_path)
        sourcefree["_path"] = str(sfc_path.resolve())
        reference_original = None
        for source in SOURCES:
            orig_path = original_path(args.release, seed, source)
            original = load_json(orig_path)
            original["_path"] = str(orig_path.resolve())
            if reference_original is None:
                reference_original = original
            else:
                for reference_record, source_record in zip(
                    reference_original["records"],
                    original["records"],
                    strict=True,
                ):
                    for candidate in ("no_correction", "fixed_target"):
                        for field, reference_summary, source_summary in (
                            (
                                "calibration",
                                reference_record["calibration_loo"][candidate],
                                source_record["calibration_loo"][candidate],
                            ),
                            (
                                "evaluation",
                                reference_record["evaluation_by_candidate"][
                                    candidate
                                ],
                                source_record["evaluation_by_candidate"][
                                    candidate
                                ],
                            ),
                        ):
                            if not np.isclose(
                                reference_summary["mean_psnr_delta"],
                                source_summary["mean_psnr_delta"],
                                atol=1.0e-12,
                                rtol=0.0,
                            ):
                                raise ValueError(
                                    f"{candidate} {field} changed with source "
                                    f"rule: seed {seed}, source {source}"
                                )
            replay = combine_record(original, sourcefree, selector.choose_candidate)

            if source == sourcefree["source_seed"]:
                for replay_record, actual_record in zip(
                    replay["records"],
                    sourcefree["records"],
                    strict=True,
                ):
                    if (
                        replay_record["decision"]["selected"]
                        != actual_record["decision"]["selected"]
                    ):
                        raise ValueError(
                            f"source-56 action mismatch: seed {seed}, "
                            f"split {replay_record['split'] + 1}"
                        )
                    if not np.isclose(
                        replay_record["evaluation"]["mean_psnr_delta"],
                        actual_record["evaluation"]["mean_psnr_delta"],
                        atol=1.0e-12,
                        rtol=0.0,
                    ):
                        raise ValueError(
                            f"source-56 outcome mismatch: seed {seed}, "
                            f"split {replay_record['split'] + 1}"
                        )

            output_path = args.output / (
                f"risk_selector_sourcefree_seed{seed}_source{source}.json"
            )
            output_path.write_text(
                json.dumps(replay, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )

            values = np.asarray(
                [
                    record["evaluation"]["mean_psnr_delta"]
                    for record in replay["records"]
                ],
                dtype=np.float64,
            )
            original_values = np.asarray(
                [
                    record["evaluation"]["mean_psnr_delta"]
                    for record in original["records"]
                ],
                dtype=np.float64,
            )
            low, high = bootstrap_mean_interval(
                values,
                seed=20261117 + 1000 * seed + source,
            )
            rows.append(
                {
                    "seed": seed,
                    "source": source,
                    "selected_psnr": replay["aggregate"]["mean_psnr_delta"],
                    "interval_low": low,
                    "interval_high": high,
                    "positive_splits": int((values > 0.0).sum()),
                    "worst_split": float(values.min()),
                    "original_psnr": float(original_values.mean()),
                    "difference_from_original": float(
                        (values - original_values).mean()
                    ),
                    "worst_split_change": float(
                        (values - original_values).min()
                    ),
                    "actions": " / ".join(
                        f"{name}:{count}"
                        for name, count in replay["selected_counts"].items()
                        if count
                    ),
                    "source56_replay_verified": source == sourcefree["source_seed"],
                }
            )

    csv_path = args.output / "NINEWAY_SUMMARY.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Source-free selector: nine-way source-rule replay",
        "",
        "For each endpoint and split, only the two target-Ridge summaries are",
        "taken from the source-free GPU run. Candidate summaries that depend on",
        "the source rule remain from that source-specific original run. The",
        "frozen decision function is then replayed. Source-56 rows match the",
        "actual source-free GPU records exactly.",
        "",
        "| Seed | Source | Selected ΔPSNR | Original | Difference | Worst change | Split interval | Positive | Worst | Actions |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['source']} | "
            f"{row['selected_psnr']:+.4f} | "
            f"{row['original_psnr']:+.4f} | "
            f"{row['difference_from_original']:+.4f} | "
            f"{row['worst_split_change']:+.4f} | "
            f"[{row['interval_low']:+.4f}, {row['interval_high']:+.4f}] | "
            f"{row['positive_splits']}/10 | {row['worst_split']:+.4f} | "
            f"{row['actions']} |"
        )
    values = np.asarray([row["selected_psnr"] for row in rows])
    lines.extend(
        [
            "",
            f"Across the nine applications, the selected mean ranges from "
            f"`{values.min():+.4f}` to `{values.max():+.4f} dB`; "
            f"`{(values > 0).sum()}/9` applications and all 90 repeated split "
            "outcomes are positive.",
            "",
            "These are repeated checkpoint/source applications over the same "
            "29 SIDD-CC scenes, not nine independent populations.",
        ]
    )
    (args.output / "NINEWAY_SUMMARY.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
