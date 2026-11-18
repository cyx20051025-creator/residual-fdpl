#!/usr/bin/env python3
"""Replay the three 3-RCAB source rules with source-free Ridge candidates."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_sourcefree_nine_way import (  # noqa: E402
    bootstrap_mean_interval,
    combine_record,
    load_json,
)


SEEDS = (56, 57, 58)
SOURCES = (56, 57, 58)


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
        "--candidate-root",
        type=Path,
        default=Path(__file__).resolve().parent
        / "sourcefree_a_20261117",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent
        / "sourcefree_a_replay_20261117",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(args.release / "code"))
    import risk_controlled_selector as selector

    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in SEEDS:
        candidate_path = args.candidate_root / (
            f"ridge_candidates_a_seed{seed}.json"
        )
        candidates = load_json(candidate_path)
        candidates["_path"] = str(candidate_path.resolve())
        for source in SOURCES:
            original_path = args.release / "results" / (
                f"risk_selector_seed{seed}_fresh20261216_full.json"
            )
            original = load_json(original_path)
            original["_path"] = str(original_path.resolve())
            replay = combine_record(
                original,
                candidates,
                selector.choose_candidate,
            )
            replay["replay_kind"] = (
                "A source-specific simple candidates plus 117-only Ridge"
            )
            replay["endpoint_seed"] = seed
            output_path = args.output / (
                f"risk_selector_sourcefree_a_seed{seed}_source{source}.json"
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
                    "checkpoint_seed": seed,
                    "source": source,
                    "selected_psnr": float(values.mean()),
                    "interval_low": low,
                    "interval_high": high,
                    "positive_splits": int((values > 0.0).sum()),
                    "worst_split": float(values.min()),
                    "original_psnr": float(original_values.mean()),
                    "difference_from_original": float(
                        (values - original_values).mean()
                    ),
                    "actions": " / ".join(
                        f"{name}:{count}"
                        for name, count in replay["selected_counts"].items()
                        if count
                    ),
                }
            )

    csv_path = args.output / "A_SOURCEFREE_SUMMARY.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Source-free selector on 3-RCAB",
        "",
        "The source-only, scaled-source and fixed-target candidate summaries",
        "come from each original source-specific run. The two target-Ridge",
        "summaries come from a GPU evaluation with zero source offsets, which",
        "is valid because the 117-only Ridge map does not use those offsets.",
        "",
        "| Endpoint | Source | Selected ΔPSNR | Original | Difference | Positive | Worst | Actions |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['checkpoint_seed']} | {row['source']} | "
            f"{row['selected_psnr']:+.4f} | {row['original_psnr']:+.4f} | "
            f"{row['difference_from_original']:+.4f} | "
            f"{row['positive_splits']}/10 | {row['worst_split']:+.4f} | "
            f"{row['actions']} |"
        )
    lines.extend(
        [
            "",
            "All nine applications reuse the same 29 SIDD-CC scenes and the",
            "same ten partitions. This is not an independent population.",
        ]
    )
    (args.output / "A_SOURCEFREE_SUMMARY.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
