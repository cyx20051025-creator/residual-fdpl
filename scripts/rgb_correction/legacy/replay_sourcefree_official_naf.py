#!/usr/bin/env python3
"""Replay source 57/58 on released NAFNet from cached candidate summaries."""

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
        "--candidate",
        type=Path,
        default=Path(__file__).resolve().parent
        / "sourcefree_official_naf_20261117"
        / "risk_selector_no_source_naf_official_source56.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent
        / "sourcefree_official_naf_replay_20261117",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(args.release / "code"))
    import risk_controlled_selector as selector

    candidates = load_json(args.candidate)
    candidates["_path"] = str(args.candidate.resolve())
    args.output.mkdir(parents=True, exist_ok=True)

    rows = []
    for source in SOURCES:
        original_path = args.release / "results" / (
            f"risk_selector_naf_official_source{source}_"
            "fresh20261216_full.json"
        )
        original = load_json(original_path)
        original["_path"] = str(original_path.resolve())
        replay = combine_record(
            original,
            candidates,
            selector.choose_candidate,
        )
        replay["replay_kind"] = (
            "official-NAFNet source-specific simple candidates plus "
            "117-only Ridge"
        )
        replay["source_seed"] = source
        output_path = args.output / (
            f"risk_selector_sourcefree_naf_official_source{source}.json"
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
            seed=20261117 + source,
        )
        changed = []
        for index, (before, after) in enumerate(
            zip(original["records"], replay["records"], strict=True),
            start=1,
        ):
            old_action = before["decision"]["selected"]
            new_action = after["decision"]["selected"]
            if old_action != new_action:
                changed.append(f"{index}:{old_action}->{new_action}")
        rows.append(
            {
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
                "identical_actions": int(
                    sum(
                        before["decision"]["selected"]
                        == after["decision"]["selected"]
                        for before, after in zip(
                            original["records"],
                            replay["records"],
                            strict=True,
                        )
                    )
                ),
                "changed_actions": "; ".join(changed),
                "actions": " / ".join(
                    f"{name}:{count}"
                    for name, count in replay["selected_counts"].items()
                    if count
                ),
            }
        )

    csv_path = args.output / "OFFICIAL_NAF_SOURCEFREE_SUMMARY.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Source-free selector on released NAFNet",
        "",
        "Source 56 is the exact GPU validation run. Source 57 and 58 are",
        "deterministic replays: their source-specific simple-candidate",
        "summaries remain unchanged, while the source-free Ridge candidates",
        "are shared because the 117-statistic input does not depend on source",
        "seed.",
        "",
        "| Source | Selected ΔPSNR | Original | Difference | Positive | "
        "Worst | Agreement | Actions |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['source']} | {row['selected_psnr']:+.4f} | "
            f"{row['original_psnr']:+.4f} | "
            f"{row['difference_from_original']:+.4f} | "
            f"{row['positive_splits']}/10 | {row['worst_split']:+.4f} | "
            f"{row['identical_actions']}/10 | {row['actions']} |"
        )
    lines.extend(
        [
            "",
            "All rows reuse the same 29 SIDD-CC scenes and ten partitions.",
            "The source repeats are predictor variants, not denoiser trainings.",
        ]
    )
    (args.output / "OFFICIAL_NAF_SOURCEFREE_SUMMARY.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
