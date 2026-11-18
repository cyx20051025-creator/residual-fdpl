#!/usr/bin/env python3
"""Target-pairing shuffle control for the 117-statistic Ridge map."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_crossendpoint_calibration import (  # noqa: E402
    load_payload,
    preclip_delta_psnr,
)


SEEDS = (62, 63, 64)
SHUFFLES = 20
BUDGET = 16


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "crossendpoint_calibration_20261117"
        / "features",
    )
    parser.add_argument(
        "--release",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "target_endpoint_calibration_20261116"
        / "risk_selector_release_full_20261117",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent
        / "target_pairing_shuffle_control_20261117",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(args.release / "code"))
    from evaluate_feature_target_adaptation import (
        fit_feature_ridge,
        predict_feature_ridge,
        project_zero_luma,
    )
    import risk_controlled_selector as selector

    rows = []
    split_records = []
    for seed in SEEDS:
        path = args.feature_root / (
            f"features_naf_finetuned_seed{seed}_sidd_benchmark.npz"
        )
        payload = load_payload(path)
        payload["targets"] = project_zero_luma(payload["targets"])
        unique_scenes = sorted(set(payload["scenes"].tolist()))

        for split in range(10):
            rng = np.random.default_rng(20261216 + split)
            calibration_scenes = sorted(
                rng.choice(
                    unique_scenes,
                    size=BUDGET,
                    replace=False,
                ).tolist()
            )
            evaluation_scenes = [
                scene
                for scene in unique_scenes
                if scene not in calibration_scenes
            ]
            calibration_mask = np.isin(
                payload["scenes"],
                calibration_scenes,
            )
            evaluation_mask = np.isin(
                payload["scenes"],
                evaluation_scenes,
            )

            actual_fit = fit_feature_ridge(
                payload["features"],
                np.zeros_like(payload["targets"]),
                payload["targets"],
                payload["scenes"],
                calibration_scenes,
                slice(0, 117),
                selector.RIDGE_GRID,
            )
            actual_prediction = predict_feature_ridge(
                actual_fit,
                payload["features"],
                np.zeros_like(payload["targets"]),
            )
            actual_prediction = project_zero_luma(actual_prediction)
            actual_values = preclip_delta_psnr(
                payload["base_mse"][evaluation_mask],
                payload["targets"][evaluation_mask],
                actual_prediction[evaluation_mask],
            )
            actual = float(actual_values.mean())

            shuffled_values = []
            for shuffle in range(SHUFFLES):
                shuffled_targets = payload["targets"].copy()
                shuffled_targets[calibration_mask] = (
                    shuffled_targets[calibration_mask][
                        np.random.default_rng(
                            20261117
                            + 10000 * seed
                            + 100 * split
                            + shuffle
                        ).permutation(calibration_mask.sum())
                    ]
                )
                shuffled_fit = fit_feature_ridge(
                    payload["features"],
                    np.zeros_like(payload["targets"]),
                    shuffled_targets,
                    payload["scenes"],
                    calibration_scenes,
                    slice(0, 117),
                    selector.RIDGE_GRID,
                )
                shuffled_prediction = predict_feature_ridge(
                    shuffled_fit,
                    payload["features"],
                    np.zeros_like(payload["targets"]),
                )
                shuffled_prediction = project_zero_luma(shuffled_prediction)
                shuffled_values.append(
                    float(
                        preclip_delta_psnr(
                            payload["base_mse"][evaluation_mask],
                            payload["targets"][evaluation_mask],
                            shuffled_prediction[evaluation_mask],
                        ).mean()
                    )
                )

            values = np.asarray(shuffled_values, dtype=np.float64)
            split_records.append(
                {
                    "seed": seed,
                    "split": split,
                    "calibration_scenes": calibration_scenes,
                    "evaluation_scenes": evaluation_scenes,
                    "actual_preclip_psnr": actual,
                    "shuffled_mean": float(values.mean()),
                    "shuffled_std": float(values.std(ddof=0)),
                    "shuffled_min": float(values.min()),
                    "shuffled_max": float(values.max()),
                    "actual_quantile_among_shuffles": float(
                        (values < actual).mean()
                    ),
                    "shuffled_values": shuffled_values,
                }
            )

    for seed in SEEDS:
        records = [
            record
            for record in split_records
            if record["seed"] == seed
        ]
        actual = np.asarray(
            [record["actual_preclip_psnr"] for record in records]
        )
        shuffled = np.asarray(
            [record["shuffled_mean"] for record in records]
        )
        differences = actual - shuffled
        rows.append(
            {
                "seed": seed,
                "actual_mean": float(actual.mean()),
                "shuffled_mean": float(shuffled.mean()),
                "paired_difference": float(differences.mean()),
                "positive_difference_splits": int((differences > 0).sum()),
                "worst_actual": float(actual.min()),
                "worst_shuffled": float(shuffled.min()),
                "actual_wins_shuffle_replicates": float(
                    np.mean(
                        [
                            record["actual_quantile_among_shuffles"]
                            for record in records
                        ]
                    )
                ),
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "TARGET_PAIRING_SHUFFLE.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "TARGET_PAIRING_SHUFFLE_SPLITS.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "budget": BUDGET,
                "shuffles_per_split": SHUFFLES,
                "split_seed": 20261216,
                "records": split_records,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Target-pairing shuffle control",
        "",
        "Each shuffled replicate preserves the calibration features and offset",
        "distribution but permutes their correspondence before fitting Ridge.",
        "Scores are exact pre-clipping PSNR changes on the same 13 outer scenes.",
        "",
        "| Endpoint | Actual map | Shuffled mean | Paired difference | "
        "Positive splits | Actual win rate vs shuffles |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['actual_mean']:+.4f} | "
            f"{row['shuffled_mean']:+.4f} | "
            f"{row['paired_difference']:+.4f} | "
            f"{row['positive_difference_splits']}/10 | "
            f"{row['actual_wins_shuffle_replicates']:.1%} |"
        )
    lines.extend(
        [
            "",
            "This is a mechanism control without clipping or perceptual",
            "metrics. Repeated splits reuse the same 29 scenes.",
        ]
    )
    (args.output / "TARGET_PAIRING_SHUFFLE.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
