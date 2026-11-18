#!/usr/bin/env python3
"""Cross-endpoint calibration sensitivity over scene budget and cap."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replay_crossendpoint_calibration import (  # noqa: E402
    bootstrap_interval,
    load_payload,
    preclip_delta_psnr,
)


BUDGETS = (8, 16, 24)
CAPS = {
    "cap001": 0.01,
    "cap002": 0.02,
    "cap004": 0.04,
}


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
        "--release-20261116",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "target_endpoint_calibration_20261116"
        / "risk_selector_release_20261116",
    )
    parser.add_argument(
        "--restormer-features",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "target_endpoint_calibration_20261116"
        / "restormer_sidd_features.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent
        / "crossendpoint_sensitivity_20261117",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(args.release / "code"))
    from evaluate_feature_target_adaptation import (
        fit_feature_ridge,
        predict_feature_ridge,
        project_zero_luma,
    )
    import risk_controlled_selector as selector

    endpoint_paths = {
        "NAFNet-ft62": args.feature_root
        / "features_naf_finetuned_seed62_sidd_benchmark.npz",
        "NAFNet-ft63": args.feature_root
        / "features_naf_finetuned_seed63_sidd_benchmark.npz",
        "NAFNet-ft64": args.feature_root
        / "features_naf_finetuned_seed64_sidd_benchmark.npz",
        "NAFNet-official": args.release_20261116
        / "artifacts/features/sidd/features_naf_official_sidd_benchmark.npz",
        "SwinIR-M": args.release_20261116
        / "artifacts/features/sidd/features_swinir_seed62_sidd_benchmark.npz",
        "Restormer": args.restormer_features,
    }
    payloads = {}
    for name, path in endpoint_paths.items():
        payload = load_payload(path)
        payload["targets"] = project_zero_luma(payload["targets"])
        payload["path"] = str(path.resolve())
        payloads[name] = payload

    endpoint_names = tuple(endpoint_paths)
    reference = payloads[endpoint_names[0]]
    unique_scenes = sorted(set(reference["scenes"].tolist()))
    for name, payload in payloads.items():
        if not np.array_equal(payload["rows"], reference["rows"]):
            raise ValueError(f"row order differs for endpoint {name}")
        if not np.array_equal(payload["scenes"], reference["scenes"]):
            raise ValueError(f"scene order differs for endpoint {name}")
        if payload["features"].shape != (len(reference["rows"]), 117):
            raise ValueError(f"feature shape differs for endpoint {name}")

    splits_by_budget = {}
    for budget in BUDGETS:
        splits = []
        for split in range(10):
            rng = np.random.default_rng(20261216 + split)
            calibration_scenes = sorted(
                rng.choice(
                    unique_scenes,
                    size=budget,
                    replace=False,
                ).tolist()
            )
            evaluation_scenes = [
                scene
                for scene in unique_scenes
                if scene not in calibration_scenes
            ]
            splits.append(
                {
                    "calibration_scenes": calibration_scenes,
                    "evaluation_scenes": evaluation_scenes,
                }
            )
        splits_by_budget[budget] = splits

    rows = []
    for budget in BUDGETS:
        for calibrator_index, calibrator_name in enumerate(endpoint_names):
            calibrator = payloads[calibrator_name]
            for recipient_index, recipient_name in enumerate(endpoint_names):
                recipient = payloads[recipient_name]
                predictions_by_split = []
                for split_index, split in enumerate(splits_by_budget[budget]):
                    fit = fit_feature_ridge(
                        calibrator["features"],
                        np.zeros_like(calibrator["targets"]),
                        calibrator["targets"],
                        calibrator["scenes"],
                        split["calibration_scenes"],
                        slice(0, 117),
                        selector.RIDGE_GRID,
                    )
                    prediction = predict_feature_ridge(
                        fit,
                        recipient["features"],
                        np.zeros_like(recipient["targets"]),
                    )
                    prediction = project_zero_luma(prediction)
                    predictions_by_split.append(prediction)

                for cap_index, (cap_name, cap) in enumerate(CAPS.items()):
                    split_values = []
                    split_norms = []
                    for split_index, split in enumerate(
                        splits_by_budget[budget]
                    ):
                        prediction = predictions_by_split[split_index]
                        if cap is not None:
                            prediction = selector.cap_offsets(prediction, cap)
                        evaluation_indices = np.isin(
                            recipient["scenes"],
                            split["evaluation_scenes"],
                        )
                        deltas = preclip_delta_psnr(
                            recipient["base_mse"][evaluation_indices],
                            recipient["targets"][evaluation_indices],
                            prediction[evaluation_indices],
                        )
                        split_values.append(float(deltas.mean()))
                        split_norms.append(
                            float(
                                np.linalg.norm(
                                    prediction[evaluation_indices],
                                    axis=1,
                                ).mean()
                            )
                        )

                    values = np.asarray(split_values, dtype=np.float64)
                    low, high = bootstrap_interval(
                        values,
                        seed=(
                            20261117
                            + 100000
                            + 1000 * budget
                            + 100 * calibrator_index
                            + 10 * recipient_index
                            + cap_index
                        ),
                    )
                    rows.append(
                        {
                            "budget": budget,
                            "cap": cap,
                            "cap_name": cap_name,
                            "calibrator": calibrator_name,
                            "recipient": recipient_name,
                            "mean_preclip_psnr": float(values.mean()),
                            "interval_low": low,
                            "interval_high": high,
                            "positive_splits": int((values > 0.0).sum()),
                            "worst_split": float(values.min()),
                            "best_split": float(values.max()),
                            "mean_offset_norm": float(np.mean(split_norms)),
                        }
                    )

    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "CROSSENDPOINT_SENSITIVITY.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fine_to_fine = [
        row
        for row in rows
        if row["calibrator"].startswith("NAFNet-ft")
        and row["recipient"].startswith("NAFNet-ft")
        and row["calibrator"] != row["recipient"]
    ]
    fine_to_official = [
        row
        for row in rows
        if row["calibrator"].startswith("NAFNet-ft")
        and row["recipient"] == "NAFNet-official"
    ]
    lines = [
        "# Cross-endpoint calibration sensitivity",
        "",
        "Scores are exact pre-clipping PSNR changes. Conditions vary only the",
        "calibration-scene budget and the frozen candidate cap; the feature",
        "map, endpoints, splits and pre-clipping identity are unchanged.",
        "",
        "| Budget | Cap | Calibrator | Recipient | ΔPSNR [95%] | Positive | Worst |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['budget']} | {row['cap_name']} | "
            f"{row['calibrator']} | {row['recipient']} | "
            f"{row['mean_preclip_psnr']:+.4f} "
            f"[{row['interval_low']:+.4f}, {row['interval_high']:+.4f}] | "
            f"{row['positive_splits']}/10 | {row['worst_split']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Decision checks",
            "",
            f"- Fine-tuned-to-fine-tuned cells: {len(fine_to_fine)}; "
            f"range {min(row['mean_preclip_psnr'] for row in fine_to_fine):+.4f} "
            f"to {max(row['mean_preclip_psnr'] for row in fine_to_fine):+.4f} dB.",
            f"- Fine-tuned-to-official cells: {len(fine_to_official)}; "
            f"range {min(row['mean_preclip_psnr'] for row in fine_to_official):+.4f} "
            f"to {max(row['mean_preclip_psnr'] for row in fine_to_official):+.4f} dB.",
            "",
            "All conditions reuse the same 29 scenes and the same split seed.",
            "The matrix is descriptive and does not render clipped images or",
            "compute perceptual metrics.",
        ]
    )
    (args.output / "CROSSENDPOINT_SENSITIVITY.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print("\n".join(lines[-10:]))


if __name__ == "__main__":
    main()
