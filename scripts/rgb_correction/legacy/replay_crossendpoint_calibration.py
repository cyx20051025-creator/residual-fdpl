#!/usr/bin/env python3
"""Cross-endpoint calibration diagnostic on cached SIDD-CC features.

For each split, fit the 117-statistic Ridge map to the zero-luma reference
offsets of a calibrator checkpoint and apply it to a recipient checkpoint.
The score is the exact pre-clipping PSNR change obtained from each crop's
base MSE and reference mean offset. This isolates the effect of calibrating
on the wrong frozen checkpoint without rerendering images.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


BUDGETS = (16,)
CAPS = {"capped": 0.02, "uncapped": None}


def load_payload(path: Path) -> dict:
    payload = np.load(path)
    return {
        "features": np.asarray(payload["features"], dtype=np.float64),
        "targets": np.asarray(payload["targets"], dtype=np.float64),
        "base_mse": np.asarray(payload["base_mse"], dtype=np.float64),
        "rows": np.asarray(payload["rows"], dtype=np.int64),
        "scenes": np.asarray(payload["scenes"]),
    }


def preclip_delta_psnr(
    base_mse: np.ndarray,
    target: np.ndarray,
    offset: np.ndarray,
) -> np.ndarray:
    corrected_mse = (
        base_mse
        - np.sum(target * offset, axis=1) / 6.0
        + np.sum(offset * offset, axis=1) / 12.0
    )
    if np.any(corrected_mse <= 0.0):
        raise ValueError("non-positive reconstructed pre-clipping MSE")
    return 10.0 * np.log10(base_mse / corrected_mse)


def bootstrap_interval(
    values: np.ndarray,
    *,
    samples: int = 5000,
    seed: int,
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
        / "crossendpoint_calibration_20261117",
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

    rows = []
    split_records = []
    split_seed = 20261216
    for calibrator_index, calibrator_name in enumerate(endpoint_names):
        calibrator = payloads[calibrator_name]
        for recipient_index, recipient_name in enumerate(endpoint_names):
            recipient = payloads[recipient_name]
            for cap_name, cap in CAPS.items():
                split_values = []
                split_norms = []
                per_split = []
                for split in range(10):
                    rng = np.random.default_rng(split_seed + split)
                    calibration_scenes = sorted(
                        rng.choice(
                            unique_scenes,
                            size=BUDGETS[0],
                            replace=False,
                        ).tolist()
                    )
                    evaluation_scenes = [
                        scene
                        for scene in unique_scenes
                        if scene not in calibration_scenes
                    ]
                    fit = fit_feature_ridge(
                        calibrator["features"],
                        np.zeros_like(calibrator["targets"]),
                        calibrator["targets"],
                        calibrator["scenes"],
                        calibration_scenes,
                        slice(0, 117),
                        selector.RIDGE_GRID,
                    )
                    prediction = predict_feature_ridge(
                        fit,
                        recipient["features"],
                        np.zeros_like(recipient["targets"]),
                    )
                    prediction = project_zero_luma(prediction)
                    if cap is not None:
                        prediction = selector.cap_offsets(prediction, cap)
                    evaluation_indices = np.isin(
                        recipient["scenes"],
                        evaluation_scenes,
                    )
                    deltas = preclip_delta_psnr(
                        recipient["base_mse"][evaluation_indices],
                        recipient["targets"][evaluation_indices],
                        prediction[evaluation_indices],
                    )
                    split_values.append(float(deltas.mean()))
                    norm = float(
                        np.linalg.norm(
                            prediction[evaluation_indices],
                            axis=1,
                        ).mean()
                    )
                    split_norms.append(norm)
                    per_split.append(
                        {
                            "split": split,
                            "calibration_scenes": calibration_scenes,
                            "evaluation_scenes": evaluation_scenes,
                            "mean_preclip_psnr_delta": float(deltas.mean()),
                            "mean_offset_norm": norm,
                            "lambda": float(fit["lambda"]),
                        }
                    )

                values = np.asarray(split_values, dtype=np.float64)
                low, high = bootstrap_interval(
                    values,
                    seed=20261117
                    + 1000 * calibrator_index
                    + 10 * recipient_index
                    + (1 if cap is not None else 2),
                )
                rows.append(
                    {
                        "calibrator": calibrator_name,
                        "recipient": recipient_name,
                        "mode": cap_name,
                        "calibration_scenes": BUDGETS[0],
                        "mean_preclip_psnr": float(values.mean()),
                        "interval_low": low,
                        "interval_high": high,
                        "positive_splits": int((values > 0.0).sum()),
                        "worst_split": float(values.min()),
                        "best_split": float(values.max()),
                        "mean_offset_norm": float(np.mean(split_norms)),
                    }
                )
                split_records.append(
                    {
                        "calibrator": calibrator_name,
                        "recipient": recipient_name,
                        "mode": cap_name,
                        "splits": per_split,
                    }
                )

    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "CROSSENDPOINT_MATRIX.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Cross-endpoint calibration diagnostic",
        "",
        "A 117-statistic Ridge map is fitted on 16 calibration scenes from the",
        "calibrator checkpoint and applied to 13 held-out scenes from the",
        "recipient checkpoint. Scores are exact pre-clipping PSNR changes from",
        "the cached base MSE and reference offsets. They do not include clipping",
        "or perceptual metrics.",
        "",
        "| Mode | Calibrator | Recipient | ΔPSNR [95% split interval] | Positive | Worst | Mean offset norm |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['mode']} | {row['calibrator']} | "
            f"{row['recipient']} | {row['mean_preclip_psnr']:+.4f} "
            f"[{row['interval_low']:+.4f}, {row['interval_high']:+.4f}] | "
            f"{row['positive_splits']}/10 | {row['worst_split']:+.4f} | "
            f"{row['mean_offset_norm']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Same-checkpoint cells are the reference fits. The cross-checkpoint",
            "cells test whether a calibration map learned from one frozen",
            "endpoint supplies the right image-statistic mapping for another.",
            "All three endpoints reuse the same scenes and splits and are not",
            "independent populations.",
        ]
    )
    (args.output / "CROSSENDPOINT_MATRIX.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    (args.output / "CROSSENDPOINT_SPLITS.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "split_seed": split_seed,
                "calibration_scenes": BUDGETS[0],
                "cap_capped": CAPS["capped"],
                "records": split_records,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
