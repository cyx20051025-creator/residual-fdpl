#!/usr/bin/env python3
"""Derive an exact evaluation payload for a frozen zero-scale comparator.

When the source-only calibration selects alpha=0, the candidate is exactly the
frozen-A output. This utility remaps an existing evaluation of the same base
checkpoint to the global-comparator schema and records that derivation.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-evaluation", type=Path, required=True)
    parser.add_argument("--model-artifact", type=Path, required=True)
    parser.add_argument("--model-artifact-sha256", required=True)
    parser.add_argument("--global-calibration", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.source_evaluation.read_text(encoding="utf-8"))
    if args.global_calibration != 0.0:
        raise ValueError("exact derivation is only valid for alpha=0")
    derived = copy.deepcopy(payload)
    derived["model_artifact"] = str(args.model_artifact)
    derived["model_artifact_sha256"] = args.model_artifact_sha256
    derived["selected_candidate"] = "global"
    derived["global_calibration"] = 0.0
    derived["derivation"] = {
        "method": "exact_frozen_base_projection",
        "source_evaluation": str(args.source_evaluation),
        "reason": (
            "The source-only global calibration is exactly zero; therefore "
            "the corrected output, metrics, band errors, and clipping are "
            "identical to frozen A."
        ),
    }
    for population in derived["populations"]:
        population["candidate_summary"] = copy.deepcopy(
            population["base_summary"]
        )
        for record in population["records"]:
            record["candidate_preclip_mse"] = record["base_preclip_mse"]
            record["candidate_clip_fraction"] = 0.0
            record["alpha"] = 0.0
            record["effective_offset_rgb"] = [0.0, 0.0, 0.0]
            for key in ("psnr", "ssim", "lpips", "dists"):
                record[f"candidate_{key}"] = record[f"base_{key}"]
            record["candidate_radial_band_mse"] = list(
                record["base_radial_band_mse"]
            )
        population["alpha_stats"] = {
            "mean": 0.0,
            "zero_fraction": 1.0,
            "one_fraction": 0.0,
            "quantiles": {
                "min": 0.0,
                "p05": 0.0,
                "p25": 0.0,
                "p50": 0.0,
                "p75": 0.0,
                "p95": 0.0,
                "max": 0.0,
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(derived, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "populations": len(derived["populations"]),
                "alpha": 0.0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
