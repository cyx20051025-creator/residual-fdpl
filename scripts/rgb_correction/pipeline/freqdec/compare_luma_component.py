#!/usr/bin/env python3
"""Compare raw-RGB and zero-luma results from aligned row records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


LUMA = np.asarray([0.299, 0.587, 0.114], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-result", type=Path, required=True)
    parser.add_argument("--chroma-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def populations(payload: dict) -> dict[str, dict]:
    return {
        item["population"]: item
        for item in payload["populations"]
    }


def stats(values: np.ndarray) -> dict[str, float | int]:
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95)),
        "positive_fraction": float((values > 0.0).mean()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def main() -> None:
    args = parse_args()
    raw = populations(
        json.loads(args.raw_result.read_text(encoding="utf-8"))
    )
    chroma = populations(
        json.loads(args.chroma_result.read_text(encoding="utf-8"))
    )
    output = {
        "schema_version": 1,
        "raw_result": str(args.raw_result),
        "chroma_result": str(args.chroma_result),
        "populations": {},
    }
    for name in raw.keys() & chroma.keys():
        raw_rows = raw[name]["records"]
        chroma_rows = chroma[name]["records"]
        if len(raw_rows) != len(chroma_rows):
            raise ValueError(f"row-count mismatch for {name}")
        if any(
            int(first["row"]) != int(second["row"])
            or first.get("source_entry") != second.get("source_entry")
            for first, second in zip(raw_rows, chroma_rows, strict=True)
        ):
            raise ValueError(f"row alignment mismatch for {name}")

        raw_delta = np.asarray([
            float(row["candidate_psnr"]) - float(row["base_psnr"])
            for row in raw_rows
        ])
        chroma_delta = np.asarray([
            float(row["candidate_psnr"]) - float(row["base_psnr"])
            for row in chroma_rows
        ])
        raw_offsets = np.asarray([
            row["effective_offset_rgb"] for row in raw_rows
        ], dtype=np.float64)
        chroma_offsets = np.asarray([
            row["effective_offset_rgb"] for row in chroma_rows
        ], dtype=np.float64)
        luma = raw_offsets @ LUMA
        delta_change = chroma_delta - raw_delta
        correlation = (
            float(np.corrcoef(luma, delta_change)[0, 1])
            if np.std(luma) > 1e-12 and np.std(delta_change) > 1e-12
            else None
        )
        output["populations"][name] = {
            "raw_delta": stats(raw_delta),
            "chroma_delta": stats(chroma_delta),
            "chroma_minus_raw": stats(delta_change),
            "raw_luma_offset": stats(luma),
            "raw_abs_luma_offset": stats(np.abs(luma)),
            "chroma_abs_luma_offset": stats(
                np.abs(chroma_offsets @ LUMA)
            ),
            "corr_abs_raw_luma_with_chroma_minus_raw": correlation,
            "corr_signed_raw_luma_with_chroma_minus_raw": (
                float(np.corrcoef(luma, delta_change)[0, 1])
                if np.std(delta_change) > 1e-12
                else None
            ),
            "raw_negative_chroma_nonnegative_fraction": float(
                (
                    (raw_delta < 0.0)
                    & (chroma_delta >= 0.0)
                ).mean()
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
