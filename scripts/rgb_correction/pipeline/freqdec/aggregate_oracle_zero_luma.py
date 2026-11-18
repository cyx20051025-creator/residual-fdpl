#!/usr/bin/env python3
"""Aggregate label-aware zero-luma oracle headroom across seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--label", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.input)
    ]
    populations = [row["population"] for row in payloads[0]["populations"]]
    aggregate = {}
    for population in populations:
        seed_values = {}
        for payload in payloads:
            matches = [
                row
                for row in payload["populations"]
                if row["population"] == population
            ]
            if len(matches) != 1:
                raise ValueError(f"population mismatch for {population}")
            seed_values[Path(payload["base_checkpoint"]).stem] = matches[0][
                "summary"
            ]["metric_means"]
        deltas = np.asarray(
            [row["psnr_delta"] for row in seed_values.values()],
            dtype=np.float64,
        )
        aggregate[population] = {
            "seed_count": len(seed_values),
            "seed_mean_psnr_delta": float(deltas.mean()),
            "worst_seed_psnr_delta": float(deltas.min()),
            "best_seed_psnr_delta": float(deltas.max()),
            "positive_seed_count": int((deltas > 0.0).sum()),
            "seeds": seed_values,
        }
    output = {
        "schema_version": 1,
        "label": args.label,
        "diagnostic": "label-aware projected channel-mean oracle",
        "input_files": [str(path) for path in sorted(args.input)],
        "aggregate": aggregate,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        f"# Zero-luma oracle headroom: {args.label}",
        "",
        "This is a label-aware upper bound, not method performance.",
        "",
        "| Population | Mean PSNR headroom | Worst seed | Positive seeds |",
        "|---|---:|---:|---:|",
    ]
    for population, row in aggregate.items():
        lines.append(
            f"| {population} | {row['seed_mean_psnr_delta']:+.6f} | "
            f"{row['worst_seed_psnr_delta']:+.6f} | "
            f"{row['positive_seed_count']}/{row['seed_count']} |"
        )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
