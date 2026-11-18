#!/usr/bin/env python3
"""Select one ensemble rule that passes source-tail protection on all seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen", action="append", type=Path, required=True)
    parser.add_argument("--required-prefix", default="")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def eligible_record(record: dict) -> bool:
    selected = record["selected_scale_record"]
    return bool(
        float(record["selected_scale"]) > 0.0
        and float(selected["pass_fraction_scene_ge_minus_0_02"]) >= 0.95
        and float(selected["p05_scene_delta"]) >= 0.0
    )


def main() -> None:
    args = parse_args()
    screens = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in args.screen
    ]
    seeds = sorted(int(screen["seed"]) for screen in screens)
    if len(seeds) != 3:
        raise ValueError("expected exactly three source screens")
    common = set(screens[0]["results"])
    for screen in screens[1:]:
        common &= set(screen["results"])
    common = {
        rule for rule in common if rule.startswith(args.required_prefix)
    }

    candidates = {}
    for rule in sorted(common):
        records = [
            next(
                screen["results"][rule]
                for screen in screens
                if int(screen["seed"]) == seed
            )
            for seed in seeds
        ]
        if not all(eligible_record(record) for record in records):
            continue
        candidates[rule] = {
            "rule": rule,
            "seeds": {
                str(seed): {
                    "selected_scale": float(record["selected_scale"]),
                    "selection_record": record["selected_scale_record"],
                    "direction_positive_fraction": float(
                        record["direction_positive_fraction"]
                    ),
                }
                for seed, record in zip(seeds, records, strict=True)
            },
            "worst_p05": float(
                min(
                    record["selected_scale_record"]["p05_scene_delta"]
                    for record in records
                )
            ),
            "mean_scene_delta": float(
                np.mean(
                    [
                        record["selected_scale_record"]["mean_scene_delta"]
                        for record in records
                    ]
                )
            ),
            "mean_source_pass_fraction": float(
                np.mean(
                    [
                        record["selected_scale_record"][
                            "pass_fraction_scene_ge_minus_0_02"
                        ]
                        for record in records
                    ]
                )
            ),
            "mean_direction_positive_fraction": float(
                np.mean(
                    [
                        record["direction_positive_fraction"]
                        for record in records
                    ]
                )
            ),
            "mean_selected_scale": float(
                np.mean([record["selected_scale"] for record in records])
            ),
        }

    ranking = sorted(
        candidates,
        key=lambda rule: (
            candidates[rule]["worst_p05"],
            candidates[rule]["mean_scene_delta"],
            candidates[rule]["mean_source_pass_fraction"],
            candidates[rule]["mean_direction_positive_fraction"],
            -candidates[rule]["mean_selected_scale"],
        ),
        reverse=True,
    )
    output = {
        "schema_version": 1,
        "source_screens": [str(path) for path in args.screen],
        "seeds": seeds,
        "common_rules": len(common),
        "required_prefix": args.required_prefix,
        "eligible_common_rules": len(ranking),
        "selected_rule": ranking[0] if ranking else None,
        "selected_record": candidates[ranking[0]] if ranking else None,
        "ranking": ranking,
        "candidates": candidates,
        "source_only": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "common_rules": output["common_rules"],
                "eligible_common_rules": output["eligible_common_rules"],
                "selected_rule": output["selected_rule"],
                "top": [
                    candidates[rule] for rule in ranking[:10]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
