#!/usr/bin/env python3
"""Select the highest-ranked rule from a source screen by exact prefix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--prefix", default="chroma__")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    screen = json.loads(args.screen.read_text(encoding="utf-8"))
    eligible = [
        name
        for name in screen["ranking"]
        if name.startswith(args.prefix)
    ]
    if not eligible:
        raise ValueError(f"no rule starts with {args.prefix!r}")
    selected = eligible[0]
    record = screen["results"][selected]
    output = {
        "schema_version": 1,
        "screen": str(args.screen),
        "prefix": args.prefix,
        "selected_rule": selected,
        "selected_scale": float(
            record["selected_scale_record"]["scale"]
        ),
        "selection_record": record,
        "source_only": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
