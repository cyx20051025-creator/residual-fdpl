#!/usr/bin/env python3
"""Create a hashed ensemble-rule variant without refitting member models."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--rule", required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument(
        "--subspace",
        choices=("raw", "chroma"),
        default="raw",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = pickle.loads(args.artifact.read_bytes())
    if artifact.get("schema_version") != 3:
        raise ValueError("expected ensemble schema version 3")
    screen = json.loads(args.screen.read_text(encoding="utf-8"))
    if args.rule not in screen["results"]:
        raise ValueError("requested ensemble rule is absent from source screen")
    record = screen["results"][args.rule]
    allowed_scales = {
        float(item["scale"]) for item in record["scale_records"]
    }
    if args.scale not in allowed_scales:
        raise ValueError("requested scale is absent from source screen")
    artifact["selected_ensemble_rule"] = args.rule
    artifact["source_selected_scale"] = args.scale
    artifact["selection_record"] = record
    artifact["correction_subspace"] = args.subspace
    artifact["source_screen"] = str(args.screen)
    artifact["source_screen_sha256"] = sha256(args.screen)
    artifact["derived_from_artifact"] = str(args.artifact)
    artifact["derived_from_artifact_sha256"] = sha256(args.artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(artifact, handle)
    report = {
        "rule": args.rule,
        "scale": args.scale,
        "subspace": args.subspace,
        "source_screen": str(args.screen),
        "source_screen_sha256": sha256(args.screen),
        "derived_from_artifact": str(args.artifact),
        "derived_from_artifact_sha256": sha256(args.artifact),
        "artifact": str(args.output),
        "artifact_sha256": sha256(args.output),
        "selection_record": record,
    }
    args.report.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
