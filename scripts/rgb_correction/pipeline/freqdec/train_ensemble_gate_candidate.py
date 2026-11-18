#!/usr/bin/env python3
"""Fit the source-selected gate around a frozen ensemble direction."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from .screen_direction_ensembles import (
    capped_candidates,
    ensemble_candidates,
    subspace_candidates,
)
from .screen_source_gate import gate_matrix, model as gate_model_factory


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ensemble-artifact", type=Path, required=True)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = pickle.loads(args.ensemble_artifact.read_bytes())
    if artifact.get("schema_version") != 3:
        raise ValueError("expected ensemble schema version 3")
    if int(artifact["seed"]) != args.seed:
        raise ValueError("seed mismatch")
    screen = json.loads(args.screen.read_text(encoding="utf-8"))
    selected_gate = str(screen["selected_gate"])
    if selected_gate == "constant_gate":
        raise ValueError("selected screen has no adaptive gate")
    selected_record = screen["results"][selected_gate]

    payload = np.load(args.oof, allow_pickle=False)
    features = np.asarray(payload["features"], dtype=np.float64)
    targets = np.asarray(payload["targets"], dtype=np.float64)
    member_predictions = {
        key.removeprefix("prediction__"): np.asarray(
            payload[key], dtype=np.float64
        )
        for key in payload.files
        if key.startswith("prediction__")
    }
    rule = str(artifact["selected_ensemble_rule"])
    candidates = subspace_candidates(
        capped_candidates(
            {
                **member_predictions,
                **ensemble_candidates(member_predictions),
            }
        )
    )
    if rule not in candidates:
        raise ValueError(f"unknown direction rule: {rule}")
    direction = candidates[rule]
    alpha_target = np.clip(
        np.einsum("ij,ij->i", direction, targets)
        / np.maximum(np.einsum("ij,ij->i", direction, direction), 1e-12),
        0.0,
        1.0,
    )
    matrix, feature_names = gate_matrix(
        features, member_predictions, direction
    )
    gate = gate_model_factory(selected_gate, args.seed + 4000)
    gate.fit(matrix, alpha_target)

    artifact["schema_version"] = 4
    artifact["gate_model_name"] = selected_gate
    artifact["gate_model"] = gate
    artifact["gate_feature_count"] = int(matrix.shape[1])
    artifact["gate_feature_names"] = [
        str(value) for value in feature_names
    ]
    artifact["source_gate_scale"] = float(
        selected_record["selected_scale"]
    )
    artifact["source_gate_screen"] = str(args.screen)
    artifact["source_gate_screen_sha256"] = sha256(args.screen)
    artifact["source_oof"] = str(args.oof)
    artifact["source_oof_sha256"] = sha256(args.oof)
    artifact["source_derived_from_artifact"] = str(
        args.ensemble_artifact
    )
    artifact["source_derived_from_artifact_sha256"] = sha256(
        args.ensemble_artifact
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(artifact, handle)
    report = {
        key: value
        for key, value in artifact.items()
        if key not in {"ensemble_members", "gate_model"}
    }
    report["artifact"] = str(args.output)
    report["artifact_sha256"] = sha256(args.output)
    report["gate_screen_record"] = selected_record
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
