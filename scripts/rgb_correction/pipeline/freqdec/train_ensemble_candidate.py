#!/usr/bin/env python3
"""Fit a source-frozen ensemble of RGB-direction predictors."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from .screen_direction_ensembles import CONFIGS
from .screen_direction_targets import (
    make_model,
    selection_key,
    transform_target,
)
from .train_protected_correction import EXT_FEATURE_NAMES, RAW_FEATURE_NAMES


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = np.load(args.features, allow_pickle=False)
    features = np.asarray(payload["features"], dtype=np.float64)
    targets = np.asarray(payload["targets"], dtype=np.float64)
    source_feature_names = tuple(payload["feature_names"].tolist())
    if source_feature_names == EXT_FEATURE_NAMES:
        feature_schema = "v1_extended36"
        extended_feature_names = EXT_FEATURE_NAMES
    elif len(source_feature_names) == 117:
        feature_schema = "v3_extended117"
        extended_feature_names = source_feature_names
    else:
        raise ValueError(
            f"unsupported extended feature schema with {len(source_feature_names)} names"
        )
    if source_feature_names[:12] != RAW_FEATURE_NAMES:
        raise ValueError("raw prefix of extended feature schema mismatch")
    screen = json.loads(args.screen.read_text(encoding="utf-8"))
    if int(screen["seed"]) != args.seed:
        raise ValueError("screen/seed mismatch")

    eligible = [
        (
            selection_key(record["selected_scale_record"]),
            name,
            record,
        )
        for name, record in screen["results"].items()
    ]
    _, selected_name, selected_record = max(
        eligible, key=lambda item: item[0]
    )
    if selected_name != screen["selected_direction"]:
        raise ValueError(
            "screen-selected ensemble disagrees with recomputed ranking"
        )

    members = []
    for feature_name, target_name, model_name in CONFIGS:
        matrix = features if feature_name != "raw12" else features[:, :12]
        transformed = transform_target(target_name, targets)
        if transformed is None:
            raise ValueError(target_name)
        model = make_model(model_name, args.seed + 2000)
        model.fit(matrix, transformed)
        members.append(
            {
                "name": f"{feature_name}__{target_name}__{model_name}",
                "feature_family": feature_name,
                "target_transform": target_name,
                "model_name": model_name,
                "model": model,
            }
        )

    artifact = {
        "schema_version": 3,
        "seed": args.seed,
        "raw_feature_names": list(RAW_FEATURE_NAMES),
        "feature_schema": feature_schema,
        "extended_feature_names": list(extended_feature_names),
        "ensemble_members": members,
        "selected_ensemble_rule": selected_name,
        "source_selected_scale": float(
            selected_record["selected_scale"]
        ),
        "selection_record": selected_record,
        "source_features": str(args.features),
        "source_features_sha256": sha256(args.features),
        "source_screen": str(args.screen),
        "source_screen_sha256": sha256(args.screen),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(artifact, handle)
    report = {
        key: value
        for key, value in artifact.items()
        if key != "ensemble_members"
    }
    report["ensemble_members"] = [
        {
            key: value
            for key, value in member.items()
            if key != "model"
        }
        for member in members
    ]
    report["artifact"] = str(args.output)
    report["artifact_sha256"] = sha256(args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
