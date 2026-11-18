#!/usr/bin/env python3
"""Append the NIND unseen-camera extension screen to a frozen manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_json(value: object) -> str:
    serialized = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.base_manifest.read_text(encoding="utf-8"))
    with h5py.File(args.data, "r") as handle:
        rows = int(handle["noisy"].shape[0])
        scenes = [text(value) for value in handle["scene"][:]]
        sensors = [text(value) for value in handle["sensor"][:]]
        conditions = [text(value) for value in handle["condition"][:]]
        isos = [text(value) for value in handle["iso"][:]]
    if rows != 348 or len(scenes) != rows:
        raise ValueError(f"expected 348 NIND unseen rows, found {rows}")
    if len(set(scenes)) != 16:
        raise ValueError(f"expected 16 scenes, found {len(set(scenes))}")

    existing_nind_scenes: set[str] = set()
    for population in manifest["populations"]:
        if not population["id"].startswith("nind_"):
            continue
        existing_nind_scenes.update(
            entry["scene"]
            for entry in population.get("entries", [])
            if "scene" in entry
        )
    overlap = sorted(set(scenes) & existing_nind_scenes)
    if overlap:
        raise ValueError(f"NIND scene overlap: {overlap}")

    entries = [
        {
            "row": row,
            "unit_id": scene,
            "scene": scene,
            "sensor": sensor,
            "condition": condition,
            "iso": iso,
        }
        for row, (scene, sensor, condition, iso) in enumerate(
            zip(scenes, sensors, conditions, isos, strict=True)
        )
    ]
    population = {
        "id": "nind_unseen_camera_16_scenes",
        "role": "external_extension_previously_seen",
        "path": "data_nind_unseen_benchmark/nind_unseen_cameras_256.h5",
        "filename": args.data.name,
        "sha256": sha256(args.data),
        "rows": rows,
        "primary_unit": "scene",
        "primary_unit_count": len(set(scenes)),
        "sensor_group_count": len(set(sensors)),
        "entries": entries,
    }
    manifest["populations"] = [
        item
        for item in manifest["populations"]
        if item["id"] != population["id"]
    ] + [population]
    manifest["nind_overlap_checks"][
        "existing_nind__nind_unseen_camera_extension__scene"
    ] = len(overlap)
    core = [
        {
            key: item[key]
            for key in (
                "id",
                "role",
                "sha256",
                "rows",
                "primary_unit",
                "primary_unit_count",
                "source_entry_count",
                "scene_count",
            )
            if key in item
        }
        for item in manifest["populations"]
    ]
    manifest["population_ids_sha256"] = hash_json(core)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": sha256(args.output),
                "population_hash": manifest["population_ids_sha256"],
                "population": {
                    key: value
                    for key, value in population.items()
                    if key != "entries"
                },
                "nind_scene_overlap": len(overlap),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
