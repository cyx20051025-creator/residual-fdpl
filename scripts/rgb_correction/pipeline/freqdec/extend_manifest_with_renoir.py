#!/usr/bin/env python3
"""Append the corrected RENOIR T3I RGB screen to a frozen manifest."""

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
    parser.add_argument("--renoir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.base_manifest.read_text(encoding="utf-8"))
    with h5py.File(args.renoir, "r") as handle:
        scenes = [text(value) for value in handle["scene"][:]]
        rows = int(handle["noisy"].shape[0])
    if rows != 37 or len(scenes) != rows:
        raise ValueError(f"expected 37 RENOIR rows, found {rows}")
    entries = [
        {
            "row": row,
            "unit_id": scene,
            "scene": scene,
        }
        for row, scene in enumerate(scenes)
    ]
    population = {
        "id": "renoir_t3i_37_scenes",
        "role": "external_extension_previously_seen",
        "path": "data_renoir_benchmark/renoir_t3i_512x37_rgb.h5",
        "filename": args.renoir.name,
        "sha256": sha256(args.renoir),
        "rows": rows,
        "primary_unit": "scene",
        "primary_unit_count": len(set(scenes)),
        "entries": entries,
    }
    manifest["populations"] = [
        item
        for item in manifest["populations"]
        if item["id"] != population["id"]
    ] + [population]
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
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
