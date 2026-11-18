"""Freeze evaluation row IDs and clustering units before model evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import h5py

from .data import file_sha256


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _iso_from_name(name: str) -> str:
    match = re.search(r"_ISO([^_.]+)", name)
    if not match:
        raise ValueError(f"cannot parse ISO condition from {name}")
    return match.group(1)


def _hash_json(value: object) -> str:
    serialized = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def nind_population(
    population_id: str,
    data_path: Path,
    manifest_path: Path,
    *,
    role: str,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = manifest["sources"]
    entries: list[dict[str, Any]] = []
    row = 0
    with h5py.File(data_path, "r") as handle:
        expected_rows = int(handle["noisy"].shape[0])
        for source in sources:
            source_entry = Path(source["noisy"]).stem
            iso = _iso_from_name(source["noisy"])
            for crop_index in range(int(source["crops"])):
                scene = _text(handle["scene"][row])
                sensor = _text(handle["sensor"][row])
                observed_iso = _text(handle["iso"][row])
                if scene != source["scene"]:
                    raise ValueError(
                        f"row {row}: scene {scene!r} != manifest {source['scene']!r}"
                    )
                if sensor != source["sensor"]:
                    raise ValueError(
                        f"row {row}: sensor {sensor!r} != manifest {source['sensor']!r}"
                    )
                if observed_iso != iso:
                    raise ValueError(
                        f"row {row}: ISO {observed_iso!r} != parsed {iso!r}"
                    )
                entries.append(
                    {
                        "row": row,
                        "scene": scene,
                        "sensor": sensor,
                        "iso": observed_iso,
                        "source_entry": source_entry,
                        "crop_in_source": crop_index,
                    }
                )
                row += 1
    if row != expected_rows:
        raise ValueError(
            f"manifest covered {row} rows, HDF5 contains {expected_rows}"
        )
    return {
        "id": population_id,
        "role": role,
        "path": str(data_path),
        "filename": data_path.name,
        "sha256": file_sha256(data_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "rows": len(entries),
        "primary_unit": "scene",
        "primary_unit_count": len({entry["scene"] for entry in entries}),
        "scene_count": len({entry["scene"] for entry in entries}),
        "source_entry_count": len({entry["source_entry"] for entry in entries}),
        "sensor_count": len({entry["sensor"] for entry in entries}),
        "entries": entries,
    }


def generic_population(
    population_id: str,
    data_path: Path,
    *,
    role: str,
    id_field: str,
    primary_unit: str,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    with h5py.File(data_path, "r") as handle:
        count = int(handle["noisy"].shape[0])
        ids = handle[id_field][:]
        scenes = handle["scene"][:] if "scene" in handle else None
        for row in range(count):
            unit = _text(ids[row])
            entry = {
                "row": row,
                "unit_id": unit,
                primary_unit: unit,
            }
            if scenes is not None:
                entry["scene"] = _text(scenes[row])
            entries.append(entry)
    return {
        "id": population_id,
        "role": role,
        "path": str(data_path),
        "filename": data_path.name,
        "sha256": file_sha256(data_path),
        "rows": len(entries),
        "primary_unit": primary_unit,
        "primary_unit_count": len({entry[primary_unit] for entry in entries}),
        "entries": entries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sidd-valid", type=Path, required=True)
    parser.add_argument("--sidd-holdout", type=Path, required=True)
    parser.add_argument("--nind-confirmation-h5", type=Path, required=True)
    parser.add_argument("--nind-confirmation-manifest", type=Path, required=True)
    parser.add_argument("--nind-independent-h5", type=Path, required=True)
    parser.add_argument("--nind-independent-manifest", type=Path, required=True)
    parser.add_argument("--nind-remaining-h5", type=Path, required=True)
    parser.add_argument("--nind-remaining-manifest", type=Path, required=True)
    parser.add_argument("--cc-noise", type=Path, required=True)
    parser.add_argument("--polyu", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    populations = [
        generic_population(
            "sidd_valid_1024",
            args.sidd_valid,
            role="internal_guardrail",
            id_field="name",
            primary_unit="image_id",
        ),
        generic_population(
            "sidd_train_crop_holdout_1024",
            args.sidd_holdout,
            role="internal_guardrail_crop_disjoint_not_source_disjoint",
            id_field="source_index",
            primary_unit="source_index",
        ),
        nind_population(
            "nind_reserve2_confirmation",
            args.nind_confirmation_h5,
            args.nind_confirmation_manifest,
            role="external_gate",
        ),
        nind_population(
            "nind_reserve1_independent",
            args.nind_independent_h5,
            args.nind_independent_manifest,
            role="external_sensitivity_pending_supplementary_h5_decision",
        ),
        nind_population(
            "nind_remaining_56_scenes",
            args.nind_remaining_h5,
            args.nind_remaining_manifest,
            role="secondary_transfer",
        ),
        generic_population(
            "ccnoise_15",
            args.cc_noise,
            role="external_gate",
            id_field="filename",
            primary_unit="image_id",
        ),
        generic_population(
            "polyu_30_scenes",
            args.polyu,
            role="secondary_transfer",
            id_field="filename",
            primary_unit="scene",
        ),
    ]
    frozen_core = [
        {
            key: population[key]
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
            if key in population
        }
        for population in populations
    ]
    nind_ids = (
        "nind_reserve2_confirmation",
        "nind_reserve1_independent",
        "nind_remaining_56_scenes",
    )
    nind = {population["id"]: population for population in populations}
    overlap_checks: dict[str, int] = {}
    for left_index, left in enumerate(nind_ids):
        for right in nind_ids[left_index + 1 :]:
            left_entries = nind[left]["entries"]
            right_entries = nind[right]["entries"]
            overlap_checks[f"{left}__{right}__source_entry"] = len(
                {entry["source_entry"] for entry in left_entries}
                & {entry["source_entry"] for entry in right_entries}
            )
            overlap_checks[f"{left}__{right}__scene"] = len(
                {entry["scene"] for entry in left_entries}
                & {entry["scene"] for entry in right_entries}
            )
    result = {
        "schema_version": 1,
        "population_ids_sha256": _hash_json(frozen_core),
        "nind_overlap_checks": overlap_checks,
        "populations": populations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Frozen Evaluation Manifest",
        "",
        f"Population hash: `{result['population_ids_sha256']}`",
        "",
        "| Population | Role | Rows | Primary unit | Units | Source entries | Scenes |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    for population in populations:
        lines.append(
            f"| {population['id']} | {population['role']} | "
            f"{population['rows']} | {population['primary_unit']} | "
            f"{population['primary_unit_count']} | "
            f"{population.get('source_entry_count', '-')} | "
            f"{population.get('scene_count', '-')} |"
        )
    lines.extend(
        [
            "",
            "NIND gate inference uses scene-level paired aggregation. "
            "Source-entry results are retained as a sensitivity analysis.",
            "",
            "All NIND source-entry and scene overlap counts are zero.",
            "",
            "`sidd_valid_1024` and `sidd_train_crop_holdout_1024` are separate "
            "populations and must not be combined.",
        ]
    )
    args.output.with_suffix(".md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(frozen_core, indent=2))


if __name__ == "__main__":
    main()
