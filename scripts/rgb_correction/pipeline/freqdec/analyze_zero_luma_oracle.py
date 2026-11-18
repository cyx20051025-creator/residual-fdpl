#!/usr/bin/env python3
"""Measure label-aware channel-mean headroom in a frozen model output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from .data import file_sha256
from .luma_weights import luma_weights
from .model import FixedBandResidualNet, load_exact_state, load_initial_state
from .nafnet_adapter import load_frozen_nafnet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", required=True, choices=("A", "C", "E", "NAF"))
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--population", action="append")
    parser.add_argument(
        "--projection",
        choices=("zero_luma", "raw"),
        default="zero_luma",
        help="Oracle correction subspace. Default preserves zero-luma runs.",
    )
    parser.add_argument(
        "--offset-scale",
        type=float,
        default=1.0,
        help="Scale applied to the oracle offset; default preserves full oracle.",
    )
    return parser.parse_args()


def grouped_summary(
    records: list[dict],
    primary_unit: str,
) -> dict[str, object]:
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(str(record[primary_unit]), []).append(record)
    units = []
    for unit, rows in sorted(grouped.items()):
        units.append(
            {
                "unit": unit,
                "count": len(rows),
                "base_psnr": float(np.mean([row["base_psnr"] for row in rows])),
                "oracle_psnr": float(
                    np.mean([row["oracle_psnr"] for row in rows])
                ),
                "psnr_delta": float(
                    np.mean([row["psnr_delta"] for row in rows])
                ),
            }
        )
    deltas = np.asarray([unit["psnr_delta"] for unit in units])
    return {
        "primary_unit": primary_unit,
        "primary_unit_count": len(units),
        "row_count": len(records),
        "metric_means": {
            "base_psnr": float(np.mean([unit["base_psnr"] for unit in units])),
            "oracle_psnr": float(
                np.mean([unit["oracle_psnr"] for unit in units])
            ),
            "psnr_delta": float(deltas.mean()),
            "positive_fraction": float((deltas > 0.0).mean()),
            "median_delta": float(np.median(deltas)),
            "p05": float(np.quantile(deltas, 0.05)),
            "p95": float(np.quantile(deltas, 0.95)),
        },
        "units": units,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if args.architecture == "NAF":
        model = load_frozen_nafnet(args.base_checkpoint, device)
    else:
        model = FixedBandResidualNet(args.architecture).to(device)
        if args.architecture == "A":
            load_initial_state(model, args.base_checkpoint)
        else:
            load_exact_state(model, args.base_checkpoint)
        model.eval()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected_populations = set(args.population or [])
    weight = luma_weights()
    weight_sq = float(weight @ weight)
    populations = []
    for population in manifest["populations"]:
        if (
            selected_populations
            and population["id"] not in selected_populations
        ):
            continue
        data_path = args.data_root / population["filename"]
        if file_sha256(data_path) != population["sha256"]:
            raise ValueError(f"hash mismatch for {population['id']}")
        records = []
        with h5py.File(data_path, "r") as handle:
            total = int(handle["noisy"].shape[0])
            for start in range(0, total, args.batch_size):
                end = min(total, start + args.batch_size)
                noisy = (
                    torch.from_numpy(
                        np.asarray(handle["noisy"][start:end], dtype=np.float32)
                    ).permute(0, 3, 1, 2)
                    / 255.0
                ).to(device)
                clean = (
                    torch.from_numpy(
                        np.asarray(handle["gt"][start:end], dtype=np.float32)
                    ).permute(0, 3, 1, 2)
                    / 255.0
                ).to(device)
                clean_coord = clean * 2.0 - 1.0
                with torch.inference_mode():
                    if args.architecture == "NAF":
                        base_01 = model(noisy)
                        base_coord = base_01 * 2.0 - 1.0
                    else:
                        base_coord = model(noisy * 2.0 - 1.0)
                        base_01 = base_coord * 0.5 + 0.5
                    target = (clean_coord - base_coord).mean(dim=(-2, -1))
                    target_numpy = target.cpu().numpy().astype(np.float64)
                    luma_component = target_numpy @ weight
                    if args.projection == "zero_luma":
                        offset = (
                            target_numpy
                            - luma_component[:, None]
                            * weight[None, :]
                            / weight_sq
                        )
                    else:
                        offset = target_numpy.copy()
                    offset = float(args.offset_scale) * offset
                    offset_tensor = (
                        torch.from_numpy(offset.astype(np.float32))
                        .to(device)
                        .view(-1, 3, 1, 1)
                    )
                    oracle_01 = (
                        (base_coord + offset_tensor) * 0.5 + 0.5
                    ).clamp(0.0, 1.0)
                    base_prediction = base_01.clamp(0.0, 1.0)
                    base_psnr = -10.0 * torch.log10(
                        (base_prediction - clean).square().mean(dim=(1, 2, 3))
                    )
                    oracle_psnr = -10.0 * torch.log10(
                        (oracle_01 - clean).square().mean(dim=(1, 2, 3))
                    )
                    delta = oracle_psnr - base_psnr

                for index in range(end - start):
                    entry = population["entries"][start + index]
                    record = {
                        "row": start + index,
                        "base_psnr": float(base_psnr[index]),
                        "oracle_psnr": float(oracle_psnr[index]),
                        "psnr_delta": float(delta[index]),
                        "target_rgb": target_numpy[index].tolist(),
                        "oracle_offset_rgb": offset[index].tolist(),
                    }
                    for key, value in entry.items():
                        if key != "row":
                            record[key] = value
                    records.append(record)
                print(f"{population['id']} {end}/{total}", flush=True)

        populations.append(
            {
                "population": population["id"],
                "role": population["role"],
                "summary": grouped_summary(records, population["primary_unit"]),
                "records": records,
            }
        )

    payload = {
        "schema_version": 1,
        "diagnostic": (
            "label-aware zero-luma channel-mean oracle"
            if args.projection == "zero_luma"
            else "label-aware raw-RGB channel-mean oracle"
        ),
        "architecture": args.architecture,
        "projection": args.projection,
        "offset_scale": float(args.offset_scale),
        "base_checkpoint": str(args.base_checkpoint),
        "base_checkpoint_sha256": file_sha256(args.base_checkpoint),
        "manifest": str(args.manifest),
        "manifest_sha256": file_sha256(args.manifest),
        "luma_weights": weight.tolist(),
        "populations": populations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                row["population"]: row["summary"]["metric_means"]
                for row in populations
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
