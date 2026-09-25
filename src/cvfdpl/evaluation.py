"""Paired-image evaluation for the canonical 3-RCAB checkpoint."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import torch

from cvfdpl.data import SIDDPairedDataset
from cvfdpl.metrics import calculate_psnr, calculate_ssim
from cvfdpl.models import RRDBPlusLightRCAB
from cvfdpl.training.checkpoint import load_state_dict
from cvfdpl.utils import get_device


def load_model(
    checkpoint: str | Path,
    num_rcab: int,
    device: torch.device,
) -> RRDBPlusLightRCAB:
    """Load a 3-RCAB checkpoint with strict key validation."""

    model = RRDBPlusLightRCAB(num_rrdb=4, num_rcab=num_rcab).to(device)
    state_dict = load_state_dict(checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def direct_inference(
    model: torch.nn.Module,
    noisy: torch.Tensor,
) -> torch.Tensor:
    """Run one full-image forward pass."""

    return model(noisy)


def sliding_inference(
    model: torch.nn.Module,
    noisy: torch.Tensor,
    tile_size: int,
    stride: int,
) -> torch.Tensor:
    """Merge overlapping tile predictions with a coverage mean."""

    _, _, height, width = noisy.shape
    if tile_size > height or tile_size > width:
        return direct_inference(model, noisy)
    output = torch.zeros_like(noisy)
    counts = torch.zeros(1, 1, height, width, device=noisy.device, dtype=noisy.dtype)
    top_positions = list(range(0, height - tile_size + 1, stride))
    left_positions = list(range(0, width - tile_size + 1, stride))
    if top_positions[-1] != height - tile_size:
        top_positions.append(height - tile_size)
    if left_positions[-1] != width - tile_size:
        left_positions.append(width - tile_size)
    for top in top_positions:
        for left in left_positions:
            tile = noisy[:, :, top : top + tile_size, left : left + tile_size]
            prediction = model(tile)
            output[:, :, top : top + tile_size, left : left + tile_size] += prediction
            counts[:, :, top : top + tile_size, left : left + tile_size] += 1
    return output / counts


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty metric list")
    return {
        "mean": sum(values) / len(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def run_evaluation(
    checkpoint: str | Path,
    noisy_dir: str | Path,
    gt_dir: str | Path,
    *,
    protocol: str = "direct",
    tile_size: int = 128,
    stride: int = 96,
    num_rcab: int = 3,
    device: str | None = None,
    limit: int = 0,
) -> dict[str, Any]:
    """Evaluate all matching image pairs and return JSON-ready results."""

    if protocol not in {"direct", "sliding"}:
        raise ValueError("protocol must be 'direct' or 'sliding'")
    resolved_device = get_device(device)
    dataset = SIDDPairedDataset(noisy_dir, gt_dir)
    model = load_model(checkpoint, num_rcab, resolved_device)
    per_image: list[dict[str, object]] = []
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    pair_count = len(dataset) if limit <= 0 else min(limit, len(dataset))

    with torch.no_grad():
        for index in range(pair_count):
            noisy, clean = dataset[index]
            noisy = noisy.unsqueeze(0).to(resolved_device)
            clean = clean.unsqueeze(0).to(resolved_device)
            if protocol == "sliding":
                prediction = sliding_inference(model, noisy, tile_size, stride)
            else:
                prediction = direct_inference(model, noisy)
            prediction = (prediction * 0.5 + 0.5).clamp(0, 1)
            clean = (clean * 0.5 + 0.5).clamp(0, 1)
            psnr = calculate_psnr(prediction, clean)
            ssim = calculate_ssim(prediction, clean)
            psnr_values.append(psnr)
            ssim_values.append(ssim)
            per_image.append(
                {
                    "index": index,
                    "noisy": dataset.pairs[index][0].name,
                    "gt": dataset.pairs[index][1].name,
                    "psnr": psnr,
                    "ssim": ssim,
                }
            )

    return {
        "checkpoint": str(checkpoint),
        "device": str(resolved_device),
        "protocol": protocol,
        "tile_size": tile_size if protocol == "sliding" else None,
        "stride": stride if protocol == "sliding" else None,
        "images": pair_count,
        "summary": {
            "psnr": _summary(psnr_values),
            "ssim": _summary(ssim_values),
        },
        "per_image": per_image,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--noisy-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--protocol", choices=["direct", "sliding"], default="direct")
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=96)
    parser.add_argument("--num-rcab", type=int, default=3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-json", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_evaluation(
        args.checkpoint,
        args.noisy_dir,
        args.gt_dir,
        protocol=args.protocol,
        tile_size=args.tile_size,
        stride=args.stride,
        num_rcab=args.num_rcab,
        device=args.device,
        limit=args.limit,
    )
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"protocol={result['protocol']}")
    print(f"images={result['images']}")
    print(f"mean_psnr={result['summary']['psnr']['mean']:.6f}")
    print(f"mean_ssim={result['summary']['ssim']['mean']:.6f}")
    if args.output_json:
        print(f"output_json={args.output_json}")


if __name__ == "__main__":
    main()
