"""Evaluate one endpoint on the frozen population manifest."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.nn import functional as F

from .data import file_sha256
from .model import FixedBandResidualNet
from .operators import GaussianLowPass, channel_mean, remove_channel_mean


def make_gaussian_ssim_window(
    window_size: int,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    coordinates = torch.arange(window_size, dtype=torch.float32, device=device)
    values = torch.exp(
        -(coordinates - window_size // 2).square() / (2 * 1.5**2)
    )
    values = values / values.sum()
    window = (values[:, None] * values[None, :])[None, None]
    return window.expand(channels, 1, window_size, window_size).to(dtype)


def ssim_per_image(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    channels = prediction.shape[1]
    window = make_gaussian_ssim_window(
        11, channels, prediction.device, prediction.dtype
    )
    c1, c2 = 0.01**2, 0.03**2
    mean_prediction = F.conv2d(
        prediction, window, padding=5, groups=channels
    )
    mean_target = F.conv2d(target, window, padding=5, groups=channels)
    var_prediction = (
        F.conv2d(prediction.square(), window, padding=5, groups=channels)
        - mean_prediction.square()
    )
    var_target = (
        F.conv2d(target.square(), window, padding=5, groups=channels)
        - mean_target.square()
    )
    covariance = (
        F.conv2d(prediction * target, window, padding=5, groups=channels)
        - mean_prediction * mean_target
    )
    score = (
        (2 * mean_prediction * mean_target + c1)
        * (2 * covariance + c2)
        / (
            (mean_prediction.square() + mean_target.square() + c1)
            * (var_prediction + var_target + c2)
        )
    )
    return score.mean((1, 2, 3))


def radial_masks(
    height: int,
    width: int,
    device: torch.device,
) -> list[torch.Tensor]:
    fy = torch.fft.fftshift(torch.fft.fftfreq(height, device=device))
    fx = torch.fft.fftshift(torch.fft.fftfreq(width, device=device))
    radius = torch.sqrt(fy[:, None].square() + fx[None, :].square()) / math.sqrt(0.5)
    edges = (0.0, 0.2, 0.4, 0.6, 0.8, 1.00001)
    return [
        (radius >= low) & (radius < high)
        for low, high in zip(edges[:-1], edges[1:])
    ]


def component_diagnostics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    low_pass: GaussianLowPass,
) -> dict[str, torch.Tensor]:
    error = prediction - target
    dc = channel_mean(error)
    low = remove_channel_mean(low_pass(error))
    high = remove_channel_mean(error - low_pass(error))
    masks = radial_masks(error.shape[-2], error.shape[-1], error.device)
    energy = torch.fft.fftshift(
        torch.fft.fft2(error, norm="ortho"), dim=(-2, -1)
    ).abs().square().sum(1)
    bands = torch.stack([energy[:, mask].mean(1) for mask in masks], dim=1)
    return {
        "dc_mse": dc.square().mean((1, 2, 3)),
        "lf_mse": low.square().mean((1, 2, 3)),
        "hf_mse": high.square().mean((1, 2, 3)),
        "radial_band_mse": bands,
    }


def group_summary(
    records: list[dict[str, object]],
    primary_unit: str,
) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[str(record[primary_unit])].append(record)
    metric_names = ("psnr", "ssim", "lpips", "dists", "dc_mse", "lf_mse", "hf_mse")
    summaries: list[dict[str, object]] = []
    for unit, rows in sorted(grouped.items()):
        summary = {"unit": unit, "count": len(rows)}
        for metric in metric_names:
            summary[metric] = float(np.mean([float(row[metric]) for row in rows]))
        summary["radial_band_mse"] = np.mean(
            [row["radial_band_mse"] for row in rows], axis=0
        ).tolist()
        summaries.append(summary)
    metric_means = {
        metric: float(np.mean([row[metric] for row in summaries]))
        for metric in metric_names
    }
    metric_means["radial_band_mse"] = np.mean(
        [row["radial_band_mse"] for row in summaries], axis=0
    ).tolist()
    return {
        "primary_unit": primary_unit,
        "primary_unit_count": len(summaries),
        "row_count": len(records),
        "metric_means": metric_means,
        "units": summaries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--arm", choices=["A", "C", "E"], required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--population", action="append")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import lpips
    import piq

    device = torch.device(args.device)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = FixedBandResidualNet(args.arm).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    low_pass = model.low_pass
    lpips_model = lpips.LPIPS(net="alex", pretrained=True, verbose=False).to(device)
    lpips_model.eval()
    dists_model = piq.DISTS(reduction="none").to(device).eval()

    selected = set(args.population or [])
    results = []
    for population in manifest["populations"]:
        if selected and population["id"] not in selected:
            continue
        data_path = args.data_root / population["filename"]
        if file_sha256(data_path) != population["sha256"]:
            raise ValueError(f"hash mismatch for {population['id']}")
        records: list[dict[str, object]] = []
        with h5py.File(data_path, "r") as handle:
            total = int(handle["noisy"].shape[0])
            for start in range(0, total, args.batch_size):
                end = min(start + args.batch_size, total)
                noisy = torch.from_numpy(
                    np.asarray(handle["noisy"][start:end], dtype=np.float32)
                ).permute(0, 3, 1, 2) / 255.0
                clean = torch.from_numpy(
                    np.asarray(handle["gt"][start:end], dtype=np.float32)
                ).permute(0, 3, 1, 2) / 255.0
                noisy = noisy.to(device)
                clean = clean.to(device)
                with torch.inference_mode():
                    raw = model(noisy * 2.0 - 1.0)
                    prediction = (raw * 0.5 + 0.5).clamp(0.0, 1.0)
                    mse = (prediction - clean).square().mean((1, 2, 3))
                    psnr = -10.0 * torch.log10(mse)
                    similarity = ssim_per_image(prediction, clean)
                    lpips_value = lpips_model(
                        prediction * 2.0 - 1.0,
                        clean * 2.0 - 1.0,
                    ).flatten()
                    dists_value = dists_model(prediction, clean).flatten()
                    diagnostics = component_diagnostics(
                        prediction, clean, low_pass
                    )
                for offset in range(end - start):
                    source_entry = population["entries"][start + offset]
                    record = {
                        "row": start + offset,
                        "psnr": float(psnr[offset]),
                        "ssim": float(similarity[offset]),
                        "lpips": float(lpips_value[offset]),
                        "dists": float(dists_value[offset]),
                        "dc_mse": float(diagnostics["dc_mse"][offset]),
                        "lf_mse": float(diagnostics["lf_mse"][offset]),
                        "hf_mse": float(diagnostics["hf_mse"][offset]),
                        "radial_band_mse": [
                            float(value[offset])
                            for value in diagnostics["radial_band_mse"].T
                        ],
                    }
                    for key, value in source_entry.items():
                        if key != "row":
                            record[key] = value
                    records.append(record)
        summary = group_summary(records, population["primary_unit"])
        results.append(
            {
                "population": population["id"],
                "role": population["role"],
                "summary": summary,
                "records": records,
            }
        )
        print(
            json.dumps(
                {
                    "population": population["id"],
                    **summary["metric_means"],
                }
            ),
            flush=True,
        )

    result = {
        "arm": args.arm,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "manifest": str(args.manifest),
        "manifest_sha256": file_sha256(args.manifest),
        "metrics": {
            "psnr": "higher",
            "ssim": "higher",
            "lpips": "lower",
            "dists": "lower",
        },
        "populations": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
