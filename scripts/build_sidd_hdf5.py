#!/usr/bin/env python3
"""Build a paired SIDD HDF5 cache without hard-coded paths."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cvfdpl.data.sidd import find_pairs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noisy-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--compression", choices=["none", "gzip", "lzf"], default="gzip")
    parser.add_argument("--compression-opts", type=int, default=4)
    parser.add_argument("--interpolation", choices=["bicubic", "bilinear"], default="bicubic")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=500)
    return parser.parse_args()


def compression_kwargs(name: str, level: int) -> dict[str, object]:
    if name == "gzip":
        return {"compression": "gzip", "compression_opts": level, "shuffle": True}
    if name == "lzf":
        return {"compression": "lzf", "shuffle": True}
    return {}


def load_image(path: Path, crop_size: int, interpolation: str) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (crop_size, crop_size):
            method = Image.Resampling.BICUBIC
            if interpolation == "bilinear":
                method = Image.Resampling.BILINEAR
            image = image.resize((crop_size, crop_size), method)
        return np.asarray(image, dtype=np.uint8)


def main() -> None:
    args = parse_args()
    pairs = find_pairs(args.noisy_dir, args.gt_dir)
    if args.limit > 0:
        pairs = pairs[: args.limit]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset_kwargs = compression_kwargs(args.compression, args.compression_opts)
    started = time.time()

    with h5py.File(output, "w") as handle:
        noisy_dataset = handle.create_dataset(
            "noisy",
            shape=(len(pairs), args.crop_size, args.crop_size, 3),
            dtype="uint8",
            chunks=(1, args.crop_size, args.crop_size, 3),
            **dataset_kwargs,
        )
        clean_dataset = handle.create_dataset(
            "gt",
            shape=(len(pairs), args.crop_size, args.crop_size, 3),
            dtype="uint8",
            chunks=(1, args.crop_size, args.crop_size, 3),
            **dataset_kwargs,
        )
        for index, (noisy_path, clean_path) in enumerate(pairs):
            noisy_dataset[index] = load_image(
                noisy_path,
                args.crop_size,
                args.interpolation,
            )
            clean_dataset[index] = load_image(
                clean_path,
                args.crop_size,
                args.interpolation,
            )
            if (index + 1) % args.progress_every == 0 or index + 1 == len(pairs):
                elapsed = time.time() - started
                print(f"[{index + 1}/{len(pairs)}] elapsed={elapsed:.1f}s")

    print(f"pairs={len(pairs)}")
    print(f"crop_size={args.crop_size}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
