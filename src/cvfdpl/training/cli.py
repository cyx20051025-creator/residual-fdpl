"""Command-line interface for canonical training."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from cvfdpl.training.config import (
    Stage0Config,
    StageConfig,
    canonical_stage0,
    canonical_stage1,
    canonical_stage2,
)
from cvfdpl.training.pipeline import run_sidd_training, run_stage0_training


def _load_mapping(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"configuration must be a JSON object: {path}")
    return payload


def _load_stage(path: str | None, fallback: StageConfig) -> StageConfig:
    return fallback if path is None else StageConfig.from_mapping(_load_mapping(path))


def _load_stage0(path: str | None, fallback: Stage0Config) -> Stage0Config:
    return fallback if path is None else Stage0Config.from_mapping(_load_mapping(path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    stage0 = subparsers.add_parser("stage0", help="Urban100 Gaussian pretraining")
    stage0.add_argument("--data-root", required=True)
    stage0.add_argument("--output-dir", required=True)
    stage0.add_argument("--config")
    stage0.add_argument("--device")
    stage0.add_argument("--num-workers", type=int, default=0)
    stage0.add_argument("--vgg-weights", help="local torchvision VGG19 checkpoint")
    stage0.add_argument(
        "--deterministic",
        action="store_true",
        help="enable cuDNN deterministic algorithms (not the archived default)",
    )

    sidd = subparsers.add_parser("sidd", help="Stage 1 -> Stage 2 SIDD fine-tuning")
    sidd.add_argument("--h5", required=True, help="paired HDF5 cache")
    sidd.add_argument("--output-dir", required=True)
    sidd.add_argument("--pretrained", help="Stage 0 checkpoint")
    sidd.add_argument("--stage1-config")
    sidd.add_argument("--stage2-config")
    sidd.add_argument(
        "--data-chain",
        choices=["bilinear256", "crop512"],
        help="main paper chain is bilinear256; crop512 is the clean-chain variant",
    )
    sidd.add_argument("--device")
    sidd.add_argument("--num-workers", type=int, default=0)
    sidd.add_argument("--smoke", type=int, default=0)
    sidd.add_argument("--force-maps", action="store_true")
    sidd.add_argument("--no-fdpl", action="store_true")
    sidd.add_argument("--ema-decay", type=float, default=0.999)
    sidd.add_argument("--vgg-weights", help="local torchvision VGG19 checkpoint")
    sidd.add_argument(
        "--deterministic",
        action="store_true",
        help="enable cuDNN deterministic algorithms (not the archived default)",
    )
    sidd.add_argument(
        "--allow-partial-pretrained",
        action="store_true",
        help="allow missing checkpoint keys; default requires an exact match",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "stage0":
        config = _load_stage0(args.config, canonical_stage0())
        result = run_stage0_training(
            args.data_root,
            args.output_dir,
            config,
            device=args.device,
            num_workers=args.num_workers,
            vgg_weights=args.vgg_weights,
            deterministic=args.deterministic,
        )
    else:
        stage1 = _load_stage(args.stage1_config, canonical_stage1())
        stage2 = _load_stage(args.stage2_config, canonical_stage2())
        if args.data_chain is not None:
            stage1 = replace(stage1, data_chain=args.data_chain)
            stage2 = replace(stage2, data_chain=args.data_chain)
        result = run_sidd_training(
            args.h5,
            args.output_dir,
            stage1,
            stage2,
            pretrained=args.pretrained,
            device=args.device,
            num_workers=args.num_workers,
            smoke=args.smoke,
            force_maps=args.force_maps,
            enable_fdpl=not args.no_fdpl,
            ema_decay=args.ema_decay,
            vgg_weights=args.vgg_weights,
            deterministic=args.deterministic,
            allow_partial_pretrained=args.allow_partial_pretrained,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
