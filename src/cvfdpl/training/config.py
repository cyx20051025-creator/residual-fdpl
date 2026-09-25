"""Locked v3.1 training configurations."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


def _section(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} must be a mapping")
    return value


def _reject_unknown(
    payload: Mapping[str, Any],
    allowed: set[str],
    section: str,
) -> None:
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"unknown {section} configuration keys: {sorted(unknown)}")


@dataclass(frozen=True)
class StageConfig:
    """Configuration for one SIDD fine-tuning stage."""

    patch_size: int
    batch_size: int
    epochs: int
    learning_rate: float
    warmup_epochs: int
    fdpl_weight: float
    target_ratio: float = 0.30
    perceptual_weight: float = 0.01
    weight_decay: float = 1e-4
    calibration_batches: int = 32
    recalibrate_every: int = 5
    weight_map_pairs: int = 15000
    weight_map_batch_size: int = 8
    weight_map_form: str = "log_inv"
    weight_map_clip_quantile: float = 0.01
    weight_map_power: bool = False
    data_chain: str = "bilinear256"
    log_scaling: float | None = None
    seed: int = 42

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> StageConfig:
        _reject_unknown(
            payload,
            {
                "dataset",
                "data_chain",
                "patch_size",
                "batch_size",
                "epochs",
                "optimizer",
                "loss",
                "calibration",
                "weight_map",
                "seed",
            },
            "Stage",
        )
        optimizer = _section(payload, "optimizer")
        loss = _section(payload, "loss")
        calibration = _section(payload, "calibration")
        weight_map = _section(payload, "weight_map")
        _reject_unknown(optimizer, {"name", "learning_rate", "weight_decay"}, "optimizer")
        _reject_unknown(
            loss,
            {
                "name",
                "vgg_weight",
                "max_fdpl_weight",
                "warmup_epochs",
                "target_ratio",
                "log_scaling",
            },
            "loss",
        )
        _reject_unknown(calibration, {"batches", "every_epochs"}, "calibration")
        _reject_unknown(
            weight_map,
            {"pairs", "batch_size", "form", "clip_quantile", "power"},
            "weight_map",
        )
        if payload.get("dataset", "SIDD") != "SIDD":
            raise ValueError("the public SIDD trainer only supports dataset='SIDD'")
        if optimizer.get("name", "AdamW") != "AdamW":
            raise ValueError("only the locked AdamW optimizer is supported")
        if loss.get("name", "ResidualFDPL") != "ResidualFDPL":
            raise ValueError("only ResidualFDPL is supported by the SIDD trainer")
        data_chain = str(payload.get("data_chain", "bilinear256"))
        if data_chain not in {"bilinear256", "crop512"}:
            raise ValueError(f"unknown data chain: {data_chain}")
        weight_form = str(weight_map.get("form", "log_inv"))
        if weight_form not in {"inv", "inv_sq", "log_inv"}:
            raise ValueError(f"unknown weight-map form: {weight_form}")
        fdpl_weight = float(loss["max_fdpl_weight"])
        log_scaling_value = loss.get("log_scaling")
        log_scaling = None if log_scaling_value is None else float(log_scaling_value)
        if log_scaling is not None:
            expected_weight = 0.08 / log_scaling
            if not math.isclose(fdpl_weight, expected_weight, rel_tol=1e-3):
                raise ValueError(
                    "max_fdpl_weight is inconsistent with 0.08 / log_scaling"
                )
        return cls(
            patch_size=int(payload["patch_size"]),
            batch_size=int(payload["batch_size"]),
            epochs=int(payload["epochs"]),
            learning_rate=float(optimizer["learning_rate"]),
            warmup_epochs=int(loss["warmup_epochs"]),
            fdpl_weight=fdpl_weight,
            target_ratio=float(loss.get("target_ratio", 0.30)),
            perceptual_weight=float(loss.get("vgg_weight", 0.01)),
            weight_decay=float(optimizer.get("weight_decay", 1e-4)),
            calibration_batches=int(calibration.get("batches", 32)),
            recalibrate_every=int(calibration.get("every_epochs", 5)),
            weight_map_pairs=int(weight_map.get("pairs", 15000)),
            weight_map_batch_size=int(weight_map.get("batch_size", 8)),
            weight_map_form=weight_form,
            weight_map_clip_quantile=float(weight_map.get("clip_quantile", 0.01)),
            weight_map_power=bool(weight_map.get("power", False)),
            data_chain=data_chain,
            log_scaling=log_scaling,
            seed=int(payload.get("seed", 42)),
        )


@dataclass(frozen=True)
class Stage0Config:
    """Configuration for Urban100 Gaussian pretraining."""

    patch_size: int = 64
    batch_size: int = 8
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    sigma: float = 0.1
    warmup_epochs: int = 10
    fdpl_weight: float = 0.05
    perceptual_weight: float = 0.01
    low_weight: float = 0.2
    high_weight: float = 0.8
    radial_cutoff: float = 0.15
    num_feat: int = 32
    num_grow_ch: int = 16
    num_rrdb: int = 4
    num_rcab: int = 3
    seed: int = 42

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> Stage0Config:
        _reject_unknown(
            payload,
            {
                "dataset",
                "patch_size",
                "batch_size",
                "noise",
                "model",
                "loss",
                "optimizer",
                "seed",
            },
            "Stage 0",
        )
        loss = _section(payload, "loss")
        model = _section(payload, "model")
        optimizer = _section(payload, "optimizer")
        noise = _section(payload, "noise")
        _reject_unknown(
            loss,
            {
                "name",
                "low_weight",
                "high_weight",
                "radial_cutoff",
                "vgg_weight",
                "max_fdpl_weight",
                "warmup_epochs",
            },
            "Stage 0 loss",
        )
        _reject_unknown(
            model,
            {"name", "num_feat", "num_grow_ch", "num_rrdb", "num_rcab"},
            "Stage 0 model",
        )
        _reject_unknown(
            optimizer,
            {"name", "learning_rate", "weight_decay", "scheduler", "epochs"},
            "Stage 0 optimizer",
        )
        _reject_unknown(
            noise,
            {"type", "sigma", "normalized_range"},
            "Stage 0 noise",
        )
        if payload.get("dataset", "Urban100") != "Urban100":
            raise ValueError("Stage 0 only supports the Urban100 dataset")
        if noise.get("type", "gaussian") != "gaussian":
            raise ValueError("Stage 0 only supports Gaussian noise")
        normalized_range = noise.get("normalized_range", [-1.0, 1.0])
        if [float(value) for value in normalized_range] != [-1.0, 1.0]:
            raise ValueError("Stage 0 requires normalized_range=[-1.0, 1.0]")
        if model.get("name", "RRDBPlusLightRCAB") != "RRDBPlusLightRCAB":
            raise ValueError("Stage 0 only supports RRDBPlusLightRCAB")
        if loss.get("name", "OriginalFDPL") != "OriginalFDPL":
            raise ValueError("Stage 0 only supports OriginalFDPL")
        if optimizer.get("name", "AdamW") != "AdamW":
            raise ValueError("Stage 0 only supports AdamW")
        if optimizer.get("scheduler", "cosine") != "cosine":
            raise ValueError("Stage 0 only supports cosine scheduling")
        return cls(
            patch_size=int(payload.get("patch_size", 64)),
            batch_size=int(payload.get("batch_size", 8)),
            epochs=int(optimizer.get("epochs", 50)),
            learning_rate=float(optimizer.get("learning_rate", 1e-4)),
            weight_decay=float(optimizer.get("weight_decay", 1e-4)),
            sigma=float(noise.get("sigma", 0.1)),
            warmup_epochs=int(loss.get("warmup_epochs", 10)),
            fdpl_weight=float(loss.get("max_fdpl_weight", 0.05)),
            perceptual_weight=float(loss.get("vgg_weight", 0.01)),
            low_weight=float(loss.get("low_weight", 0.2)),
            high_weight=float(loss.get("high_weight", 0.8)),
            radial_cutoff=float(loss.get("radial_cutoff", 0.15)),
            num_feat=int(model.get("num_feat", 32)),
            num_grow_ch=int(model.get("num_grow_ch", 16)),
            num_rrdb=int(model.get("num_rrdb", 4)),
            num_rcab=int(model.get("num_rcab", 3)),
            seed=int(payload.get("seed", 42)),
        )


def canonical_stage1() -> StageConfig:
    """Return the locked 64 x 64 Stage 1 configuration."""

    return StageConfig(
        patch_size=64,
        batch_size=8,
        epochs=30,
        learning_rate=1e-5,
        warmup_epochs=5,
        fdpl_weight=0.08,
    )


def canonical_stage2() -> StageConfig:
    """Return the locked 256 x 256 Stage 2 configuration."""

    return StageConfig(
        patch_size=256,
        batch_size=4,
        epochs=10,
        learning_rate=5e-6,
        warmup_epochs=3,
        fdpl_weight=0.08 / 1.37,
        log_scaling=1.37,
    )


def canonical_stage0() -> Stage0Config:
    """Return the locked Urban100 Gaussian pretraining configuration."""

    return Stage0Config()
