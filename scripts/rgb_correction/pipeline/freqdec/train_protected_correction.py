#!/usr/bin/env python3
"""Develop and freeze an input-only gate for frozen-A RGB correction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import warnings
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .luma_weights import luma_weights
from .model import FixedBandResidualNet, load_initial_state


RAW_FEATURE_NAMES = tuple(
    f"{source}_{channel}_{statistic}"
    for source in ("noisy", "base")
    for channel in ("r", "g", "b")
    for statistic in ("mean", "std")
)
EXT_FEATURE_NAMES = RAW_FEATURE_NAMES + (
    "residual_r_mean",
    "residual_g_mean",
    "residual_b_mean",
    "residual_r_std",
    "residual_g_std",
    "residual_b_std",
    "noisy_luma_mean",
    "noisy_luma_std",
    "base_luma_mean",
    "base_luma_std",
    "noisy_rg_mean",
    "noisy_rg_std",
    "noisy_bg_mean",
    "noisy_bg_std",
    "base_rg_mean",
    "base_rg_std",
    "base_bg_mean",
    "base_bg_std",
    "noisy_gradient_rms",
    "base_gradient_rms",
    "noisy_saturation_mean",
    "noisy_saturation_std",
    "base_saturation_mean",
    "base_saturation_std",
)
GATE_FEATURE_NAMES = EXT_FEATURE_NAMES + (
    "ridge_r",
    "ridge_g",
    "ridge_b",
    "ridge_norm",
    "ridge_luma",
    "ridge_rg",
    "ridge_bg",
)
SCALES = (0.0, 0.25, 0.5, 0.75, 1.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scene_ids_from_metadata(path: Path, source_indices: np.ndarray) -> tuple[np.ndarray, list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) <= int(source_indices.max()):
        raise ValueError("source metadata has fewer rows than the training index")
    scenes = []
    names = []
    for source_index in source_indices:
        name = Path(rows[int(source_index)]["Noisy Image:FILE"]).stem
        match = re.match(r"^(\d+)_SRGB_", name)
        if match is None:
            raise ValueError(f"cannot parse SIDD scene from {name}")
        scenes.append(match.group(1))
        names.append(name)
    return np.asarray(scenes), names


def extended_features(noisy: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    if noisy.shape != base.shape or noisy.ndim != 4:
        raise ValueError("noisy and base must be matching NCHW tensors")
    statistics = []
    for image in (noisy, base):
        statistics.extend(
            (
                image.mean(dim=(-2, -1)),
                image.std(dim=(-2, -1), unbiased=False),
            )
        )
    spatial_mean = (2, 3)
    residual = noisy - base
    residual_mean = residual.mean(dim=spatial_mean)
    residual_std = residual.std(dim=spatial_mean, unbiased=False)

    def luma(image: torch.Tensor) -> torch.Tensor:
        weights = torch.as_tensor(
            luma_weights(),
            dtype=image.dtype,
            device=image.device,
        )
        return torch.sum(image * weights[None, :, None, None], dim=1)

    def gradient_rms(image: torch.Tensor) -> torch.Tensor:
        dx = image[:, :, :, 1:] - image[:, :, :, :-1]
        dy = image[:, :, 1:, :] - image[:, :, :-1, :]
        return torch.sqrt(
            (dx[:, :, :-1].square().mean(dim=(-2, -1))
             + dy[:, :, :, :-1].square().mean(dim=(-2, -1)))
            .mean(dim=1)
            .clamp_min(0.0)
        )

    noisy_luma = luma(noisy)
    base_luma = luma(base)
    noisy_rg = noisy[:, 0] - noisy[:, 1]
    noisy_bg = noisy[:, 2] - noisy[:, 1]
    base_rg = base[:, 0] - base[:, 1]
    base_bg = base[:, 2] - base[:, 1]
    noisy_saturation = noisy.max(dim=1).values - noisy.min(dim=1).values
    base_saturation = base.max(dim=1).values - base.min(dim=1).values

    extra = torch.stack(
        (
            residual_mean[:, 0],
            residual_mean[:, 1],
            residual_mean[:, 2],
            residual_std[:, 0],
            residual_std[:, 1],
            residual_std[:, 2],
            noisy_luma.mean(dim=(-2, -1)),
            noisy_luma.std(dim=(-2, -1), unbiased=False),
            base_luma.mean(dim=(-2, -1)),
            base_luma.std(dim=(-2, -1), unbiased=False),
            noisy_rg.mean(dim=(-2, -1)),
            noisy_rg.std(dim=(-2, -1), unbiased=False),
            noisy_bg.mean(dim=(-2, -1)),
            noisy_bg.std(dim=(-2, -1), unbiased=False),
            base_rg.mean(dim=(-2, -1)),
            base_rg.std(dim=(-2, -1), unbiased=False),
            base_bg.mean(dim=(-2, -1)),
            base_bg.std(dim=(-2, -1), unbiased=False),
            gradient_rms(noisy),
            gradient_rms(base),
            noisy_saturation.mean(dim=(-2, -1)),
            noisy_saturation.std(dim=(-2, -1), unbiased=False),
            base_saturation.mean(dim=(-2, -1)),
            base_saturation.std(dim=(-2, -1), unbiased=False),
        ),
        dim=1,
    )
    raw_statistics = []
    for image in (noisy, base):
        raw_statistics.extend(
            (
                image.mean(dim=spatial_mean),
                image.std(dim=spatial_mean, unbiased=False),
            )
        )
    return torch.cat((*raw_statistics, extra), dim=1)


def ridge_features(extended: np.ndarray) -> np.ndarray:
    return extended[:, : len(RAW_FEATURE_NAMES)]


def add_ridge_features(extended: np.ndarray, ridge_offset: np.ndarray) -> np.ndarray:
    luma = ridge_offset @ luma_weights()
    rg = ridge_offset[:, 0] - ridge_offset[:, 1]
    bg = ridge_offset[:, 2] - ridge_offset[:, 1]
    norm = np.linalg.norm(ridge_offset, axis=1)
    return np.concatenate(
        (
            extended,
            ridge_offset,
            norm[:, None],
            luma[:, None],
            rg[:, None],
            bg[:, None],
        ),
        axis=1,
    )


def extract(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    model = FixedBandResidualNet("A").to(device)
    load_initial_state(model, args.base_checkpoint)
    model.eval()
    with h5py.File(args.train_h5, "r") as handle:
        total = int(handle["noisy"].shape[0])
        source_indices = np.asarray(handle["source_index"][:], dtype=np.int64)
        scenes, crop_names = scene_ids_from_metadata(
            args.source_metadata, source_indices
        )
        feature_batches: list[np.ndarray] = []
        target_batches: list[np.ndarray] = []
        base_mse_batches: list[np.ndarray] = []
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
            with torch.inference_mode():
                base = model(noisy * 2.0 - 1.0)
                base_01 = base * 0.5 + 0.5
                features = extended_features(noisy, base_01)
                targets = ((clean * 2.0 - 1.0) - base).mean(
                    dim=(-2, -1)
                )
                base_mse = (base_01 - clean).square().mean(dim=(1, 2, 3))
            feature_batches.append(features.cpu().numpy().astype(np.float32))
            target_batches.append(targets.cpu().numpy().astype(np.float32))
            base_mse_batches.append(base_mse.cpu().numpy().astype(np.float32))
            if (start // args.batch_size) % 25 == 0:
                print(f"extracted {end}/{total}", flush=True)

    features = np.concatenate(feature_batches)
    targets = np.concatenate(target_batches)
    base_mse = np.concatenate(base_mse_batches)
    np.savez_compressed(
        args.output,
        features=features,
        targets=targets,
        base_mse=base_mse,
        source_indices=source_indices,
        scenes=scenes,
        crop_names=np.asarray(crop_names),
        feature_names=np.asarray(EXT_FEATURE_NAMES),
    )
    metadata = {
        "seed": args.seed,
        "train_h5": str(args.train_h5),
        "train_h5_sha256": sha256(args.train_h5),
        "source_metadata": str(args.source_metadata),
        "source_metadata_sha256": sha256(args.source_metadata),
        "base_checkpoint": str(args.base_checkpoint),
        "base_checkpoint_sha256": sha256(args.base_checkpoint),
        "rows": int(features.shape[0]),
        "feature_count": int(features.shape[1]),
        "feature_names": list(EXT_FEATURE_NAMES),
        "scene_count": int(np.unique(scenes).size),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


def fit_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    alpha: float = 0.01,
) -> object:
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(train_x, train_y)


def linear_gate(seed: int) -> object:
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0))


def tree_gate(seed: int) -> object:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=120,
        max_leaf_nodes=7,
        min_samples_leaf=40,
        l2_regularization=10.0,
        max_bins=128,
        random_state=seed,
    )


def row_mse(base_mse: np.ndarray, b: np.ndarray, t: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    dot = np.einsum("ij,ij->i", b, t)
    norm_sq = np.einsum("ij,ij->i", b, b)
    return np.maximum(
        base_mse - alpha * dot / 6.0 + alpha * alpha * norm_sq / 12.0,
        1e-12,
    )


def row_delta_psnr(
    base_mse: np.ndarray,
    b: np.ndarray,
    t: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    candidate = row_mse(base_mse, b, t, alpha)
    return -10.0 * np.log10(candidate) + 10.0 * np.log10(base_mse)


def scene_deltas(
    scenes: np.ndarray,
    base_mse: np.ndarray,
    b: np.ndarray,
    t: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    deltas = row_delta_psnr(base_mse, b, t, alpha)
    unique = np.unique(scenes)
    return np.asarray([deltas[scenes == scene].mean() for scene in unique])


def scale_score(
    scenes: np.ndarray,
    base_mse: np.ndarray,
    b: np.ndarray,
    t: np.ndarray,
    predictions: np.ndarray,
    scale: float,
) -> dict[str, float]:
    per_scene = scene_deltas(
        scenes, base_mse, b, t, np.clip(scale * predictions, 0.0, 1.0)
    )
    return {
        "scale": float(scale),
        "pass_fraction_scene_ge_minus_0_02": float((per_scene >= -0.02).mean()),
        "worst_scene_delta": float(per_scene.min()),
        "p05_scene_delta": float(np.quantile(per_scene, 0.05)),
        "mean_scene_delta": float(per_scene.mean()),
        "median_scene_delta": float(np.median(per_scene)),
    }


def choose_scale(
    scenes: np.ndarray,
    base_mse: np.ndarray,
    b: np.ndarray,
    t: np.ndarray,
    predictions: np.ndarray,
) -> tuple[float, list[dict[str, float]]]:
    records = [
        scale_score(scenes, base_mse, b, t, predictions, scale)
        for scale in SCALES
    ]

    def key(record: dict[str, float]) -> tuple[float, ...]:
        protected_tail = (
            record["pass_fraction_scene_ge_minus_0_02"] >= 0.95
            and record["p05_scene_delta"] >= 0.0
        )
        return (
            float(protected_tail),
            record["p05_scene_delta"],
            record["mean_scene_delta"],
            record["pass_fraction_scene_ge_minus_0_02"],
            -record["scale"],
        )

    selected = max(records, key=key)
    return selected["scale"], records


def fold_predictions_gate(
    model_factory,
    features: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, list[object]]:
    splitter = GroupKFold(n_splits=5)
    oof = np.full(target.shape, np.nan, dtype=np.float64)
    models = []
    for fold, (train_index, validation_index) in enumerate(
        splitter.split(features, target, groups)
    ):
        model = model_factory(seed + fold)
        model.fit(features[train_index], target[train_index])
        oof[validation_index] = np.clip(
            model.predict(features[validation_index]), 0.0, 1.0
        )
        models.append(model)
    if np.isnan(oof).any():
        raise RuntimeError("gate OOF predictions incomplete")
    return oof, models


def full_gate(
    model_factory,
    features: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> tuple[object, np.ndarray]:
    oof, _ = fold_predictions_gate(
        model_factory, features, target, groups, seed
    )
    model = model_factory(seed)
    model.fit(features, target)
    return model, oof


def evaluate_source(args: argparse.Namespace) -> None:
    payload = np.load(args.features, allow_pickle=False)
    features = np.asarray(payload["features"], dtype=np.float64)
    targets = np.asarray(payload["targets"], dtype=np.float64)
    base_mse = np.asarray(payload["base_mse"], dtype=np.float64)
    scenes = np.asarray(payload["scenes"]).astype(str)
    ridge_x = ridge_features(features)

    outer = GroupKFold(n_splits=5)
    oof = {
        "ridge": np.full_like(targets, np.nan),
        "global": np.zeros(len(scenes), dtype=np.float64),
        "linear": np.zeros(len(scenes), dtype=np.float64),
        "tree": np.zeros(len(scenes), dtype=np.float64),
    }
    fold_records = []
    for fold, (train_index, validation_index) in enumerate(
        outer.split(ridge_x, targets, scenes)
    ):
        train_groups = scenes[train_index]
        inner = GroupKFold(n_splits=5)
        inner_b = np.full((len(train_index), 3), np.nan, dtype=np.float64)
        for inner_train, inner_validation in inner.split(
            ridge_x[train_index], targets[train_index], train_groups
        ):
            ridge = fit_ridge(
                ridge_x[train_index][inner_train],
                targets[train_index][inner_train],
            )
            inner_b[inner_validation] = ridge.predict(
                ridge_x[train_index][inner_validation]
            )
        if np.isnan(inner_b).any():
            raise RuntimeError("nested ridge OOF predictions incomplete")
        alpha_target = np.clip(
            np.einsum("ij,ij->i", inner_b, targets[train_index])
            / np.maximum(np.einsum("ij,ij->i", inner_b, inner_b), 1e-12),
            0.0,
            1.0,
        )
        gate_x_inner = add_ridge_features(
            features[train_index], inner_b
        )

        linear_oof, _ = fold_predictions_gate(
            lambda seed: linear_gate(seed),
            gate_x_inner,
            alpha_target,
            train_groups,
            args.seed + 100,
        )
        tree_oof, _ = fold_predictions_gate(
            lambda seed: tree_gate(seed),
            gate_x_inner,
            alpha_target,
            train_groups,
            args.seed + 200,
        )
        global_calibration, global_scale_records = choose_scale(
            train_groups,
            base_mse[train_index],
            inner_b,
            targets[train_index],
            np.ones(len(train_index), dtype=np.float64),
        )
        linear_calibration, linear_scale_records = choose_scale(
            train_groups,
            base_mse[train_index],
            inner_b,
            targets[train_index],
            linear_oof,
        )
        tree_calibration, tree_scale_records = choose_scale(
            train_groups,
            base_mse[train_index],
            inner_b,
            targets[train_index],
            tree_oof,
        )

        ridge = fit_ridge(ridge_x[train_index], targets[train_index])
        b_validation = ridge.predict(ridge_x[validation_index])
        linear = linear_gate(args.seed + 100)
        linear.fit(gate_x_inner, alpha_target)
        tree = tree_gate(args.seed + 200)
        tree.fit(gate_x_inner, alpha_target)
        gate_x_validation = add_ridge_features(
            features[validation_index], b_validation
        )
        linear_prediction = np.clip(
            linear.predict(gate_x_validation), 0.0, 1.0
        )
        tree_prediction = np.clip(
            tree.predict(gate_x_validation), 0.0, 1.0
        )
        oof["ridge"][validation_index] = b_validation
        oof["global"][validation_index] = global_calibration
        oof["linear"][validation_index] = np.clip(
            linear_calibration * linear_prediction, 0.0, 1.0
        )
        oof["tree"][validation_index] = np.clip(
            tree_calibration * tree_prediction, 0.0, 1.0
        )
        fold_records.append(
            {
                "fold": fold,
                "validation_rows": int(len(validation_index)),
                "validation_scenes": int(np.unique(scenes[validation_index]).size),
                "global_calibration": global_calibration,
                "linear_calibration": linear_calibration,
                "tree_calibration": tree_calibration,
                "global_scale_records": global_scale_records,
                "linear_scale_records": linear_scale_records,
                "tree_scale_records": tree_scale_records,
                "inner_alpha_target_mean": float(alpha_target.mean()),
                "inner_alpha_target_zero_fraction": float((alpha_target == 0).mean()),
            }
        )

    summaries = {}
    for name in ("global", "linear", "tree"):
        delta = row_delta_psnr(
            base_mse, oof["ridge"], targets, oof[name]
        )
        per_scene = scene_deltas(
            scenes, base_mse, oof["ridge"], targets, oof[name]
        )
        summaries[name] = {
            "alpha_mean": float(oof[name].mean()),
            "alpha_zero_fraction": float((oof[name] == 0.0).mean()),
            "alpha_one_fraction": float((oof[name] == 1.0).mean()),
            "row_mean_delta_psnr": float(delta.mean()),
            "scene_mean_delta_psnr": float(per_scene.mean()),
            "scene_median_delta_psnr": float(np.median(per_scene)),
            "scene_p05_delta_psnr": float(np.quantile(per_scene, 0.05)),
            "worst_scene_delta_psnr": float(per_scene.min()),
            "scenes_not_below_minus_0_02": int((per_scene >= -0.02).sum()),
            "scene_count": int(per_scene.size),
        }

    selection_key = lambda name: (
        summaries[name]["scenes_not_below_minus_0_02"]
        >= math.ceil(0.95 * summaries[name]["scene_count"])
        and summaries[name]["scene_p05_delta_psnr"] >= 0.0,
        summaries[name]["scene_p05_delta_psnr"],
        summaries[name]["scene_mean_delta_psnr"],
        summaries[name]["scenes_not_below_minus_0_02"],
        -summaries[name]["alpha_mean"],
    )
    selected = max(summaries, key=selection_key)

    # Fit the final stack. Gate training uses cross-fitted ridge directions;
    # inference uses a final ridge fitted on all source scenes.
    splitter = GroupKFold(n_splits=5)
    oof_ridge = np.full_like(targets, np.nan)
    for train_index, validation_index in splitter.split(
        ridge_x, targets, scenes
    ):
        ridge = fit_ridge(ridge_x[train_index], targets[train_index])
        oof_ridge[validation_index] = ridge.predict(
            ridge_x[validation_index]
        )
    final_ridge = fit_ridge(ridge_x, targets)
    alpha_target = np.clip(
        np.einsum("ij,ij->i", oof_ridge, targets)
        / np.maximum(np.einsum("ij,ij->i", oof_ridge, oof_ridge), 1e-12),
        0.0,
        1.0,
    )
    gate_x_oof = add_ridge_features(features, oof_ridge)
    factory = (
        (lambda seed: linear_gate(seed))
        if selected == "linear"
        else (lambda seed: tree_gate(seed))
        if selected == "tree"
        else None
    )
    if factory is None:
        final_calibration, _ = choose_scale(
            scenes,
            base_mse,
            oof_ridge,
            targets,
            np.ones(len(scenes), dtype=np.float64),
        )
        final_gate = None
        gate_oof = np.ones(len(scenes), dtype=np.float64)
    else:
        gate_oof, _ = fold_predictions_gate(
            factory, gate_x_oof, alpha_target, scenes, args.seed + 300
        )
        final_calibration, _ = choose_scale(
            scenes,
            base_mse,
            oof_ridge,
            targets,
            gate_oof,
        )
        final_gate = factory(args.seed + 300)
        final_gate.fit(gate_x_oof, alpha_target)

    output = {
        "schema_version": 1,
        "seed": args.seed,
        "source_features": str(args.features),
        "source_features_sha256": sha256(args.features),
        "raw_feature_names": list(RAW_FEATURE_NAMES),
        "extended_feature_names": list(EXT_FEATURE_NAMES),
        "gate_feature_names": list(GATE_FEATURE_NAMES),
        "ridge_alpha": 0.01,
        "scale_grid": list(SCALES),
        "selected_candidate": selected,
        "source_group_cv": summaries,
        "outer_folds": fold_records,
        "final": {
            "ridge_model": final_ridge,
            "gate_model": final_gate,
            "global_calibration": final_calibration,
            "alpha_target_mean": float(alpha_target.mean()),
            "alpha_target_zero_fraction": float((alpha_target == 0).mean()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import pickle

    with args.output.open("wb") as handle:
        pickle.dump(output, handle)
    report = {
        key: value
        for key, value in output.items()
        if key != "final"
    }
    report["final"] = {
        key: value
        for key, value in output["final"].items()
        if key not in {"ridge_model", "gate_model"}
    }
    report["artifact"] = str(args.output)
    report["artifact_sha256"] = sha256(args.output)
    args.report.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--seed", type=int, required=True)
    extract_parser.add_argument("--train-h5", type=Path, required=True)
    extract_parser.add_argument("--source-metadata", type=Path, required=True)
    extract_parser.add_argument("--base-checkpoint", type=Path, required=True)
    extract_parser.add_argument("--output", type=Path, required=True)
    extract_parser.add_argument("--batch-size", type=int, default=8)
    extract_parser.add_argument("--device", default="cuda")
    extract_parser.set_defaults(func=extract)

    evaluate_parser = subparsers.add_parser("evaluate-source")
    evaluate_parser.add_argument("--seed", type=int, required=True)
    evaluate_parser.add_argument("--features", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.add_argument("--report", type=Path, required=True)
    evaluate_parser.set_defaults(func=evaluate_source)
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message="Early stopping is set but no validation data is given",
    )
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
