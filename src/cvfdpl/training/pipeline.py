"""End-to-end canonical Residual FDPL training pipelines."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset, random_split

from cvfdpl.data import HDF5PairDataset
from cvfdpl.losses import (
    BrightnessAwareLoss,
    OriginalFDPLoss,
    PerChannelLogFDPLoss,
    VGGPerceptualLoss,
)
from cvfdpl.models import RRDBPlusLightRCAB, count_parameters
from cvfdpl.training.checkpoint import EMAModel, load_state_dict
from cvfdpl.training.config import Stage0Config, StageConfig
from cvfdpl.training.data import GaussianUrban100Dataset
from cvfdpl.training.trainer import (
    build_sidd_loaders,
    train_stage,
    validate_model,
)
from cvfdpl.training.weight_map import get_or_compute_weight_map
from cvfdpl.utils import get_device, set_seed


def _load_initial_model(
    checkpoint: str | Path | None,
    device: torch.device,
    num_rcab: int = 3,
    *,
    allow_partial: bool = False,
) -> RRDBPlusLightRCAB:
    model = RRDBPlusLightRCAB(num_rrdb=4, num_rcab=num_rcab).to(device)
    if checkpoint is None:
        return model
    state_dict = load_state_dict(checkpoint, map_location=device)
    if not allow_partial:
        model.load_state_dict(state_dict, strict=True)
        return model
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected checkpoint keys: {sorted(unexpected)}")
    if missing:
        print(
            f"initialized with {len(state_dict) - len(unexpected)} checkpoint keys; "
            f"{len(missing)} keys use random initialization"
        )
    return model


def run_stage0_training(
    data_root: str | Path,
    output_dir: str | Path,
    config: Stage0Config,
    *,
    device: str | None = None,
    num_workers: int = 0,
    vgg_weights: str | Path | None = None,
    deterministic: bool = False,
) -> dict[str, object]:
    """Run Urban100 Gaussian pretraining with Original FDPL."""

    set_seed(config.seed, deterministic=deterministic)
    resolved_device = get_device(device)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    dataset = GaussianUrban100Dataset(
        data_root,
        size=config.patch_size,
        sigma=config.sigma,
    )
    if len(dataset) < 2:
        raise ValueError("at least two Urban100 images are required")
    train_size = int(0.8 * len(dataset))
    train_size = min(max(train_size, 1), len(dataset) - 1)
    train_split, validation_split = random_split(
        dataset,
        [train_size, len(dataset) - train_size],
        generator=torch.Generator().manual_seed(config.seed),
    )
    loader_kwargs = {
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        train_split,
        batch_size=config.batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    validation_loader = DataLoader(
        validation_split,
        batch_size=config.batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    model = RRDBPlusLightRCAB(
        num_feat=config.num_feat,
        num_grow_ch=config.num_grow_ch,
        num_rrdb=config.num_rrdb,
        num_rcab=config.num_rcab,
    ).to(resolved_device)
    mse_fn = BrightnessAwareLoss().to(resolved_device)
    perceptual_fn = VGGPerceptualLoss(weights_path=vgg_weights).to(resolved_device)
    fdpl_fn = OriginalFDPLoss(
        low_weight=config.low_weight,
        high_weight=config.high_weight,
        radial_cutoff=config.radial_cutoff,
    ).to(resolved_device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    history: list[dict[str, object]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        schedule_weight = min(
            config.fdpl_weight,
            config.fdpl_weight * epoch / config.warmup_epochs,
        )
        for noisy, clean in train_loader:
            noisy = noisy.to(resolved_device)
            clean = clean.to(resolved_device)
            prediction = model(noisy)
            prediction_01 = (prediction * 0.5 + 0.5).clamp(0, 1)
            clean_01 = (clean * 0.5 + 0.5).clamp(0, 1)
            loss = (
                mse_fn(prediction, clean)
                + config.perceptual_weight * perceptual_fn(prediction_01, clean_01)
                + schedule_weight * fdpl_fn(prediction, clean)
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item())
        scheduler.step()
        psnr, ssim = validate_model(model, validation_loader, resolved_device)
        history.append(
            {
                "stage": 0,
                "epoch": epoch,
                "loss": round(total_loss / len(train_loader), 6),
                "fdpl_weight": round(schedule_weight, 6),
                "val_psnr": round(psnr, 6),
                "val_ssim": round(ssim, 6),
            }
        )
        print(
            f"Stage 0 epoch {epoch:02d}/{config.epochs} | "
            f"loss={total_loss / len(train_loader):.6f} | "
            f"PSNR={psnr:.4f} | SSIM={ssim:.6f}",
            flush=True,
        )

    checkpoint_path = output_path / "stage0.pth"
    torch.save(model.state_dict(), checkpoint_path)
    result = {
        "config": asdict(config),
        "parameters": count_parameters(model),
        "deterministic": deterministic,
        "checkpoint": str(checkpoint_path),
        "history": history,
    }
    (output_path / "stage0_history.json").write_text(
        json.dumps(history, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_path / "stage0_results.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def run_sidd_training(
    h5_path: str | Path,
    output_dir: str | Path,
    stage1: StageConfig,
    stage2: StageConfig,
    *,
    pretrained: str | Path | None = None,
    device: str | None = None,
    num_workers: int = 0,
    smoke: int = 0,
    force_maps: bool = False,
    enable_fdpl: bool = True,
    ema_decay: float = 0.999,
    vgg_weights: str | Path | None = None,
    deterministic: bool = False,
    allow_partial_pretrained: bool = False,
) -> dict[str, object]:
    """Run the locked Stage 1 -> Stage 2 SIDD fine-tuning pipeline."""

    if stage1.seed != stage2.seed:
        raise ValueError("Stage 1 and Stage 2 must use the same seed")
    if stage1.data_chain != stage2.data_chain:
        raise ValueError("Stage 1 and Stage 2 must use the same data chain")
    set_seed(stage1.seed, deterministic=deterministic)
    resolved_device = get_device(device)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    map_64 = None
    map_256 = None
    if enable_fdpl:
        if stage1.data_chain == "bilinear256":
            map_dataset_64 = HDF5PairDataset(
                h5_path,
                resize_size=stage1.patch_size,
            )
            map_dataset_256 = HDF5PairDataset(
                h5_path,
                resize_size=stage2.patch_size,
            )
        elif stage1.data_chain == "crop512":
            map_dataset_64 = HDF5PairDataset(
                h5_path,
                crop_size=stage1.patch_size,
                crop_mode="random",
            )
            map_dataset_256 = HDF5PairDataset(
                h5_path,
                crop_size=stage2.patch_size,
                crop_mode="random",
            )
        else:
            raise ValueError(f"unknown data chain: {stage1.data_chain}")
        if smoke > 0:
            map_dataset_64 = Subset(
                map_dataset_64,
                range(min(smoke, len(map_dataset_64))),
            )
            map_dataset_256 = Subset(
                map_dataset_256,
                range(min(smoke, len(map_dataset_256))),
            )
        map_64 = get_or_compute_weight_map(
            map_dataset_64,
            stage1.patch_size,
            output_path,
            num_pairs=stage1.weight_map_pairs,
            batch_size=stage1.weight_map_batch_size,
            seed=stage1.seed,
            form=stage1.weight_map_form,
            use_power=stage1.weight_map_power,
            clip_quantile=stage1.weight_map_clip_quantile,
            num_workers=num_workers,
            force=force_maps,
        )
        map_256 = get_or_compute_weight_map(
            map_dataset_256,
            stage2.patch_size,
            output_path,
            num_pairs=stage2.weight_map_pairs,
            batch_size=stage2.weight_map_batch_size,
            seed=stage2.seed,
            form=stage2.weight_map_form,
            use_power=stage2.weight_map_power,
            clip_quantile=stage2.weight_map_clip_quantile,
            num_workers=num_workers,
            force=force_maps,
        )

    model = _load_initial_model(
        pretrained,
        resolved_device,
        allow_partial=allow_partial_pretrained,
    )
    parameters = count_parameters(model)
    if parameters != 197819:
        raise RuntimeError(f"unexpected 3-RCAB parameter count: {parameters}")
    ema = EMAModel(model, decay=ema_decay)
    mse_fn = BrightnessAwareLoss().to(resolved_device)
    perceptual_fn = VGGPerceptualLoss(weights_path=vgg_weights).to(resolved_device)
    fdpl_fn = (
        PerChannelLogFDPLoss(
            map_64,
            map_256,
            target_ratio=stage1.target_ratio,
            rest_perceptual_weight=stage1.perceptual_weight,
        ).to(resolved_device)
        if enable_fdpl
        else None
    )
    train_loader_64, validation_loader_64 = build_sidd_loaders(
        h5_path,
        patch_size=stage1.patch_size,
        batch_size=stage1.batch_size,
        seed=stage1.seed,
        num_workers=num_workers,
        limit=smoke,
        data_chain=stage1.data_chain,
    )
    train_loader_256, validation_loader_256 = build_sidd_loaders(
        h5_path,
        patch_size=stage2.patch_size,
        batch_size=stage2.batch_size,
        seed=stage2.seed,
        num_workers=num_workers,
        limit=smoke,
        data_chain=stage2.data_chain,
    )

    history: list[dict[str, object]] = []
    stage1_result = train_stage(
        model,
        ema,
        train_loader_64,
        validation_loader_64,
        mse_fn=mse_fn,
        perceptual_fn=perceptual_fn,
        fdpl_fn=fdpl_fn,
        device=resolved_device,
        epochs=stage1.epochs,
        learning_rate=stage1.learning_rate,
        warmup_epochs=stage1.warmup_epochs,
        fdpl_weight=stage1.fdpl_weight,
        perceptual_weight=stage1.perceptual_weight,
        stage_name="Stage 1: 64x64",
        stage_id=1,
        output_path=output_path / "stage1_final.pth",
        best_path=output_path / "stage1_best.pth",
        weight_decay=stage1.weight_decay,
        calibration_batches=stage1.calibration_batches,
        recalibrate_every=stage1.recalibrate_every,
        history=history,
    )
    stage2_result = train_stage(
        model,
        ema,
        train_loader_256,
        validation_loader_256,
        mse_fn=mse_fn,
        perceptual_fn=perceptual_fn,
        fdpl_fn=fdpl_fn,
        device=resolved_device,
        epochs=stage2.epochs,
        learning_rate=stage2.learning_rate,
        warmup_epochs=stage2.warmup_epochs,
        fdpl_weight=stage2.fdpl_weight,
        perceptual_weight=stage2.perceptual_weight,
        stage_name="Stage 2: 256x256",
        stage_id=2,
        output_path=output_path / "stage2_final.pth",
        best_path=output_path / "stage2_best.pth",
        weight_decay=stage2.weight_decay,
        calibration_batches=stage2.calibration_batches,
        recalibrate_every=stage2.recalibrate_every,
        history=history,
    )
    ema.copy_to(model)
    ema_psnr, ema_ssim = validate_model(model, validation_loader_256, resolved_device)
    torch.save(ema.state_dict(), output_path / "ema.pth")

    result: dict[str, object] = {
        "parameters": parameters,
        "fdpl": enable_fdpl,
        "data_chain": stage1.data_chain,
        "deterministic": deterministic,
        "stage1": asdict(stage1_result),
        "stage2": asdict(stage2_result),
        "ema": {"psnr": ema_psnr, "ssim": ema_ssim},
        "scale_factor": None if fdpl_fn is None else fdpl_fn.scale_factor,
        "stage1_config": asdict(stage1),
        "stage2_config": asdict(stage2),
    }
    (output_path / "loss_history.json").write_text(
        json.dumps(history, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_path / "results.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    return result
