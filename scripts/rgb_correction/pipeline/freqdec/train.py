"""Train one A/C/E screen arm from a common checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import HDF5Pairs, file_sha256, load_plan, make_canonical_plan, save_plan
from .model import FixedBandResidualNet, count_parameters, load_initial_state


class BrightnessAwareMSE(nn.Module):
    """The TVC brightness-weighted MSE on normalized targets."""

    def __init__(self, threshold: float = 0.6, weight: float = 1.5) -> None:
        super().__init__()
        self.threshold = threshold
        self.weight = weight

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        brightness = (
            0.299 * target[:, 0]
            + 0.587 * target[:, 1]
            + 0.114 * target[:, 2]
        )
        pixel_weight = torch.where(
            brightness > self.threshold,
            torch.full_like(brightness, self.weight),
            torch.ones_like(brightness),
        ).unsqueeze(1)
        return ((prediction - target).square() * pixel_weight).mean()


class VGGPerceptual(nn.Module):
    def __init__(
        self,
        weights_path: str | Path | None = None,
        *,
        allow_random: bool = False,
    ) -> None:
        super().__init__()
        from torchvision.models import VGG19_Weights, vgg19

        if weights_path is not None:
            backbone = vgg19(weights=None)
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            backbone.load_state_dict(state, strict=True)
        elif allow_random:
            torch.manual_seed(314159)
            backbone = vgg19(weights=None)
        else:
            backbone = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        self.features = backbone.features[:16].eval()
        for parameter in self.features.parameters():
            parameter.requires_grad_(False)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (self.features(prediction) - self.features(target)).square().mean()


def state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False


def learning_rate(step: int, steps: int, initial: float, final: float, warmup: int) -> float:
    if warmup > 0 and step < warmup:
        return initial * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, steps - warmup))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return final + (initial - final) * cosine


def atomic_torch_save(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["A", "C", "E"], required=True)
    parser.add_argument("--seed", type=int, default=57)
    parser.add_argument("--train-h5", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--initial-lr", type=float, default=1e-5)
    parser.add_argument("--final-lr", type=float, default=1e-6)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--perceptual-weight", type=float, default=0.01)
    parser.add_argument("--stream-seed", type=int, default=20261114)
    parser.add_argument("--vgg-weights", type=Path)
    parser.add_argument("--allow-random-vgg", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.batch_size <= 0:
        raise ValueError("steps and batch-size must be positive")
    if args.perceptual_weight < 0:
        raise ValueError("perceptual-weight cannot be negative")
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = HDF5Pairs(args.train_h5)
    if args.plan is None:
        plan_path = args.output_dir / "canonical_plan.npz"
        indices, flip_h, flip_v, rotations, digest = make_canonical_plan(
            len(dataset), args.steps, args.batch_size, args.stream_seed
        )
        save_plan(plan_path, indices, flip_h, flip_v, rotations)
        args.plan = plan_path
    else:
        indices, flip_h, flip_v, rotations, digest = load_plan(args.plan)
    expected_shape = (args.steps, args.batch_size)
    if indices.shape != expected_shape:
        raise ValueError(
            f"plan shape {indices.shape} does not match {expected_shape}"
        )

    model = FixedBandResidualNet(args.arm).to(device)
    init_info = load_initial_state(model, args.init_checkpoint)
    initial_hash = state_sha256(model)
    mse_loss = BrightnessAwareMSE().to(device)
    perceptual_loss = VGGPerceptual(
        args.vgg_weights, allow_random=args.allow_random_vgg
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.initial_lr,
        betas=(0.9, 0.9),
        weight_decay=1e-4,
    )
    start_step = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("plan_sha256") != digest:
            raise ValueError("resume plan hash differs from the current plan")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = int(checkpoint["step"])

    config = {
        "arm": args.arm,
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "initial_lr": args.initial_lr,
        "final_lr": args.final_lr,
        "warmup": args.warmup,
        "perceptual_weight": args.perceptual_weight,
        "stream_seed": args.stream_seed,
        "plan_sha256": digest,
        "train_h5": str(args.train_h5),
        "train_h5_sha256": file_sha256(args.train_h5),
        "initial_checkpoint": str(args.init_checkpoint),
        "initial_checkpoint_sha256": file_sha256(args.init_checkpoint),
        "initialization": init_info,
        "parameter_count": count_parameters(model),
        "initial_state_sha256": initial_hash,
        "random_vgg": args.allow_random_vgg,
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    history: list[dict[str, float | int]] = []
    running_loss = 0.0
    running_count = 0
    started = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    finished = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    if started is not None:
        started.record()

    for step in range(start_step, args.steps):
        noisy, clean = dataset.batch(
            indices[step], flip_h[step], flip_v[step], rotations[step]
        )
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        prediction = model(noisy)
        mse = mse_loss(prediction, clean)
        prediction_01 = (prediction * 0.5 + 0.5).clamp(0.0, 1.0)
        clean_01 = (clean * 0.5 + 0.5).clamp(0.0, 1.0)
        perceptual = perceptual_loss(prediction_01, clean_01)
        loss = mse + args.perceptual_weight * perceptual
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"non-finite loss at step {step + 1}")

        for group in optimizer.param_groups:
            group["lr"] = learning_rate(
                step,
                args.steps,
                args.initial_lr,
                args.final_lr,
                args.warmup,
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True
        )
        optimizer.step()

        running_loss += float(loss.detach())
        running_count += 1
        completed = step + 1
        if completed % args.log_every == 0 or completed == args.steps:
            event = {
                "step": completed,
                "loss": running_loss / running_count,
                "mse": float(mse.detach()),
                "perceptual": float(perceptual.detach()),
                "gradient_norm": float(gradient_norm),
                "lr": optimizer.param_groups[0]["lr"],
            }
            history.append(event)
            print(json.dumps(event), flush=True)
            running_loss = 0.0
            running_count = 0

        if args.checkpoint_every > 0 and completed % args.checkpoint_every == 0:
            atomic_torch_save(
                {
                    "step": completed,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "plan_sha256": digest,
                    "config": config,
                },
                args.output_dir / "checkpoint.pt",
            )

    if started is not None and finished is not None:
        finished.record()
        torch.cuda.synchronize()
        elapsed_seconds = started.elapsed_time(finished) / 1000.0
    else:
        elapsed_seconds = 0.0

    endpoint = args.output_dir / "endpoint.pt"
    atomic_torch_save(
        {
            "step": args.steps,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "plan_sha256": digest,
            "config": config,
            "elapsed_seconds": elapsed_seconds,
        },
        endpoint,
    )
    result = {
        **config,
        "elapsed_seconds": elapsed_seconds,
        "endpoint": str(endpoint),
        "endpoint_sha256": file_sha256(endpoint),
        "final_state_sha256": state_sha256(model),
        "history": history,
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"endpoint": str(endpoint), "elapsed_seconds": elapsed_seconds}))


if __name__ == "__main__":
    main()
