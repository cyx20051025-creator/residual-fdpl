"""Checkpoint and exponential-moving-average helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def load_state_dict(
    checkpoint: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Load a plain state dictionary or a common checkpoint wrapper."""

    payload: Any = torch.load(checkpoint, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("checkpoint payload must be a mapping")
    for key in ("model_state_dict", "state_dict"):
        candidate = payload.get(key)
        if isinstance(candidate, dict):
            return candidate
    return payload


class EMAModel:
    """Maintain a CPU shadow copy of model weights and buffers."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("decay must be in [0, 1)")
        self.decay = decay
        self.shadow = {
            key: value.detach().clone().float().cpu()
            for key, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Blend the current model state into the shadow state."""

        for key, value in model.state_dict().items():
            current = value.detach().float().cpu()
            self.shadow[key].mul_(self.decay).add_(current, alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """Copy the shadow state into a compatible model."""

        state = model.state_dict()
        missing = set(self.shadow) - set(state)
        if missing:
            raise KeyError(f"EMA state has unknown keys: {sorted(missing)}")
        for key, value in self.shadow.items():
            state[key].copy_(value.to(device=state[key].device, dtype=state[key].dtype))

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return a detached copy of the shadow state."""

        return {key: value.clone() for key, value in self.shadow.items()}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load a shadow state produced by :meth:`state_dict`."""

        if set(state_dict) != set(self.shadow):
            raise KeyError("EMA state keys do not match the model")
        self.shadow = {
            key: value.detach().clone().float().cpu()
            for key, value in state_dict.items()
        }
