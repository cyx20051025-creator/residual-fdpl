"""Small interface adapter for the frozen NAFNet external backbone."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn

from .nafnet_arch import NAFNet
from .operators import GaussianLowPass


def _unwrap_state_dict(state: object) -> OrderedDict[str, torch.Tensor]:
    if not isinstance(state, dict):
        raise TypeError(f"expected checkpoint mapping, got {type(state)}")
    for key in ("state_dict", "params", "model_state_dict"):
        nested = state.get(key)
        if isinstance(nested, dict):
            state = nested
            break
    return OrderedDict(
        (str(key), value)
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
    )


class FrozenNAFNet(nn.Module):
    """NAFNet with a 0-1 input/output interface.

    The selected endpoints were trained by the historical wrapper with 0-1
    image tensors. An empirical forward comparison showed that approximately
    [-1,1] inputs saturate this checkpoint, so this adapter deliberately uses
    the validated 0-1 convention.
    """

    def __init__(self) -> None:
        super().__init__()
        self.network = NAFNet(
            width=32,
            middle_blk_num=12,
            enc_blk_nums=(2, 2, 4, 8),
            dec_blk_nums=(2, 2, 2, 2),
        )
        self.low_pass = GaussianLowPass(sigma=1.0, radius=2)

    def forward(self, noisy_01: torch.Tensor) -> torch.Tensor:
        return self.network(noisy_01).clamp(0.0, 1.0)


def load_frozen_nafnet(
    checkpoint: str | Path,
    device: torch.device,
) -> FrozenNAFNet:
    model = FrozenNAFNet().to(device)
    state = _unwrap_state_dict(
        torch.load(
            Path(checkpoint),
            map_location="cpu",
            weights_only=True,
        )
    )
    missing, unexpected = model.network.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"NAFNet checkpoint mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
