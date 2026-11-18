"""A/C/E models for the fixed-band residual-head screen."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import math

import torch
from torch import nn

from .operators import GaussianLowPass, channel_mean, remove_channel_mean


class RRDB(nn.Module):
    def __init__(self, channels: int = 32, growth: int = 16) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(channels, growth, 3, padding=1)
        self.c2 = nn.Conv2d(channels + growth, growth, 3, padding=1)
        self.c3 = nn.Conv2d(channels + 2 * growth, channels, 3, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        first = self.act(self.c1(value))
        second = self.act(self.c2(torch.cat((value, first), dim=1)))
        third = self.c3(torch.cat((value, first, second), dim=1))
        return value + 0.25 * third


class RCAB(nn.Module):
    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        features = self.feat(value)
        return value + features * self.attn(features)


class FixedBandResidualNet(nn.Module):
    """The existing 3-RCAB trunk with one of five output constructions."""

    ARMS = ("A", "C", "E", "DC", "S")

    def __init__(self, arm: str, channels: int = 32) -> None:
        super().__init__()
        if arm not in self.ARMS:
            raise ValueError(f"unknown arm: {arm}")
        self.arm = arm
        self.conv_in = nn.Conv2d(3, channels, 3, padding=1)
        self.body = nn.Sequential(*(RRDB(channels) for _ in range(4)))
        self.branch_high = nn.Sequential(*(RCAB(channels) for _ in range(3)))
        self.conv_fusion = nn.Conv2d(2 * channels, channels, 3, padding=1)

        if arm == "A":
            output_layers = 1
        elif arm == "DC":
            output_layers = 2
        else:
            output_layers = 3
        for index in range(output_layers):
            setattr(self, f"head_{index}", nn.Conv2d(channels, 3, 3, padding=1))
        self.low_pass = GaussianLowPass(sigma=1.0, radius=2)

    def forward(
        self,
        noisy: torch.Tensor,
        *,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        features = self.conv_in(noisy)
        fused = self.conv_fusion(
            torch.cat((self.body(features), self.branch_high(features)), dim=1)
        )
        if self.arm == "A":
            residual = self.head_0(fused)
            components = {"residual": residual}
        elif self.arm == "DC":
            spatial = self.head_0(fused)
            dc_source = self.head_1(fused)
            term_non_dc = remove_channel_mean(spatial)
            term_dc = channel_mean(dc_source)
            residual = term_non_dc + term_dc
            components = {
                "head_spatial": spatial,
                "head_dc": dc_source,
                "term_non_dc": term_non_dc,
                "term_dc": term_dc,
                "residual": residual,
            }
        elif self.arm in {"C", "E"}:
            head_dc = self.head_0(fused)
            head_lf = self.head_1(fused)
            head_hf = self.head_2(fused)
            if self.arm == "C":
                residual, terms = combine_fixed_band_heads(
                    head_dc, head_lf, head_hf, self.low_pass
                )
                term_dc = terms["term_dc"]
                term_lf = terms["term_lf"]
                term_hf = terms["term_hf"]
            else:
                residual = (head_dc + head_lf + head_hf) / 3.0
                term_dc = channel_mean(head_dc)
                term_lf = remove_channel_mean(self.low_pass(head_lf))
                term_hf = remove_channel_mean(head_hf - self.low_pass(head_hf))
            components = {
                "head_dc": head_dc,
                "head_lf": head_lf,
                "head_hf": head_hf,
                "term_dc": term_dc,
                "term_lf": term_lf,
                "term_hf": term_hf,
                "residual": residual,
            }
        else:
            head_0 = self.head_0(fused)
            head_1 = self.head_1(fused)
            head_2 = self.head_2(fused)
            masks = spatial_mix_masks(
                noisy.shape[-2],
                noisy.shape[-1],
                noisy.device,
                noisy.dtype,
            )
            residual = (
                masks[0] * head_0
                + masks[1] * head_1
                + masks[2] * head_2
            )
            components = {
                "head_0": head_0,
                "head_1": head_1,
                "head_2": head_2,
                "mask_0": masks[0],
                "mask_1": masks[1],
                "mask_2": masks[2],
                "residual": residual,
            }
        prediction = noisy + residual
        if return_components:
            return prediction, components
        return prediction


def combine_fixed_band_heads(
    head_dc: torch.Tensor,
    head_lf: torch.Tensor,
    head_hf: torch.Tensor,
    low_pass: nn.Module,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply the corrected, DC-isolated fixed-band projection."""
    term_dc = channel_mean(head_dc)
    term_lf = remove_channel_mean(low_pass(head_lf))
    term_hf = remove_channel_mean(head_hf - low_pass(head_hf))
    residual = term_dc + term_lf + term_hf
    return residual, {
        "term_dc": term_dc,
        "term_lf": term_lf,
        "term_hf": term_hf,
    }


def spatial_mix_masks(
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return fixed positive spatial blends whose sum is exactly one.

    At matched initialization all heads are identical, so this arm starts from
    the same output as A. During training the spatially varying masks break head
    symmetry and cannot be folded into the original single 3x3 output layer.
    The masks impose no fixed frequency partition.
    """
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    raw = []
    for phase in (0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0):
        field = torch.sin(3.0 * math.pi * xx + phase) * torch.cos(
            2.0 * math.pi * yy - 0.5 * phase
        )
        raw.append((1.0 + 0.6 * field).unsqueeze(0).unsqueeze(0))
    total = raw[0] + raw[1] + raw[2]
    return tuple(value / total for value in raw)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _unwrap_state_dict(state: object) -> OrderedDict[str, torch.Tensor]:
    if not isinstance(state, dict):
        raise TypeError(f"expected checkpoint mapping, got {type(state)}")
    for key in ("model_state_dict", "params", "state_dict"):
        nested = state.get(key)
        if isinstance(nested, dict):
            state = nested
            break
    return OrderedDict(
        (str(key), value)
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
    )


def load_initial_state(
    model: FixedBandResidualNet,
    checkpoint: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, object]:
    """Load one common single-head checkpoint into A, C, or E."""
    source = _unwrap_state_dict(
        torch.load(Path(checkpoint), map_location=map_location, weights_only=True)
    )
    mapped: OrderedDict[str, torch.Tensor] = OrderedDict()
    output_key = "conv_out.weight" if "conv_out.weight" in source else "head_0.weight"
    bias_key = "conv_out.bias" if "conv_out.bias" in source else "head_0.bias"
    if output_key not in source or bias_key not in source:
        raise KeyError("checkpoint does not contain a single 3-channel output layer")

    for key, value in source.items():
        if key.startswith(("conv_out.", "head_0.", "head_1.", "head_2.")):
            continue
        mapped[key] = value
    if model.arm == "A":
        mapped["head_0.weight"] = source[output_key]
        mapped["head_0.bias"] = source[bias_key]
    elif model.arm == "DC":
        for index in range(2):
            mapped[f"head_{index}.weight"] = source[output_key].clone()
            mapped[f"head_{index}.bias"] = source[bias_key].clone()
    else:
        for index in range(3):
            mapped[f"head_{index}.weight"] = source[output_key].clone()
            mapped[f"head_{index}.bias"] = source[bias_key].clone()

    missing, unexpected = model.load_state_dict(mapped, strict=False)
    allowed_missing = {"low_pass.kernel"}
    if set(missing) - allowed_missing or unexpected:
        raise RuntimeError(
            f"checkpoint mismatch: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    return {
        "checkpoint": str(checkpoint),
        "output_source_key": output_key,
        "cloned_heads": 3 if model.arm in {"C", "E"} else 1,
    }


def load_exact_state(
    model: FixedBandResidualNet,
    checkpoint: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, object]:
    """Load an already-trained A/C/E endpoint without cloning output heads."""
    source = _unwrap_state_dict(
        torch.load(Path(checkpoint), map_location=map_location, weights_only=True)
    )
    missing, unexpected = model.load_state_dict(source, strict=False)
    allowed_missing = {"low_pass.kernel"}
    if set(missing) - allowed_missing or unexpected:
        raise RuntimeError(
            f"checkpoint mismatch: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    return {
        "checkpoint": str(checkpoint),
        "exact_load": True,
        "missing": sorted(missing),
    }
