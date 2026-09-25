"""Data-free smoke test."""

from __future__ import annotations

import torch

from cvfdpl.losses import PerChannelLogFDPLoss
from cvfdpl.models import RRDBPlusLightRCAB, count_parameters
from cvfdpl.utils import get_device, set_seed


def main() -> None:
    set_seed(42)
    device = get_device("cpu")
    model = RRDBPlusLightRCAB(num_rrdb=4, num_rcab=3).to(device)
    parameters = count_parameters(model)
    if parameters != 197819:
        raise RuntimeError(f"Expected 197819 parameters, found {parameters}")

    noisy = torch.rand(1, 3, 64, 64, device=device)
    target = noisy + 0.01 * torch.randn_like(noisy)
    prediction = model(noisy)
    if prediction.shape != noisy.shape or not torch.isfinite(prediction).all():
        raise RuntimeError("Model output is invalid")

    weight_map_64 = torch.ones(3, 64, 64)
    weight_map_256 = torch.ones(3, 256, 256)
    fdpl = PerChannelLogFDPLoss(weight_map_64, weight_map_256).to(device)
    loss = fdpl(prediction, target, noisy)
    loss.backward()
    if not torch.isfinite(loss):
        raise RuntimeError("Residual FDPL loss is not finite")

    print(f"device={device}")
    print(f"parameters={parameters}")
    print(f"output_shape={tuple(prediction.shape)}")
    print(f"loss={loss.item():.8f}")
    print("smoke_test=passed")

