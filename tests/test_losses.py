import pytest
import torch

from cvfdpl import OriginalFDPLoss, PerChannelLogFDPLoss


def test_original_fdpl_is_zero_for_identical_inputs() -> None:
    image = torch.rand(1, 3, 32, 32)
    loss = OriginalFDPLoss()(image, image)
    assert loss.item() == pytest.approx(0.0, abs=1e-8)


def test_residual_fdpl_is_zero_for_identical_predictions() -> None:
    noisy = torch.rand(1, 3, 64, 64)
    prediction = noisy - 0.1
    target = noisy - 0.1
    loss = PerChannelLogFDPLoss(
        torch.ones(3, 64, 64),
        torch.ones(3, 256, 256),
    )
    assert loss(prediction, target, noisy).item() == pytest.approx(0.0, abs=1e-8)


def test_calibration_targets_requested_ratio() -> None:
    loss = PerChannelLogFDPLoss(
        torch.ones(3, 64, 64),
        torch.ones(3, 256, 256),
        target_ratio=0.30,
        rest_perceptual_weight=0.01,
    )
    scale = loss.calibrate_weight(
        mse_value=0.10,
        perceptual_value=0.20,
        fdpl_value=0.05,
        fdpl_lambda=0.08,
    )
    rest = 0.10 + 0.01 * 0.20
    expected = (0.30 / 0.70) * (rest / (0.08 * 0.05))
    assert scale == pytest.approx(expected)
    assert loss.calibrated

