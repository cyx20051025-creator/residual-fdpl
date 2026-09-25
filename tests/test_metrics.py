import pytest
import torch

from cvfdpl import calculate_psnr, calculate_ssim


def test_identical_images_have_perfect_metrics() -> None:
    image = torch.rand(1, 3, 32, 32)
    assert calculate_psnr(image, image) == float("inf")
    assert calculate_ssim(image, image) == pytest.approx(1.0, abs=1e-6)


def test_metrics_drop_after_perturbation() -> None:
    image = torch.rand(1, 3, 32, 32)
    perturbed = (image + 0.05).clamp(0, 1)
    assert calculate_psnr(perturbed, image) < float("inf")
    assert calculate_ssim(perturbed, image) < 1.0
