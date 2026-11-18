import numpy as np
import pytest

from rgb_correction.ops import (
    REC601_WEIGHTS,
    apply_rgb_offset,
    mse_reduction,
    preclip_psnr_delta,
    project_zero_luma,
    validate_luma_weights,
)


def test_zero_luma_projection_is_exact_and_idempotent() -> None:
    offset = np.asarray((0.03, -0.01, -0.02))
    projected = project_zero_luma(offset)
    assert np.isclose(np.dot(REC601_WEIGHTS, projected), 0.0)
    assert np.allclose(project_zero_luma(projected), projected)


def test_mse_identity_matches_direct_computation() -> None:
    output = np.asarray((0.10, 0.20, 0.30))
    reference = np.asarray((0.16, 0.17, 0.31))
    offset = project_zero_luma(np.asarray((0.02, 0.03, -0.01)))
    before = np.mean((reference - output) ** 2)
    after = np.mean((reference - (output + offset)) ** 2)
    assert np.isclose(mse_reduction(reference - output, offset), before - after)
    assert np.isclose(
        preclip_psnr_delta(before, reference - output, offset),
        10.0 * np.log10(before / after),
    )


def test_apply_rgb_offset_projects_and_clips_once() -> None:
    output = np.asarray((0.99, 0.01, 0.50))
    corrected = apply_rgb_offset(output, np.asarray((0.10, -0.10, 0.05)))
    assert corrected.shape == (3,)
    assert np.all((corrected >= 0.0) & (corrected <= 1.0))
    projected = project_zero_luma(np.asarray((0.10, -0.10, 0.05)))
    assert np.allclose(corrected, np.clip(output + projected, 0.0, 1.0))


def test_invalid_luma_weights_are_rejected() -> None:
    with pytest.raises(ValueError):
        validate_luma_weights((0.3, 0.3, 0.3))
