"""Frozen-denoiser RGB correction utilities.

This package contains the small, dependency-light portion of the
risk-controlled RGB correction study: zero-luma color geometry, pre-clipping
MSE identities, scene-balanced Ridge calibration, and action selection.
Research replay scripts live under ``scripts/rgb_correction``.
"""

from rgb_correction.calibration import (
    RidgeModel,
    bootstrap_mean_interval,
    fit_zero_luma_ridge,
    predict_ridge,
    scene_balanced_weights,
)
from rgb_correction.ops import (
    REC601_WEIGHTS,
    apply_rgb_offset,
    mse_reduction,
    preclip_psnr_delta,
    project_zero_luma,
    validate_luma_weights,
)
from rgb_correction.selection import (
    CandidateScore,
    RiskPolicy,
    SelectionResult,
    select_action,
)
from rgb_correction.validation import validate_scene_split

__all__ = [
    "CandidateScore",
    "REC601_WEIGHTS",
    "RidgeModel",
    "RiskPolicy",
    "SelectionResult",
    "apply_rgb_offset",
    "bootstrap_mean_interval",
    "fit_zero_luma_ridge",
    "mse_reduction",
    "predict_ridge",
    "preclip_psnr_delta",
    "project_zero_luma",
    "scene_balanced_weights",
    "select_action",
    "validate_luma_weights",
    "validate_scene_split",
]
