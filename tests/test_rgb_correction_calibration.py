import numpy as np
import pytest

from rgb_correction.calibration import (
    bootstrap_mean_interval,
    fit_zero_luma_ridge,
    predict_ridge,
    scene_balanced_weights,
)
from rgb_correction.ops import REC601_WEIGHTS


def test_scene_balanced_weights_equalize_scene_totals() -> None:
    scenes = np.asarray(("a", "a", "a", "b"))
    weights = scene_balanced_weights(scenes)
    assert np.isclose(weights[:3].sum(), weights[3])
    assert np.isclose(weights.mean(), 1.0)


def test_ridge_recovers_a_zero_luma_linear_map() -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(200, 6))
    coefficients = np.asarray(
        (
            (1.0, 0.0, 0.0, 0.5, -0.5, 0.0),
            (-0.2, 0.7, 0.0, 0.0, 0.3, 0.1),
            (-0.1, -0.2, 0.6, -0.3, 0.1, 0.4),
        )
    ).T
    raw_targets = features @ coefficients
    targets = raw_targets - (
        raw_targets @ REC601_WEIGHTS
    )[:, None] * REC601_WEIGHTS / np.dot(REC601_WEIGHTS, REC601_WEIGHTS)
    model = fit_zero_luma_ridge(features, targets, ridge=1e-8)
    prediction = predict_ridge(model, features)
    assert np.max(np.abs(prediction - targets)) < 1e-7


def test_ridge_rejects_scene_length_mismatch() -> None:
    with pytest.raises(ValueError):
        fit_zero_luma_ridge(
            np.zeros((3, 2)),
            np.zeros((3, 3)),
            scenes=("a", "b"),
        )


def test_bootstrap_interval_is_deterministic() -> None:
    values = np.arange(10, dtype=np.float64)
    first = bootstrap_mean_interval(values, seed=19, repeats=500)
    second = bootstrap_mean_interval(values, seed=19, repeats=500)
    assert first == second
    assert first[0] <= values.mean() <= first[1]
