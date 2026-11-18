"""Ridge calibration and paired-bootstrap utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from rgb_correction.ops import project_zero_luma


@dataclass(frozen=True)
class RidgeModel:
    """A standardized multi-output Ridge model with one intercept."""

    feature_mean: NDArray[np.float64]
    feature_scale: NDArray[np.float64]
    coefficients: NDArray[np.float64]
    intercept: NDArray[np.float64]

    def predict(self, features: ArrayLike) -> NDArray[np.float64]:
        return predict_ridge(self, features)


def scene_balanced_weights(scenes: ArrayLike) -> NDArray[np.float64]:
    """Give every scene equal total weight, regardless of crop count."""
    labels = np.asarray(scenes)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("scenes must be a non-empty one-dimensional array")
    unique, inverse, counts = np.unique(
        labels,
        return_inverse=True,
        return_counts=True,
    )
    del unique
    weights = 1.0 / counts[inverse].astype(np.float64)
    return weights * (len(weights) / weights.sum())


def fit_zero_luma_ridge(
    features: ArrayLike,
    targets: ArrayLike,
    *,
    scenes: ArrayLike | None = None,
    ridge: float = 1e-3,
    project_targets: bool = True,
) -> RidgeModel:
    """Fit a standardized Ridge map for one RGB offset per input.

    ``ridge`` is relative to the number of fitting rows after weights are
    normalized to mean one. Setting ``scenes`` enables equal total weight per
    scene.
    """
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or y.shape[0] != x.shape[0] or y.shape[1] != 3:
        raise ValueError("features must be (N, D) and targets must be (N, 3)")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("features and targets must be finite")
    if ridge < 0.0:
        raise ValueError("ridge must be non-negative")
    if project_targets:
        y = project_zero_luma(y)

    weights = (
        np.ones(len(x), dtype=np.float64)
        if scenes is None
        else scene_balanced_weights(scenes)
    )
    if len(weights) != len(x):
        raise ValueError("scenes and features must have the same length")

    feature_mean = np.average(x, axis=0, weights=weights)
    variance = np.average((x - feature_mean) ** 2, axis=0, weights=weights)
    feature_scale = np.sqrt(variance)
    feature_scale[feature_scale < 1e-12] = 1.0
    standardized = (x - feature_mean) / feature_scale

    intercept = np.average(y, axis=0, weights=weights)
    centered_targets = y - intercept
    root_weights = np.sqrt(weights)
    weighted_features = standardized * root_weights[:, None]
    weighted_targets = centered_targets * root_weights[:, None]
    gram = weighted_features.T @ weighted_features
    penalty = ridge * len(x) * np.eye(gram.shape[0])
    coefficients = np.linalg.solve(
        gram + penalty,
        weighted_features.T @ weighted_targets,
    )
    intercept = project_zero_luma(intercept)
    return RidgeModel(
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        coefficients=coefficients,
        intercept=intercept,
    )


def predict_ridge(model: RidgeModel, features: ArrayLike) -> NDArray[np.float64]:
    """Predict zero-luma RGB offsets for a feature matrix."""
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != model.feature_mean.shape[0]:
        raise ValueError("feature dimension does not match the fitted model")
    standardized = (x - model.feature_mean) / model.feature_scale
    return project_zero_luma(
        standardized @ model.coefficients + model.intercept
    )


def bootstrap_mean_interval(
    values: ArrayLike,
    *,
    seed: int,
    repeats: int = 10_000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return a deterministic percentile interval for the sample mean."""
    sample = np.asarray(values, dtype=np.float64)
    if sample.ndim != 1 or sample.size == 0:
        raise ValueError("values must be a non-empty one-dimensional array")
    if not np.isfinite(sample).all():
        raise ValueError("values must be finite")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(sample), size=(repeats, len(sample)))
    draws = sample[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(draws, (tail, 1.0 - tail))
    return float(low), float(high)
