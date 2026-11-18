"""Exact RGB-offset operations used by the paper."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

REC601_WEIGHTS = np.asarray((0.299, 0.587, 0.114), dtype=np.float64)


def validate_luma_weights(weights: ArrayLike = REC601_WEIGHTS) -> NDArray[np.float64]:
    """Return three finite, non-negative luma weights that sum to one."""
    values = np.asarray(weights, dtype=np.float64)
    if values.shape != (3,):
        raise ValueError("luma weights must have shape (3,)")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("luma weights must be finite and non-negative")
    if not np.isclose(values.sum(), 1.0, atol=1e-9):
        raise ValueError("luma weights must sum to one")
    return values


def _rgb_array(value: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (0,) or array.shape[-1:] != (3,):
        raise ValueError(f"{name} must have a trailing RGB dimension of size 3")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _finish(value: NDArray[np.float64]) -> float | NDArray[np.float64]:
    return float(value) if value.ndim == 0 else value


def project_zero_luma(
    offset: ArrayLike,
    *,
    weights: ArrayLike = REC601_WEIGHTS,
) -> NDArray[np.float64]:
    """Project RGB offsets onto the exact Rec.601 zero-luma plane."""
    values = _rgb_array(offset, "offset")
    luma = validate_luma_weights(weights)
    luma_component = np.einsum("...c,c->...", values, luma)
    return values - luma_component[..., None] * luma / np.dot(luma, luma)


def mse_reduction(
    mean_error: ArrayLike,
    offset: ArrayLike,
) -> float | NDArray[np.float64]:
    """Return the exact pre-clipping channel-averaged MSE reduction.

    For ``e = reference - output`` and a constant RGB offset ``d``, the
    reduction is ``(2 * mean(e)^T d - ||d||^2) / 3``. Positive output means
    that MSE decreases. This identity assumes no clipping.
    """
    error = _rgb_array(mean_error, "mean_error")
    delta = _rgb_array(offset, "offset")
    error, delta = np.broadcast_arrays(error, delta)
    value = (
        2.0 * np.einsum("...c,...c->...", error, delta)
        - np.einsum("...c,...c->...", delta, delta)
    ) / 3.0
    return _finish(value)


def preclip_psnr_delta(
    base_mse: ArrayLike,
    mean_error: ArrayLike,
    offset: ArrayLike,
) -> float | NDArray[np.float64]:
    """Return the PSNR change implied by :func:`mse_reduction`."""
    base = np.asarray(base_mse, dtype=np.float64)
    reduction = np.asarray(mse_reduction(mean_error, offset), dtype=np.float64)
    base, reduction = np.broadcast_arrays(base, reduction)
    corrected = base - reduction
    if np.any(base <= 0.0) or np.any(corrected <= 0.0):
        raise ValueError("base and corrected MSE values must be positive")
    return _finish(10.0 * np.log10(base / corrected))


def apply_rgb_offset(
    output: ArrayLike,
    offset: ArrayLike,
    *,
    weights: ArrayLike = REC601_WEIGHTS,
    project: bool = True,
    clip: bool = True,
    value_range: tuple[float, float] = (0.0, 1.0),
) -> NDArray[np.float64]:
    """Apply one RGB correction to one or more RGB outputs.

    ``output`` and ``offset`` use sRGB coordinates. If ``project`` is true,
    the offset is projected onto the Rec.601 zero-luma plane before addition.
    Clipping is applied once after addition.
    """
    base = _rgb_array(output, "output")
    delta = _rgb_array(offset, "offset")
    if project:
        delta = project_zero_luma(delta, weights=weights)
    corrected = base + delta
    if clip:
        low, high = value_range
        if low >= high:
            raise ValueError("value_range must be increasing")
        corrected = np.clip(corrected, low, high)
    return corrected
