#!/usr/bin/env python3
"""Select and validate the RGB-to-luma coefficients used by an experiment."""

from __future__ import annotations

import os

import numpy as np


DEFAULT_LUMA_WEIGHTS = "0.299,0.587,0.114"


def luma_weights() -> np.ndarray:
    raw = os.environ.get("FREQDEC_LUMA_WEIGHTS", DEFAULT_LUMA_WEIGHTS)
    try:
        weights = np.asarray(
            [float(value) for value in raw.split(",")],
            dtype=np.float64,
        )
    except ValueError as error:
        raise ValueError(
            "FREQDEC_LUMA_WEIGHTS must contain three comma-separated numbers"
        ) from error
    if weights.shape != (3,):
        raise ValueError("FREQDEC_LUMA_WEIGHTS must contain exactly three values")
    if np.any(weights < 0.0) or not np.isfinite(weights).all():
        raise ValueError("luma weights must be finite and non-negative")
    if not np.isclose(weights.sum(), 1.0, atol=1e-9):
        raise ValueError("luma weights must sum to one")
    return weights
