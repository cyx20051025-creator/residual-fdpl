"""Validation helpers for scene-held-out calibration."""

from __future__ import annotations

from collections.abc import Iterable, Sized
from typing import TypeVar

Element = TypeVar("Element")


def _as_set(values: Iterable[Element] | Sized, name: str) -> set[Element]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a collection of units, not a string")
    result = set(values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def validate_scene_split(
    calibration_units: Iterable[Element] | Sized,
    evaluation_units: Iterable[Element] | Sized,
    *,
    expected_total: int | None = None,
) -> tuple[set[Element], set[Element]]:
    """Reject overlapping or malformed scene-level calibration splits."""
    calibration = _as_set(calibration_units, "calibration_units")
    evaluation = _as_set(evaluation_units, "evaluation_units")
    overlap = calibration & evaluation
    if overlap:
        raise ValueError(f"calibration/evaluation overlap: {sorted(overlap)!r}")
    if expected_total is not None:
        observed = len(calibration) + len(evaluation)
        if observed != expected_total:
            raise ValueError(
                f"expected {expected_total} total units, observed {observed}"
            )
    return calibration, evaluation
