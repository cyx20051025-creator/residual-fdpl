"""Tests for the Table 3 paired-statistics helper."""

from __future__ import annotations

import pytest

from cvfdpl.table3 import TABLE3_ROWS, paired_statistics


def test_table3_rows_cover_five_reported_comparisons() -> None:
    assert [(row.protocol, row.seed) for row in TABLE3_ROWS] == [
        ("sliding", 42),
        ("sliding", 43),
        ("sliding", 44),
        ("direct", 42),
        ("direct", 43),
    ]


def test_paired_statistics_computes_gain_count_and_normal_p_value() -> None:
    stats = paired_statistics([1.0, 2.0, 4.0], [2.0, 3.0, 5.0])

    assert stats["images"] == 3
    assert stats["mean_delta"] == pytest.approx(1.0)
    assert stats["std_delta"] == pytest.approx(0.0)
    assert stats["fdpl_better"] == 3
    assert stats["t_statistic"] is None
    assert stats["p_two_sided"] == 0.0


def test_paired_statistics_rejects_misaligned_lists() -> None:
    with pytest.raises(ValueError, match="equal length"):
        paired_statistics([1.0, 2.0], [1.0])
