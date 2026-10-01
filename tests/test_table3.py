"""Tests for the Table 3 paired-statistics helper."""

from __future__ import annotations

import pytest

from cvfdpl.table3 import (
    TABLE3_ROWS,
    _student_t_two_sided_p,
    paired_statistics,
)


def test_table3_rows_cover_six_reported_comparisons() -> None:
    assert [(row.protocol, row.seed) for row in TABLE3_ROWS] == [
        ("sliding", 42),
        ("sliding", 43),
        ("sliding", 44),
        ("direct", 42),
        ("direct", 43),
        ("direct", 44),
    ]


def test_paired_statistics_computes_gain_count_and_student_t_p_value() -> None:
    stats = paired_statistics([1.0, 2.0, 4.0], [2.0, 3.0, 5.0])

    assert stats["images"] == 3
    assert stats["mean_delta"] == pytest.approx(1.0)
    assert stats["std_delta"] == pytest.approx(0.0)
    assert stats["fdpl_better"] == 3
    assert stats["t_statistic"] is None
    assert stats["p_two_sided"] == 0.0


def test_paired_statistics_uses_student_t_tail_probability() -> None:
    stats = paired_statistics(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        [1.1, 1.8, 3.3, 4.4, 4.9, 6.2, 7.05, 7.85],
    )

    assert stats["t_statistic"] == pytest.approx(0.9770084209183946)
    assert stats["p_two_sided"] == pytest.approx(0.3611134216129299)


@pytest.mark.parametrize(
    ("t_statistic", "expected"),
    [
        (23.6884, 3.0090366173277202e-99),
        (30.7336, 1.8351393781875648e-147),
        (27.6374, 4.9226446166662685e-126),
        (22.1755, 2.7982782807756625e-89),
        (28.1165, 2.4643591406629983e-129),
    ],
)
def test_student_t_matches_reference_tail_probabilities(
    t_statistic: float,
    expected: float,
) -> None:
    assert _student_t_two_sided_p(t_statistic, 1023) == pytest.approx(
        expected,
        rel=5e-12,
    )


def test_paired_statistics_rejects_misaligned_lists() -> None:
    with pytest.raises(ValueError, match="equal length"):
        paired_statistics([1.0, 2.0], [1.0])
