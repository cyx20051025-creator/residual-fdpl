import pytest

from rgb_correction.validation import validate_scene_split


def test_scene_split_accepts_disjoint_complete_partition() -> None:
    calibration, evaluation = validate_scene_split(
        (1, 2, 3),
        (4, 5),
        expected_total=5,
    )
    assert calibration == {1, 2, 3}
    assert evaluation == {4, 5}


def test_scene_split_rejects_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        validate_scene_split((1, 2), (2, 3))
