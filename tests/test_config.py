import json
from pathlib import Path

import pytest

from cvfdpl.training.config import Stage0Config, StageConfig


def test_public_configs_are_strict_and_round_trip() -> None:
    root = Path(__file__).resolve().parents[1]
    stage0 = Stage0Config.from_mapping(
        json.loads((root / "configs" / "stage0_gaussian.json").read_text())
    )
    stage1 = StageConfig.from_mapping(
        json.loads((root / "configs" / "sidd_stage1_64.json").read_text())
    )
    stage2 = StageConfig.from_mapping(
        json.loads((root / "configs" / "sidd_stage2_256.json").read_text())
    )
    assert stage0.patch_size == 64
    assert stage1.data_chain == "bilinear256"
    assert stage2.log_scaling == 1.37
    assert stage2.fdpl_weight == pytest.approx(0.08 / 1.37)


def test_unknown_config_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown Stage configuration keys"):
        StageConfig.from_mapping(
            {
                "patch_size": 64,
                "batch_size": 8,
                "epochs": 1,
                "optimizer": {"learning_rate": 1e-5},
                "loss": {"warmup_epochs": 1, "max_fdpl_weight": 0.08},
                "calibration": {},
                "weight_map": {},
                "ignored_typo": 1,
            }
        )


def test_inconsistent_log_scaling_is_rejected() -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        StageConfig.from_mapping(
            {
                "patch_size": 256,
                "batch_size": 4,
                "epochs": 1,
                "optimizer": {"name": "AdamW", "learning_rate": 5e-6},
                "loss": {
                    "name": "ResidualFDPL",
                    "warmup_epochs": 1,
                    "max_fdpl_weight": 0.08,
                    "log_scaling": 1.37,
                },
                "calibration": {},
                "weight_map": {},
            }
        )
