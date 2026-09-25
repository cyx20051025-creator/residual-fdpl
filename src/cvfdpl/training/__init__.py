"""Training utilities for the canonical Residual FDPL protocol."""

from cvfdpl.training.checkpoint import EMAModel, load_state_dict
from cvfdpl.training.config import Stage0Config, StageConfig
from cvfdpl.training.pipeline import run_sidd_training, run_stage0_training
from cvfdpl.training.trainer import StageResult, build_sidd_loaders, train_stage
from cvfdpl.training.weight_map import compute_weight_map, get_or_compute_weight_map

__all__ = [
    "EMAModel",
    "Stage0Config",
    "StageConfig",
    "StageResult",
    "build_sidd_loaders",
    "compute_weight_map",
    "get_or_compute_weight_map",
    "load_state_dict",
    "run_sidd_training",
    "run_stage0_training",
    "train_stage",
]
