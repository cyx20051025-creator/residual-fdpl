from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cvfdpl.data import HDF5PairDataset
from cvfdpl.training import EMAModel, compute_weight_map, pipeline, train_stage
from cvfdpl.training.config import StageConfig
from cvfdpl.training.trainer import StageResult


def test_weight_map_is_mean_normalized_and_shape_stable(tmp_path: Path) -> None:
    path = tmp_path / "pairs.h5"
    generator = np.random.default_rng(7)
    noisy = generator.integers(0, 256, size=(4, 8, 8, 3), dtype=np.uint8)
    clean = np.clip(
        noisy.astype(np.int16) + generator.integers(-10, 11, size=noisy.shape),
        0,
        255,
    ).astype(np.uint8)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("noisy", data=noisy)
        handle.create_dataset("gt", data=clean)

    dataset = HDF5PairDataset(path)
    weight_map = compute_weight_map(
        dataset,
        num_pairs=4,
        batch_size=2,
        shuffle=False,
        seed=42,
        form="log_inv",
        clip_quantile=0.1,
        device="cpu",
    )
    assert tuple(weight_map.shape) == (3, 8, 8)
    assert torch.isfinite(weight_map).all()
    assert weight_map.mean().item() == pytest.approx(1.0, abs=1e-6)


def test_weight_map_preserves_cpu_and_optional_mps_rng_streams(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pairs.h5"
    generator = np.random.default_rng(11)
    noisy = generator.integers(0, 256, size=(4, 8, 8, 3), dtype=np.uint8)
    clean = np.clip(
        noisy.astype(np.int16) + generator.integers(-5, 6, size=noisy.shape),
        0,
        255,
    ).astype(np.uint8)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("noisy", data=noisy)
        handle.create_dataset("gt", data=clean)

    torch.manual_seed(123)
    expected_cpu = torch.rand(1).item()
    expected_mps = None
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(123)
        expected_mps = torch.rand(1, device="mps").item()

    torch.manual_seed(123)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(123)
    compute_weight_map(
        HDF5PairDataset(path),
        num_pairs=4,
        batch_size=2,
        seed=42,
        form="log_inv",
        clip_quantile=0.1,
        device="cpu",
    )
    assert torch.rand(1).item() == pytest.approx(expected_cpu)
    if expected_mps is not None:
        assert torch.rand(1, device="mps").item() == pytest.approx(expected_mps)


def test_shared_weight_map_seed_is_independent_of_training_seed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    recorded_seeds: list[int] = []

    def fake_get_or_compute_weight_map(*args, **kwargs):
        recorded_seeds.append(kwargs["seed"])
        return torch.ones(3, 8, 8)

    monkeypatch.setattr(
        pipeline,
        "get_or_compute_weight_map",
        fake_get_or_compute_weight_map,
    )
    for seed in (42, 43, 44):
        stage = StageConfig(
            patch_size=64,
            batch_size=8,
            epochs=1,
            learning_rate=1e-5,
            warmup_epochs=1,
            fdpl_weight=0.08,
            seed=seed,
        )
        pipeline._get_weight_map_for_stage(
            object(),
            stage.patch_size,
            stage,
            tmp_path,
            num_workers=0,
            force=False,
        )

    assert recorded_seeds == [42, 42, 42]


def test_run_sidd_training_passes_stages_to_map_builder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "pairs.h5"
    generator = np.random.default_rng(13)
    noisy = generator.integers(0, 256, size=(4, 8, 8, 3), dtype=np.uint8)
    clean = np.clip(
        noisy.astype(np.int16) + generator.integers(-5, 6, size=noisy.shape),
        0,
        255,
    ).astype(np.uint8)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("noisy", data=noisy)
        handle.create_dataset("gt", data=clean)

    map_calls: list[tuple[int, StageConfig]] = []

    def fake_get_weight_map_for_stage(
        dataset,
        image_size,
        stage,
        output_dir,
        **kwargs,
    ):
        map_calls.append((image_size, stage))
        return torch.ones(3, image_size, image_size)

    class DummyPerceptualLoss(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()

    def fake_load_initial_model(*args, **kwargs):
        return nn.Linear(1, 1)

    def fake_train_stage(*args, **kwargs):
        return StageResult(final_psnr=30.0, final_ssim=0.9, best_psnr=30.0)

    def fake_validate_model(*args, **kwargs):
        return 30.0, 0.9

    monkeypatch.setattr(
        pipeline,
        "_get_weight_map_for_stage",
        fake_get_weight_map_for_stage,
    )
    monkeypatch.setattr(pipeline, "_load_initial_model", fake_load_initial_model)
    monkeypatch.setattr(pipeline, "VGGPerceptualLoss", DummyPerceptualLoss)
    monkeypatch.setattr(pipeline, "count_parameters", lambda model: 197819)
    monkeypatch.setattr(pipeline, "train_stage", fake_train_stage)
    monkeypatch.setattr(pipeline, "validate_model", fake_validate_model)

    stage1 = StageConfig(
        patch_size=64,
        batch_size=2,
        epochs=1,
        learning_rate=1e-5,
        warmup_epochs=1,
        fdpl_weight=0.08,
        seed=43,
    )
    stage2 = StageConfig(
        patch_size=256,
        batch_size=2,
        epochs=1,
        learning_rate=5e-6,
        warmup_epochs=1,
        fdpl_weight=0.08 / 1.37,
        log_scaling=1.37,
        seed=43,
    )
    output_dir = tmp_path / "run"
    pipeline.run_sidd_training(
        path,
        output_dir,
        stage1,
        stage2,
        device="cpu",
        smoke=4,
        enable_fdpl=True,
    )

    assert [(size, stage.seed) for size, stage in map_calls] == [
        (64, 43),
        (256, 43),
    ]


def test_one_stage_cpu_optimization_writes_checkpoints(tmp_path: Path) -> None:
    torch.manual_seed(3)
    model = nn.Conv2d(3, 3, kernel_size=1)
    ema = EMAModel(model, decay=0.9)
    noisy = torch.rand(2, 3, 8, 8)
    clean = noisy * 0.9
    dataset = TensorDataset(noisy, clean)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    output_path = tmp_path / "stage1_final.pth"
    best_path = tmp_path / "stage1_best.pth"
    history: list[dict[str, object]] = []

    result = train_stage(
        model,
        ema,
        loader,
        loader,
        mse_fn=nn.MSELoss(),
        perceptual_fn=nn.MSELoss(),
        fdpl_fn=None,
        device=torch.device("cpu"),
        epochs=1,
        learning_rate=1e-4,
        warmup_epochs=1,
        fdpl_weight=0.0,
        perceptual_weight=0.0,
        stage_name="test",
        stage_id=1,
        output_path=output_path,
        best_path=best_path,
        calibration_batches=1,
        history=history,
    )

    assert output_path.is_file()
    assert best_path.is_file()
    assert len(history) == 1
    assert torch.isfinite(torch.tensor(result.final_psnr))
