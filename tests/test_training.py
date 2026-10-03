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


def test_weight_map_preserves_the_training_rng_stream(tmp_path: Path) -> None:
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
    expected = torch.rand(1).item()
    torch.manual_seed(123)
    compute_weight_map(
        HDF5PairDataset(path),
        num_pairs=4,
        batch_size=2,
        seed=42,
        form="log_inv",
        clip_quantile=0.1,
        device="cpu",
    )
    assert torch.rand(1).item() == pytest.approx(expected)


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
