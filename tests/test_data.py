from pathlib import Path

import h5py
import numpy as np

from cvfdpl.data import HDF5PairDataset


def test_hdf5_pair_dataset_crops_with_shared_coordinates(tmp_path: Path) -> None:
    path = tmp_path / "pairs.h5"
    noisy = np.arange(1 * 8 * 8 * 3, dtype=np.uint8).reshape(1, 8, 8, 3)
    clean = np.flip(noisy, axis=2).copy()
    with h5py.File(path, "w") as handle:
        handle.create_dataset("noisy", data=noisy)
        handle.create_dataset("gt", data=clean)

    dataset = HDF5PairDataset(path, crop_size=4, crop_mode="center")
    noisy_tensor, clean_tensor = dataset[0]
    assert tuple(noisy_tensor.shape) == (3, 4, 4)
    assert tuple(clean_tensor.shape) == (3, 4, 4)
    assert dataset.image_shape == (8, 8, 3)
    assert float(noisy_tensor.min()) >= -1.0
    assert float(noisy_tensor.max()) <= 1.0


def test_hdf5_pair_dataset_resizes_to_requested_size(tmp_path: Path) -> None:
    path = tmp_path / "pairs.h5"
    image = np.arange(1 * 16 * 12 * 3, dtype=np.uint8).reshape(1, 16, 12, 3)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("noisy", data=image)
        handle.create_dataset("gt", data=image)

    dataset = HDF5PairDataset(path, resize_size=8)
    noisy_tensor, clean_tensor = dataset[0]
    assert tuple(noisy_tensor.shape) == (3, 8, 8)
    assert tuple(clean_tensor.shape) == (3, 8, 8)
