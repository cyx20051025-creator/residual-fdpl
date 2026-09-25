"""SIDD paired-image datasets."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def find_pairs(noisy_dir: str | Path, clean_dir: str | Path) -> list[tuple[Path, Path]]:
    """Match noisy and clean files by relative path."""

    noisy_root = Path(noisy_dir)
    clean_root = Path(clean_dir)
    clean_by_relative = {
        path.relative_to(clean_root): path
        for path in sorted(clean_root.rglob("*"))
        if path.is_file()
    }
    pairs: list[tuple[Path, Path]] = []
    for noisy_path in sorted(noisy_root.rglob("*")):
        if not noisy_path.is_file():
            continue
        clean_path = clean_by_relative.get(noisy_path.relative_to(noisy_root))
        if clean_path is not None:
            pairs.append((noisy_path, clean_path))
    if not pairs:
        raise FileNotFoundError(
            f"No matching noisy/clean pairs found in {noisy_root} and {clean_root}"
        )
    return pairs


class SIDDPairedDataset(Dataset):
    """Load matching RGB files and normalize them to ``[-1, 1]``."""

    def __init__(self, noisy_dir: str | Path, clean_dir: str | Path) -> None:
        self.pairs = find_pairs(noisy_dir, clean_dir)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        noisy_path, clean_path = self.pairs[index]
        noisy = np.asarray(Image.open(noisy_path).convert("RGB"), dtype=np.uint8).copy()
        clean = np.asarray(Image.open(clean_path).convert("RGB"), dtype=np.uint8).copy()
        if noisy.shape != clean.shape:
            raise ValueError(
                f"Shape mismatch for {noisy_path.name}: {noisy.shape} vs {clean.shape}"
            )
        noisy_tensor = torch.from_numpy(noisy).permute(2, 0, 1).float() / 255.0
        clean_tensor = torch.from_numpy(clean).permute(2, 0, 1).float() / 255.0
        return noisy_tensor.mul(2).sub(1), clean_tensor.mul(2).sub(1)


class HDF5PairDataset(Dataset):
    """Lazy HDF5 reader with paired cropping or resizing.

    The default output follows the training convention in this repository and
    maps uint8 values to ``[-1, 1]``. Set ``normalize=False`` for ``[0, 1]``.
    """

    def __init__(
        self,
        h5_path: str | Path,
        crop_size: int | None = None,
        resize_size: int | None = None,
        crop_mode: str = "center",
        augment: bool = False,
        normalize: bool = True,
    ) -> None:
        if crop_mode not in {"center", "random"}:
            raise ValueError("crop_mode must be 'center' or 'random'")
        if crop_size is not None and resize_size is not None:
            raise ValueError("use either crop_size or resize_size, not both")
        self.h5_path = str(h5_path)
        self.crop_size = crop_size
        self.resize_size = resize_size
        self.crop_mode = crop_mode
        self.augment = augment
        self.normalize = normalize
        self._file: h5py.File | None = None
        with h5py.File(self.h5_path, "r") as handle:
            if "noisy" not in handle or "gt" not in handle:
                raise KeyError("HDF5 must contain 'noisy' and 'gt' datasets")
            if handle["noisy"].shape != handle["gt"].shape:
                raise ValueError("HDF5 noisy and gt dataset shapes must match")
            self.length = handle["noisy"].shape[0]
            self.image_shape = handle["noisy"].shape[1:]

    def _open(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        return self._file

    def _crop(self, noisy: np.ndarray, clean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.crop_size is None:
            return noisy, clean
        height, width = noisy.shape[:2]
        size = self.crop_size
        if height < size or width < size:
            raise ValueError(f"Crop size {size} exceeds image shape {noisy.shape}")
        if self.crop_mode == "center":
            top = (height - size) // 2
            left = (width - size) // 2
        else:
            top = int(torch.randint(0, height - size + 1, (1,)).item())
            left = int(torch.randint(0, width - size + 1, (1,)).item())
        return (
            noisy[top : top + size, left : left + size],
            clean[top : top + size, left : left + size],
        )

    def _resize(self, noisy: np.ndarray, clean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.resize_size is None:
            return noisy, clean
        size = (self.resize_size, self.resize_size)
        noisy_resized = np.asarray(
            Image.fromarray(noisy).resize(size, Image.BILINEAR),
            dtype=np.uint8,
        ).copy()
        clean_resized = np.asarray(
            Image.fromarray(clean).resize(size, Image.BILINEAR),
            dtype=np.uint8,
        ).copy()
        return noisy_resized, clean_resized

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        handle = self._open()
        noisy = np.asarray(handle["noisy"][index]).copy()
        clean = np.asarray(handle["gt"][index]).copy()
        noisy, clean = self._crop(noisy, clean)
        noisy, clean = self._resize(noisy, clean)
        if self.augment and torch.rand(1).item() < 0.5:
            noisy = noisy[:, ::-1].copy()
            clean = clean[:, ::-1].copy()
        noisy_tensor = torch.from_numpy(noisy).permute(2, 0, 1).float() / 255.0
        clean_tensor = torch.from_numpy(clean).permute(2, 0, 1).float() / 255.0
        if self.normalize:
            noisy_tensor = noisy_tensor.mul(2).sub(1)
            clean_tensor = clean_tensor.mul(2).sub(1)
        return noisy_tensor, clean_tensor

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self) -> None:
        self.close()
