"""Deterministic HDF5 sampling for paired architecture screens."""

from __future__ import annotations

import hashlib
from pathlib import Path

import h5py
import numpy as np
import torch


def file_sha256(path: str | Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def plan_sha256(indices: np.ndarray, flip_h: np.ndarray, flip_v: np.ndarray, rot: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in (indices, flip_h, flip_v, rot):
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def make_canonical_plan(
    length: int,
    steps: int,
    batch_size: int,
    stream_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Create one paired sample/augmentation stream reused by every arm."""
    if length <= 0 or steps <= 0 or batch_size <= 0:
        raise ValueError("length, steps, and batch_size must be positive")
    batches_per_epoch = int(np.ceil(length / batch_size))
    indices = np.empty((steps, batch_size), dtype=np.int64)
    flip_h = np.empty((steps, batch_size), dtype=np.bool_)
    flip_v = np.empty((steps, batch_size), dtype=np.bool_)
    rotations = np.empty((steps, batch_size), dtype=np.int8)

    rng = np.random.RandomState(stream_seed)
    for step in range(steps):
        epoch = step // batches_per_epoch
        position = step % batches_per_epoch
        if position == 0:
            epoch_rng = np.random.RandomState(stream_seed + epoch)
            permutation = epoch_rng.permutation(length)
        start = position * batch_size
        selected = permutation[start : start + batch_size]
        if len(selected) < batch_size:
            wrap = epoch_rng.permutation(length)
            selected = np.concatenate((selected, wrap[: batch_size - len(selected)]))
        indices[step] = selected
        flip_h[step] = rng.random_sample(batch_size) < 0.5
        flip_v[step] = rng.random_sample(batch_size) < 0.5
        rotations[step] = rng.randint(0, 4, size=batch_size).astype(np.int8)

    return indices, flip_h, flip_v, rotations, plan_sha256(indices, flip_h, flip_v, rotations)


def save_plan(
    path: str | Path,
    indices: np.ndarray,
    flip_h: np.ndarray,
    flip_v: np.ndarray,
    rotations: np.ndarray,
) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        indices=indices,
        flip_h=flip_h,
        flip_v=flip_v,
        rotations=rotations,
    )
    return plan_sha256(indices, flip_h, flip_v, rotations)


def load_plan(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    with np.load(Path(path), allow_pickle=False) as archive:
        indices = np.asarray(archive["indices"], dtype=np.int64)
        flip_h = np.asarray(archive["flip_h"], dtype=np.bool_)
        flip_v = np.asarray(archive["flip_v"], dtype=np.bool_)
        rotations = np.asarray(archive["rotations"], dtype=np.int8)
    return indices, flip_h, flip_v, rotations, plan_sha256(
        indices, flip_h, flip_v, rotations
    )


class HDF5Pairs:
    def __init__(self, path: str | Path, *, normalized: bool = True) -> None:
        self.path = Path(path)
        with h5py.File(self.path, "r") as handle:
            if "noisy" not in handle or "gt" not in handle:
                raise KeyError("HDF5 must contain noisy and gt arrays")
            self.noisy = np.asarray(handle["noisy"][:], dtype=np.uint8)
            self.clean = np.asarray(handle["gt"][:], dtype=np.uint8)
        if self.noisy.shape != self.clean.shape or self.noisy.ndim != 4:
            raise ValueError(
                f"unexpected pair shapes: noisy={self.noisy.shape}, gt={self.clean.shape}"
            )
        self.normalized = normalized

    def __len__(self) -> int:
        return int(self.noisy.shape[0])

    def batch(
        self,
        indices: np.ndarray,
        flip_h: np.ndarray,
        flip_v: np.ndarray,
        rotations: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noisy = torch.from_numpy(
            np.asarray(self.noisy[indices], dtype=np.float32) / 255.0
        ).permute(0, 3, 1, 2)
        clean = torch.from_numpy(
            np.asarray(self.clean[indices], dtype=np.float32) / 255.0
        ).permute(0, 3, 1, 2)
        if self.normalized:
            noisy = noisy * 2.0 - 1.0
            clean = clean * 2.0 - 1.0

        for row in range(noisy.shape[0]):
            if bool(flip_h[row]):
                noisy[row] = noisy[row].flip(-1)
                clean[row] = clean[row].flip(-1)
            if bool(flip_v[row]):
                noisy[row] = noisy[row].flip(-2)
                clean[row] = clean[row].flip(-2)
            turns = int(rotations[row])
            if turns:
                noisy[row] = torch.rot90(noisy[row], turns, dims=(-2, -1))
                clean[row] = torch.rot90(clean[row], turns, dims=(-2, -1))
        return noisy.contiguous(), clean.contiguous()
