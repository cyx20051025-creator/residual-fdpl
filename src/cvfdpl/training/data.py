"""Datasets used by the canonical training pipeline."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class GaussianUrban100Dataset(Dataset):
    """Urban100 images with on-the-fly Gaussian noise in ``[-1, 1]``."""

    def __init__(self, root: str | Path, size: int = 64, sigma: float = 0.1) -> None:
        root_path = Path(root)
        if not root_path.is_dir():
            raise FileNotFoundError(f"Urban100 directory not found: {root_path}")
        self.paths = sorted(
            path
            for path in root_path.rglob("*")
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        if not self.paths:
            raise FileNotFoundError(f"No images found in {root_path}")
        self.transform = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )
        self.sigma = sigma

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        with Image.open(self.paths[index]) as image:
            clean = self.transform(image.convert("RGB"))
        noisy = (clean + self.sigma * torch.randn_like(clean)).clamp(-1, 1)
        return noisy, clean
