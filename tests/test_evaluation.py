from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cvfdpl.evaluation import run_evaluation, sliding_inference
from cvfdpl.models import RRDBPlusLightRCAB


def test_sliding_inference_with_identity_model() -> None:
    image = torch.rand(1, 3, 8, 8)
    output = sliding_inference(torch.nn.Identity(), image, tile_size=4, stride=3)
    assert torch.allclose(output, image)


def test_evaluation_returns_per_image_json(tmp_path: Path) -> None:
    noisy_dir = tmp_path / "noisy"
    gt_dir = tmp_path / "gt"
    noisy_dir.mkdir()
    gt_dir.mkdir()
    rng = np.random.default_rng(5)
    noisy = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    clean = np.clip(noisy.astype(np.int16) - 2, 0, 255).astype(np.uint8)
    Image.fromarray(noisy).save(noisy_dir / "0001.png")
    Image.fromarray(clean).save(gt_dir / "0001.png")

    model = RRDBPlusLightRCAB(num_rrdb=4, num_rcab=3)
    checkpoint = tmp_path / "model.pth"
    torch.save(model.state_dict(), checkpoint)
    result = run_evaluation(
        checkpoint,
        noisy_dir,
        gt_dir,
        protocol="direct",
        device="cpu",
    )
    assert result["images"] == 1
    assert result["per_image"][0]["noisy"] == "0001.png"
    assert np.isfinite(result["summary"]["psnr"]["mean"])
