# Reproducibility Notes

## Locked Protocol

The canonical v3.1 carrier and training values are:

| Component | Value |
|---|---|
| Carrier | 4 x RRDB + 3 x RCAB |
| Parameters | 197,819 |
| Stage 0 | Urban100 Gaussian pretraining with Original FDPL |
| Training data chain | 512 -> 256 bilinear cache, then bilinear resize per stage |
| Stage 1 | SIDD 64 x 64 fine-tuning |
| Stage 2 | SIDD 256 x 256 fine-tuning |
| FDPL target share | `alpha = 0.30` |
| FDPL maximum schedule weight | `0.08` |
| VGG perceptual weight | `0.01` |
| Map statistics | 15,000 SIDD noisy-clean pairs |
| Reported checkpoints | final checkpoints |
| Sliding inference | 128-pixel tiles, 32-pixel overlap, stride 96, boundary-clipped tails |
| Direct inference | 256 x 256 |
| Sliding seeds | 42, 43, 44 |
| Direct seeds | 42, 43 |

The Stage 1 and Stage 2 maps are precomputed and fixed during training.
Periodic calibration updates the loss scale, not the spatial-frequency map.

The sliding implementation follows the archived protocol: tile origins are
`0, 96, 192, ...`; the final tile in each dimension is clipped to the image
boundary and can be smaller than 128 pixels.

## Important Reporting Boundaries

- Results use matched same-seed no-RFDPL comparisons.
- Cross-seed summaries use sample standard deviation.
- Seed 44 direct results are not reported because that direct run was not
  trained.
- Frequency-band and texture analyses retain their stated subset and
  descriptive boundaries.
- PSNR significance does not transfer automatically to SSIM.
- RIDNet and SwinIR-windowed are boundary evidence, not a cross-architecture
  ranking.

## Environment

The clean-room validation baseline was Python 3.11.9 on macOS. The exact
package versions from the independent source-only audit are recorded in
`requirements/validated.txt`:

| Component | Validated version |
|---|---|
| Python | 3.11.9 |
| PyTorch | 2.14.0 |
| torchvision | 0.29.0 |
| NumPy | 2.4.6 |
| Pillow | 12.3.0 |
| h5py | 3.16.0 |
| pytest | 9.1.1 |
| Ruff | 0.16.9 |
| build | 1.6.1 |

The minimum supported ranges remain in `pyproject.toml`. The earlier local
training baseline used PyTorch 2.12.1 and Apple MPS. Linux CUDA support is part
of the public target matrix but still requires a clean-room verification.

The archived runs seeded Python, NumPy, PyTorch, and CUDA but did not enable
cuDNN deterministic algorithms. The default public trainer matches that
behavior. Use `--deterministic` only as a stricter repeatability option.

## Migration Status

- The clean source tree covers the canonical Stage 0, Stage 1/2, and
  direct/sliding evaluation paths.
- Data-free evaluator coverage and sliding-grid regression tests are included
  in `tests/`.
- Exact paper-result regression checks use private reference JSON and
  checkpoints rather than redistributing them in the public repository.
- The archived evaluation summary is checked against the private assets before
  the `v3.1-review.2` release is replaced or extended.
