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
| Sliding inference | 128 x 128 tiles, stride 96 |
| Direct inference | 256 x 256 |
| Sliding seeds | 42, 43, 44 |
| Direct seeds | 42, 43 |

The Stage 1 and Stage 2 maps are precomputed and fixed during training.
Periodic calibration updates the loss scale, not the spatial-frequency map.

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

The local validation baseline was Python 3.11 with PyTorch using Apple MPS.
Linux CUDA support is part of the public target matrix but still requires a
clean-room verification.

The archived runs seeded Python, NumPy, PyTorch, and CUDA but did not enable
cuDNN deterministic algorithms. The default public trainer matches that
behavior. Use `--deterministic` only as a stricter repeatability option.

## Remaining Migration Work

- Add a data-free evaluator smoke fixture
- Add exact result regression tests
- Re-run the locked evaluation summary after refactoring
