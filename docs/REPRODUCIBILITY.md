# Reproducibility Notes

## Locked Protocol

The main v3.1 carrier and training values are:

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
| Direct seeds | 42, 43, 44 |

The Stage 1 and Stage 2 maps are precomputed and fixed during training.
Periodic calibration updates the loss scale, not the spatial-frequency map.
Both map resolutions use the same fixed map-statistics seed, `42`, regardless
of the model-training seed. The public trainer isolates the PyTorch RNG state
while building or loading maps so that map preparation does not alter the
subsequent training random stream.

The sliding implementation follows the archived protocol: tile origins are
`0, 96, 192, ...`; the final tile in each dimension is clipped to the image
boundary and can be smaller than 128 pixels.

## Important Reporting Boundaries

- Results use matched same-seed no-RFDPL comparisons.
- Cross-seed summaries use sample standard deviation.
- The seed-44 direct row uses the same matched final checkpoint pair as the
  sliding row and is included in the six-comparison reproduction script.
- Frequency-band and texture analyses retain their stated subset and
  descriptive boundaries.
- PSNR significance does not transfer automatically to SSIM.
- PSNR p-values use two-sided paired Student-t tests and are conditional on
  the evaluated final checkpoints; they do not estimate training-seed
  uncertainty.
- RIDNet and SwinIR-windowed are boundary evidence, not a cross-architecture
  ranking.

## Original FDPL Convention

`OriginalFDPLoss` follows the legacy convention used by the TVC experiments:

- the amplitude spectra are computed with unshifted `torch.fft.fft2`
- the fixed radial mask is indexed around the array center, not around the
  zero-frequency bin
- the two weighted amplitude errors use `torch.nn.functional.mse_loss`, which
  averages over all elements

The TVC manuscript explicitly distinguishes this grid convention from the
shifted Fourier representation used by Residual FDPL. The public repository
does not contain the original CMC implementation, so these notes document the
behavior of this code and do not assert bitwise equivalence with the published
CMC model.

## Public Reproduction Scope

| Scope | Status | Evidence |
|---|---|---|
| CPU tests and smoke path | fully reproducible from the public source tag | `tests/`, `scripts/smoke_test.py` |
| Six Table 3 checkpoint comparisons | fully reproducible with the released checkpoints and evaluation bundle | `scripts/reproduce_table3.sh` |
| Direct and sliding evaluation implementation | public | `src/cvfdpl/evaluation.py` |
| Stage 0 and SIDD training implementation | public | `train.py`, `configs/`, `src/cvfdpl/training/` |
| Original CMC Gaussian benchmark pipeline | not included | earlier CMC publication |
| Historical training from scratch | not independently reproduced in this release | shared Stage-0 artifacts, exact training histories, and some experiment checkpoints are not public |
| Historical per-image paper JSON | not distributed | users can regenerate the six main Table 3 comparisons from released assets |
| Alternative-carrier experiments | partially represented in the manuscript, not fully packaged here | RIDNet, SwinIR-windowed, and related training paths are not release assets |

## Environment

Two validations have been run:

| Validation | Scope | Environment |
|---|---|---|
| Source-only clean-room audit, 2026-09-25 | install, tests, and public source paths | Python 3.11.9, macOS, package versions in `requirements/validated.txt` |
| Full checkpoint-evaluation reproduction, 2026-10-04 | clean clone, asset digest checks, 12 complete 1,024-pair evaluations, six paired Table 3 comparisons, and independent Student-t recomputation | Python 3.11.9, PyTorch 2.12.1, torchvision 0.27.1, NumPy 2.4.6, Pillow 12.3.0, h5py 3.16.0, SciPy 1.17.1, Apple M5 Pro with MPS |

The exact package versions from the source-only audit are recorded in
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

The minimum supported ranges remain in `pyproject.toml`. Linux CUDA support is
part of the public target matrix but has not been independently verified in
the two audits above.

The archived runs seeded Python, NumPy, PyTorch, and CUDA but did not enable
cuDNN deterministic algorithms. The default public trainer matches that
behavior. Use `--deterministic` only as a stricter repeatability option.

## Maintenance Status

- The clean source tree covers the main Stage 0, Stage 1/2, and
  direct/sliding evaluation paths.
- Data-free evaluator coverage and sliding-grid regression tests are included
  in `tests/`.
- Historical paper-result regression JSON is not included. The six
  checkpoints and the 1,024-pair SIDD+ evaluation archive are available from
  the `v3.1-review` Release under their separate terms.
- The asset Release remains immutable. Documentation-only corrections are
  published under a new source tag rather than by moving or replacing existing
  tags and Release assets.
