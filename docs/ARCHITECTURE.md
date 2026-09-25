# Repository Architecture

## Public Scope

The first release is a clean implementation of the v3.1 canonical pipeline.
It does not include historical model generations, cloud queue scripts, local
logs, datasets, checkpoints, or paper source.

```text
Residual-FDPL/
|-- README.md
|-- train.py
|-- pyproject.toml
|-- configs/
|   |-- stage0_gaussian.json
|   |-- sidd_stage1_64.json
|   |-- sidd_stage2_256.json
|   `-- eval_sidd.json
|-- docs/
|   |-- ARCHITECTURE.md
|   |-- DATA.md
|   |-- MODEL_ZOO.md
|   |-- OPEN_SOURCE_DECISIONS.md
|   |-- RELEASE_PLAN.md
|   `-- REPRODUCIBILITY.md
|-- scripts/
|   |-- build_sidd_hdf5.py
|   |-- evaluate.py
|   `-- smoke_test.py
|-- src/cvfdpl/
|   |-- data/
|   |-- losses/
|   |-- metrics/
|   |-- models/
|   |-- training/
|   `-- utils/
|-- tests/
`-- results/
```

## Ownership Boundaries

| Area | Responsibility |
|---|---|
| `models/` | 3-RCAB carrier and parameter count |
| `losses/` | Original FDPL, Residual FDPL, brightness-aware MSE, VGG perceptual loss |
| `data/` | PNG pairs, HDF5 pairs, cropping, and normalization |
| `training/` | Locked configurations, weight maps, EMA, calibration, loops, and checkpoints |
| `metrics/` | PSNR and Gaussian-window SSIM |
| `train.py` | User-facing canonical training entry point |
| `scripts/evaluate.py` | Paired-image direct or sliding evaluation |
| `results/` | Public placeholder; paper JSON stays in the private asset repo |

## Migration Map

The public tree is a curated migration of the canonical research entry points:

| Research entry point | Public boundary |
|---|---|
| `train_rrdb_rcab_gaussian.py` | `run_stage0_training` |
| `train_residual_fdpl_v3_1.py` | Weight maps, EMA, calibration, and stage loop |
| `train_rrdb_rcab_v3_1.py` | `run_sidd_training` with the 3-RCAB carrier |
| `dataset_sidd.py`, `sidd_to_hdf5_512.py` | `data.HDF5PairDataset` and `build_sidd_hdf5.py` |
| `eval_v31_bilinear_paired.py` | `scripts/evaluate.py` |

Historical dual-branch, NAFNet, RIDNet, SwinIR, cloud queue, and ablation
scripts are outside the first public maintenance surface. They can be added
later only after their licenses, provenance, and reproducible scope are
reviewed.

## Asset Policy

The Git repository stores source code, configurations, tests, documentation,
and small JSON results. It does not store:

- SIDD, Urban100, BSDS500, BSD68, or CBSD68 image files
- derived HDF5 caches
- model checkpoints or pretrained baseline weights
- logs, cloud credentials, local output directories, or paper material

Data and weights require separate redistribution review. If approved, they
belong in release assets or an external archive with SHA-256 manifests.

## Protocol Invariants

- The carrier is 4 x RRDB plus 3 x RCAB and has 197,819 parameters.
- Stage 0 uses Original FDPL.
- Stage 1 and Stage 2 use Residual FDPL with target share `alpha = 0.30`.
- The main training chain is 512 -> 256 bilinear cache, then bilinear resize
  to the 64 or 256 patch used by each stage.
- Weight maps are generated from 15,000 paired SIDD noise samples and fixed
  during training.
- Calibration changes only the loss scale, not the spatial-frequency map.
- Stage 1 and Stage 2 are executed in one process for the canonical result.
- Configuration files are strict: unknown keys and protocol inconsistencies
  fail before training.
- VGG19 weights can be supplied locally for offline runs.
- Reported metrics use matched same-seed comparisons.
