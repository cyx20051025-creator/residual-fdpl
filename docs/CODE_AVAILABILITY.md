# Code and Data Availability

## Repository Scope

The public repository contains source for two research components. The TVC
training/evaluation component is represented by source tag `v3.1-review-8` and
contains the 3-RCAB model, Original FDPL and Residual FDPL implementations,
calibration logic, SIDD data-preparation tools, direct and sliding evaluators,
and CPU-only tests. The `v3.1-review` Release supplies the checkpoints,
evaluation archive, and manifests.

The residual RGB-correction component is maintained under
`src/rgb_correction/` with frozen protocols, compact aggregate result
summaries, and archival replay scripts. It includes Rec.601 zero-luma
projection, the exact pre-clipping MSE identity, scene-balanced Ridge
calibration, scene-split validation, and risk-controlled action selection.

The complete SIDD and Urban100 datasets are obtained from their official
distribution channels and are not stored in Git. The `v3.1-review` Release provides
the exact 1,024-pair SIDD+ validation subset used in the paper under its
upstream research/education notice. The six final checkpoints are distributed
as Release assets with SHA-256 verification. Full per-image result files
remain outside ordinary Git history.

## Repository URL

<https://github.com/cyx20051025-creator/residual-fdpl>

The TVC source revision is tag `v3.1-review-8`; its asset Release
`v3.1-review` remains unchanged. The RGB-correction component is released from
the branch or later source tag containing `src/rgb_correction/`. A permanent
archival tag or DOI is added only when those details become available.

## Reproduction Scope

| Claim | Public evidence |
|---|---|
| 3-RCAB architecture and parameter count | `src/cvfdpl/models/rrdb_rcab.py`, `tests/test_model.py` |
| Original FDPL and Residual FDPL | `src/cvfdpl/losses/fdpl.py`, `tests/test_losses.py` |
| Locked Stage 0 and SIDD protocol | `configs/`, `src/cvfdpl/training/config.py` |
| Direct and sliding inference | `src/cvfdpl/evaluation.py`, `tests/test_evaluation.py` |
| Dataset preparation | `scripts/build_sidd_hdf5.py`, `docs/DATA.md` |
| Full Table 3 paired summary | release checkpoints plus `scripts/reproduce_table3.sh` |
| Six-row Table 3 paired summary | release checkpoints, `scripts/reproduce_table3.sh`, and machine-readable output |
| Historical paper JSON and training histories | not included; current scope is summarized in `docs/REPRODUCIBILITY.md` |
| RGB zero-luma and pre-clipping identities | `src/rgb_correction/ops.py`, `tests/test_rgb_correction_ops.py` |
| RGB paired Ridge calibration and split validation | `src/rgb_correction/calibration.py`, `src/rgb_correction/validation.py`, focused CPU tests |
| Risk-controlled correction selection | `src/rgb_correction/selection.py`, `rgb-correction select`, focused CPU tests |
| RGB cross-endpoint, budget, and selector summaries | `results/rgb_correction/summaries/`, frozen files under `protocols/rgb_correction/` |
| Full RGB image/checkpoint/feature replay | historical scripts are public, but upstream assets are not redistributed; see `docs/rgb_correction/REPRODUCIBILITY.md` |

The `v3.1-review` Release provides the 1,024 validation pairs used for the
reported comparison, together with `docs/SIDD_QUICK_EVAL_SHA256SUMS` and
`docs/SIDD_PLUS_LICENSE_AND_NOTICE.md`.

The RGB-correction source tree does not redistribute SIDD/SIDD+, PolyU,
CC-noise, NIND, RENOIR, DND, denoiser checkpoints, or third-party pretrained
weights. The compact result files are aggregate records; they do not replace
the original data licenses or make the historical scripts a turnkey pipeline.
