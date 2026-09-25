# Code and Data Availability

## Manuscript Statement

The source code, locked training configuration, and evaluation entry points
for the canonical 3-RCAB pipeline are available in the public repository tagged
`v3.1-review.4`. The release contains the model, Original FDPL and Residual FDPL
implementations, calibration logic, SIDD data-preparation tools, direct and
sliding evaluators, and CPU-only tests.

The complete SIDD and Urban100 datasets are obtained from their official
distribution channels and are not stored in Git. The review Release provides
the exact 1,024-pair SIDD+ validation subset used in the paper under its
upstream research/education notice. The six final checkpoints are distributed
as Release assets with SHA-256 verification. Full per-image result files
remain outside ordinary Git history.

## Repository URL

<https://github.com/cyx20051025-creator/residual-fdpl>

The fixed revision for the second review is tag `v3.1-review.4`. After acceptance,
the final release will use tag `v3.1.0` and a persistent archive DOI when
available.

## Reproduction Scope

| Claim | Public evidence |
|---|---|
| 3-RCAB architecture and parameter count | `src/cvfdpl/models/rrdb_rcab.py`, `tests/test_model.py` |
| Original FDPL and Residual FDPL | `src/cvfdpl/losses/fdpl.py`, `tests/test_losses.py` |
| Locked Stage 0 and SIDD protocol | `configs/`, `src/cvfdpl/training/config.py` |
| Direct and sliding inference | `src/cvfdpl/evaluation.py`, `tests/test_evaluation.py` |
| Dataset preparation | `scripts/build_sidd_hdf5.py`, `docs/DATA.md` |
| Full Table 3 paired summary | release checkpoints plus `scripts/reproduce_table3.sh` |
| Exact paper table values | private release JSON mapped in `docs/PAPER_ARTIFACTS.md` |

The `v3.1-review.4` Release provides the 1,024 validation pairs used for the
reported comparison, together with `docs/SIDD_QUICK_EVAL_SHA256SUMS` and
`docs/SIDD_PLUS_LICENSE_AND_NOTICE.md`.
