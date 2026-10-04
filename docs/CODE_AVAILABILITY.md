# Code and Data Availability

## Repository Scope

The source code, versioned training configuration, and evaluation entry points
for the main 3-RCAB pipeline are available in the public repository tagged
`v3.1-review-8`. The source tag contains the model, Original FDPL and Residual
FDPL implementations, calibration logic, SIDD data-preparation tools, direct
and sliding evaluators, and CPU-only tests. The `v3.1-review` Release supplies
the checkpoints, evaluation archive, and manifests.

The complete SIDD and Urban100 datasets are obtained from their official
distribution channels and are not stored in Git. The `v3.1-review` Release provides
the exact 1,024-pair SIDD+ validation subset used in the paper under its
upstream research/education notice. The six final checkpoints are distributed
as Release assets with SHA-256 verification. Full per-image result files
remain outside ordinary Git history.

## Repository URL

<https://github.com/cyx20051025-creator/residual-fdpl>

The current source revision is tag `v3.1-review-8`. The asset Release
`v3.1-review` remains unchanged. A permanent archival tag or DOI is added only
when those details become available.

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

The `v3.1-review` Release provides the 1,024 validation pairs used for the
reported comparison, together with `docs/SIDD_QUICK_EVAL_SHA256SUMS` and
`docs/SIDD_PLUS_LICENSE_AND_NOTICE.md`.
