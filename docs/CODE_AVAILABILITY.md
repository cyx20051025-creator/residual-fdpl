# Code and Data Availability

## Manuscript Statement

The source code, locked training configuration, and evaluation entry points
for the canonical 3-RCAB pipeline are available in the public repository tagged
`v3.1-review`. The release contains the model, Original FDPL and Residual FDPL
implementations, calibration logic, SIDD data-preparation tools, direct and
sliding evaluators, and CPU-only tests.

The SIDD and Urban100 datasets are obtained from their official distribution
channels and are not redistributed by this repository. The repository provides
the expected directory layout and HDF5 preparation scripts. The six final
checkpoints are distributed as release assets with SHA-256 verification. Full
per-image result files and any dataset subset whose redistribution is not
confirmed remain outside the ordinary Git history.

## Repository URL

The final GitHub URL must be inserted after the repository owner is confirmed:

```text
https://github.com/<owner>/residual-fdpl
```

The fixed revision for the second review is tag `v3.1-review`. After acceptance,
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
| Full evaluation summary | release checkpoints plus `scripts/run_eval_all.sh` |
| Exact paper table values | private release JSON mapped in `docs/PAPER_ARTIFACTS.md` |
