# RGB-Correction Scripts

This directory preserves the paper-specific analysis code without coupling it to
the maintained `rgb_correction` package.

## Contents

- `legacy/` contains compact calibration, replay, bootstrap, aggregation, and
  plotting scripts used for the later cross-endpoint and selector analyses.
- `pipeline/` contains the archived RGB-offset pilot feature, fitting,
  evaluation, and summarization pipeline. Its `README.md` and `PROTOCOL.md`
  describe that experiment.

## Status

The scripts are archival. They are copied so reviewers and readers can inspect
the exact analysis logic, but they are not presented as a turnkey end-to-end
pipeline. Several scripts expect experiment-local relative paths such as
cached features, frozen model outputs, per-image records, or a working
directory produced outside this repository.

The maintained operations and decision policy live in `src/rgb_correction/` and
are covered by CPU tests. Prefer the package API when reusing the method.

## Required Inputs

The repository does not include upstream images, denoiser checkpoints, feature
caches, fitted third-party weights, or metric model weights. Users must obtain
those artifacts under their providers' terms and verify any published checksum
or manifest before running an analysis.

Before replaying a script:

1. Read the corresponding file in `protocols/rgb_correction/` or the local
   `pipeline/PROTOCOL.md`.
2. Confirm whether scores are pre-clipping diagnostics or rendered metrics.
3. Confirm the split unit and whether repeated draws reuse the same population.
4. Keep generated outputs outside Git history.
5. Do not execute unknown cloud launcher scripts or reuse credentials from an
   older experiment package.

See `docs/rgb_correction/REPRODUCIBILITY.md` for the public verification levels.
