# SIDD+ calibration-budget replication on a second endpoint

Protocol frozen: 2026-11-17, before running the seed-63 budget sweep.

## Purpose

Repeat the already frozen SIDD+ calibration-budget sweep on a second
independently trained NAFNet endpoint. This tests whether any change in the
selector-versus-fixed-target relation is specific to seed 62.

## Fixed factors

The population, image-level folds, split seed, candidate family, risk
thresholds, projection, cap, source predictor, starting checkpoints and
metrics are unchanged from `PROTOCOL.md`.

- Population: `sidd_valid_1024`.
- Endpoint: project fine-tuned NAFNet seed 63.
- Source predictor: seed 56.
- Folds: ten fixed image-level leaves.
- Split seed: `20261127`.
- Calibration budgets: 16, 64, 128, 256 and 512 images.
- Cap: `0.02` in model coordinates.

The seed-63 feature file and source-offset file already exist and are not tuned
using budget results.

## Decision rule

Report seed 63 separately and beside the seed-62 curve. Do not pool the two
endpoints as independent populations.

The replication supports a general claim only if the same qualitative
selector-versus-fixed-target pattern appears at both endpoints. If seed 63
reverses the pattern, retain the cross-endpoint variation as part of the
method's boundary rather than selecting the favorable seed.
