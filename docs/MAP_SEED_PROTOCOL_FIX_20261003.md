# Shared Weight-Map Seed Protocol Fix

Date: 2026-10-03.

The public SIDD trainer previously passed each stage's training seed to
`get_or_compute_weight_map`. As a result, seeds 43 and 44 selected different
15,000-pair subsets when the maps were not already cached. That behavior
contradicted the locked protocol, which specifies maps estimated once from
seed-42 statistics and shared across training seeds 42, 43, and 44.

## Correction

- The Stage 1 and Stage 2 map computations now use the fixed
  `WEIGHT_MAP_SEED = 42`.
- Map construction isolates the PyTorch RNG state, so temporary map-sampling
  seeds cannot alter the subsequent training random stream.
- Regression tests verify both the fixed map seed and RNG-state preservation.

The existing released checkpoints and evaluation assets are unchanged. This
fix affects training-from-scratch reproducibility through the public trainer;
it does not alter Table 3, the final-checkpoint evaluation, or any reported
manuscript number.

## Verification

```text
PYTHONPATH=src python3 -m pytest -q
35 passed

python3 -m ruff check src/cvfdpl/training/pipeline.py \
  src/cvfdpl/training/weight_map.py tests/test_training.py
All checks passed
```
