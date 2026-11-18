# Source-free selector replay on SIDD+

Protocol frozen: 2026-11-17, before the SIDD+ source-free selector run.

## Purpose

The SIDD-CC replay tests whether removing source columns from the fitted Ridge
candidate preserves the original risk-controlled decisions. This addendum asks
the same question on the separate SIDD+ boundary population at the original
16-image calibration budget.

## Fixed factors

- Population: `sidd_valid_1024`.
- Endpoint: project fine-tuned NAFNet seed 62.
- Source predictor: seed 56.
- Ten image-level leaves with 16 calibration and 1008 evaluation images.
- Split seed: `20261127`.
- Candidate set, thresholds, projection and `0.02` cap are unchanged.

## Only changed factor

The source columns are removed from the fitted capped and uncapped Ridge
candidates. Source-frozen, source-scaled and fixed-target candidates remain
unchanged.

## Comparison

Compare the source-free run with the completed 16-image SIDD+ budget run on the
same partitions:

- selected PSNR, SSIM, LPIPS and DISTS;
- selected actions in all ten splits;
- positive and severe-harm rates;
- selector-minus-fixed-target difference.

## Decision rule

This is a boundary replay, not a new hyperparameter search. If source-free and
original selectors choose the same or equivalent actions without worsening
risk, the method simplification is supported cross-population. If they differ,
the source columns have a selection-stage effect even though the always-on
Ridge maps were numerically equivalent.
