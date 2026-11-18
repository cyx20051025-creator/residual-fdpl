# NIND 56-scene calibration-budget protocol

Frozen: 2026-11-17, Europe/London, before any source-free NIND budget
result.

## Question

Does the endpoint-selector calibration curve remain stable on a larger
camera-scene population outside SIDD? This extends the source-free selector
from the five-budget SIDD+ check and the six-budget SIDD-CC check to the
56-scene NIND screen.

The NIND screen appeared in earlier project work. This is a supplementary
pressure test, not a new blind population.

## Fixed factors

- Population: `nind_remaining_56_scenes`, 1,344 crops from 56 scenes.
- Endpoints: fine-tuned NAFNet seeds 62 and 63, one per GPU.
- Source predictor: seed 56, recomputed on each endpoint's output.
- Split seed: `20261219`.
- Ten partitions per budget.
- Calibration budgets: 4, 8, 16, 28 and 40 scenes.
- Candidate family, leave-one-scene-out decision, Ridge grid, quality guards,
  zero-luma projection, `0.02` cap and thresholds remain frozen.
- Ridge input: 117 image-statistics features plus intercept; source columns
  are zero.

## Reporting

Report rendered PSNR, SSIM, LPIPS and DISTS, selected action counts, positive
and severe-harm split rates, and the worst split. At the 28-scene budget,
compare with the existing features-plus-source result on the identical split
seed.

Calibration draws are not forced to be nested. Repeated leaves reuse scenes.
At 40 calibration scenes only 16 outer-evaluation scenes remain, so this
point is a sensitivity diagnostic rather than a new primary operating point.

## Decision use

- A positive curve with both endpoints supports calibration-cost robustness
  outside SIDD.
- A null or mixed curve defines a cross-population boundary and is reported
  without changing the main operating point.
- No budget, threshold or endpoint may be changed after observing results.
