# Cross-backbone calibration-budget protocol

Frozen: 2026-11-17, Europe/London, before any new cross-backbone budget
output.

## Question

Does the scene-level calibration-density behaviour observed on the three
fine-tuned NAFNet checkpoints hold for independently trained denoiser
families? This experiment extends the 117-feature-only selector to a
fine-tuned SwinIR-M endpoint and the official Restormer real-noise checkpoint
on the same SIDD-CC population.

It is a robustness diagnostic on an already used population. It does not
create an untouched confirmation set and cannot establish architecture-level
causality.

## Fixed factors

- Population: SIDD-CC benchmark, 928 crops from 29 scenes.
- Endpoints:
  - SwinIR-M seed 62, project fine-tuned real-noise checkpoint.
  - Official Restormer real-image-denoising checkpoint.
- Source predictor: seed 56, recomputed on each endpoint's own images.
- Split seed: `20261216`.
- Ten partitions per calibration budget.
- Calibration budgets: 4, 8, 12, 16, 20 and 24 scenes.
- Candidate family, leave-one-scene-out risk selection, quality guards,
  Ridge grid, zero-luma projection, `0.02` cap and every threshold remain
  frozen.
- The Ridge candidates receive 117 image-statistics features and an
  intercept; the three source columns are zero.

## Reporting

Report rendered PSNR, SSIM, LPIPS and DISTS, selected action counts, positive
and severe-harm split rates, and the worst split. Use the same scene-level
partitions as the main selector. Keep the two architectures separate; do not
pool them as additional independent training seeds.

Because only 29 scenes exist, larger calibration budgets leave fewer
outer-evaluation scenes. All budget points are descriptive and the draws are
not forced to be nested.

## Decision use

- If the selector remains positive as calibration grows for both families,
  present the density effect as cross-backbone robustness.
- If one family needs more calibration or changes action, report an
  endpoint-specific calibration-cost boundary.
- Do not alter budgets, thresholds or the operating point after observing
  these outputs.
