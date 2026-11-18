# SIDD-CC source-free selector calibration-budget protocol

Frozen: 2026-11-17, Europe/London, before any 4/8/12/20/24-scene
source-free selector output was produced.

## Question

The source-free SIDD+ sweep tests calibration density at the image level.
SIDD-CC can additionally test the complete risk-controlled selector with
scene-held-out calibration and evaluation. This experiment asks how the
117-feature-only selector changes as the number of paired calibration scenes
grows, and whether source removal alters the low-budget decisions.

This is a calibration-cost diagnostic on an already used population, not a
new blind validation set.

## Fixed factors

- Population: SIDD-CC benchmark, 928 crops from 29 scenes.
- Endpoints: project fine-tuned NAFNet seeds 62, 63 and 64.
- Source predictor: seed 56.
- Split seed: `20261216`.
- Ten partitions per calibration budget.
- Calibration budgets: 4, 8, 12, 16, 20 and 24 scenes.
- The existing 16-scene source-free run is reused rather than rerun.
- Candidate order, leave-one-scene-out calibration, quality guards, Ridge
  penalty grid, zero-luma projection, `0.02` cap and all selection
  thresholds remain exactly those in `PROTOCOL_RISK_CONTROLLED_SELECTOR.md`.
- The only method change is that the three source-offset columns supplied to
  the Ridge candidates are replaced by zeros. The source-only candidates
  remain unchanged.

## Comparators

- Existing features-plus-source selector at 4, 8, 12 and 16 scenes.
- The 117-only selector at 16 scenes.
- Reference-aware offline Ridge trends at 4, 8, 12, 16, 20 and 24 scenes.

The 20- and 24-scene full-selector runs have no matching original-input
selector outputs. They are descriptive continuation points, not matched
source-removal ablations.

## Metrics and statistics

Report rendered PSNR, SSIM, LPIPS and DISTS, selected actions, positive and
severe-harm split rates, worst split and the candidate means. For source
removal at matched budgets, report the exact paired split difference and a
bootstrap interval over the ten shared split assignments.

Calibration draws at different budgets are not forced to be nested and are
not independent populations. At 24 scenes only five scenes remain for outer
evaluation, so the high-budget endpoint is deliberately described as a
sensitivity diagnostic.

## Decision use

- If source-free actions agree with the original selector at 4, 8, 12 and 16
  scenes and the paired difference stays inside the predeclared `0.05 dB`
  equivalence margin, treat source removal as validated across budgets.
- If actions change but quality does not materially degrade, retain the
  simpler input and report the changed decisions.
- If the low-budget curve is unstable in either method, present calibration
  cost as endpoint- and population-dependent.
- Do not search for a new budget after inspecting these outputs and do not
  replace the primary 16-scene operating point retrospectively.
