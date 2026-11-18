# Selector with an image-statistics-only Ridge candidate

Protocol frozen: 2026-11-17, before running the simplified selector.

## Question

The SIDD-CC source-feature ablation found that the 117 image-statistics
features match or slightly exceed the current 117-features-plus-source Ridge
map on every endpoint seed. This check asks whether removing the three source
components from the fitted Ridge candidate changes the frozen selector's
held-out decisions or risk control.

## Fixed factors

- Population: SIDD-CC benchmark, 928 crops from 29 scenes.
- Endpoints: project fine-tuned NAFNet seeds 62, 63 and 64.
- Ten fixed leaves with 16 calibration and 13 evaluation scenes.
- Split seed: `20261216`.
- Source predictor: seed 56.
- Candidate set, risk thresholds, Ridge penalty grid, scene balancing,
  zero-luma projection and `0.02` cap are unchanged.
- Simple candidates retain the source predictor because they are separate
  diagnostic/action families.

## Only changed factor

For the capped and uncapped Ridge candidates, the three source-offset columns
are replaced by zeros. The Ridge input is therefore the 117 image-statistics
features plus intercept only. The source-frozen, source-scaled and fixed-target
candidates remain exactly as specified in
`PROTOCOL_RISK_CONTROLLED_SELECTOR.md`.

## Metrics and comparisons

Report all frozen selector outputs: selected PSNR, SSIM, LPIPS and DISTS;
positive-split rate; severe-harm rate; action counts; and calibration-only
decision reasons. Compare paired split outcomes with the existing
features-plus-source selector on the exact same partitions.

## Decision rule

No split, endpoint, threshold, candidate or cap may change after observing
results.

- If the simplified selector matches or exceeds the current selector without
  an increase in negative or severe-harm splits, remove source from the
  deployable Ridge input.
- If it loses more than `0.05 dB` in selected mean PSNR or increases the
  negative-split rate, retain the current input and report source as having a
  selector-level but not always-on Ridge benefit.
- Small differences below that practical margin are reported as equivalent,
  with the simpler design preferred unless another metric reverses the
  conclusion.

These repeated endpoint checkpoints share scenes and are not independent
populations.
