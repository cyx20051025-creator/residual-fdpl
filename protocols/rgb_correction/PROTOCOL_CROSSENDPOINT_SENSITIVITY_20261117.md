# Cross-endpoint calibration sensitivity protocol

Frozen: 2026-11-17, before the sensitivity outputs were produced.

## Question

Does the cross-endpoint sign reversal depend on the primary 16-scene
calibration budget or the `0.02` offset cap?

## Design

- Six frozen endpoints: three fine-tuned NAFNet checkpoints, official
  NAFNet, SwinIR-M and Restormer.
- Donor and recipient matrix: every endpoint is used as both calibrator and
  recipient.
- Calibration budgets: 8, 16 and 24 scenes.
- Caps: `0.01`, `0.02` and `0.04`.
- Ten deterministic 16/13-scene partitions per budget, using split seed
  `20261216`.
- Ridge input: 117 image statistics plus intercept.
- Score: exact pre-clipping MSE identity from cached base errors and
  reference offsets.

The 16-scene, `0.02`-cap condition reproduces the primary diagnostic. The
new conditions test sensitivity; they do not replace rendered PSNR or
perceptual evaluation.

## Decision

- The manuscript retains the endpoint-conditioning conclusion if the
  fine-tuned-to-official-NAFNet transfer remains negative across all
  requested budgets and caps while fine-tuned-to-fine-tuned transfer
  remains positive.
- If a sign reversal appears only at the primary setting, the text reports
  the matrix as setting-dependent and does not use it as a general transfer
  claim.
- Repeated splits and budgets reuse the same 29 scenes and are not
  independent populations.
