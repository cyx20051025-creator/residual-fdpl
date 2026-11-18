# SIDD+ validation calibration-budget check

Protocol frozen: 2026-11-17, before running any additional calibration-budget
size on this endpoint.

## Question

The previous frozen check used 16 image-level calibration units on SIDD+
validation. The primary SIDD-CC selector uses 16 scenes and up to 512 paired
crops. Does the apparent advantage of the fixed target offset on SIDD+
validation persist when the calibration budget is increased to a comparable
number of paired observations?

## Fixed factors

- Population: `sidd_valid_1024`, 1024 paired 256x256 images.
- Population SHA-256:
  `a442b6b8313469519537755e9e4c3a0c027462637eb2088bac137ecd172053e7`.
- Endpoint: project fine-tuned NAFNet seed 62.
- Checkpoint SHA-256:
  `f1c5379dee4db835344e9eb4427825d6bca21a5b636957f4ba14b8bc680b35a6`.
- Frozen source predictor: seed 56.
- Source artifact SHA-256:
  `9d0cd219e12c088deb5da287116a09b2b535eea999e41a36d47b9b78319d780a`.
- Candidate list, risk thresholds, zero-luma projection, Ridge grid, and
  `0.02` model-coordinate norm cap are unchanged from
  `PROTOCOL_RISK_CONTROLLED_SELECTOR.md`.
- Ten fixed random leaves, split seed `20261127`.
- Bootstrap samples: 2000.

## Only changed factor

Calibration units:

- 16 images;
- 64 images;
- 128 images;
- 256 images;
- 512 images.

For each budget, the remaining images form the outer evaluation set. An image
never appears in both partitions within a leaf. The same split seed is reused
across budgets to provide a matched comparison, but the implementation samples
each calibration subset directly without forcing the smaller subsets to be
nested within the larger ones. Budget differences are therefore descriptive
comparisons over matched random draws, not causal effects of individual
samples. Repeated leaves reuse images and are not independent corpora.

## Metrics and comparisons

For every budget, report rendered:

- PSNR;
- SSIM;
- LPIPS;
- DISTS.

The selected action is compared with no correction and every always-on
candidate, especially:

- fixed target offset;
- capped Ridge;
- uncapped Ridge;
- source-frozen and source-scaled rules.

Also report:

- selected action counts;
- positive split count;
- severe-harm rate (`PSNR < -0.10 dB`);
- pooled image-level positive rate;
- worst image event.

## Decision rules

No result from 64, 128, 256 or 512 calibration units may change the endpoint,
source rule, candidate family, thresholds, cap or split seed.

Interpretation:

- If the selector approaches or exceeds fixed target as calibration increases,
  the previous 16-unit gap is evidence about calibration density rather than
  a stable failure of the selection family.
- If the gap remains large while severe harm improves, report the selector as
  a harm-control procedure, not a mean-accuracy method.
- If the selector neither catches fixed target nor improves harm, remove the
  selector from the main contribution and place it in supplementary analysis.
- If fixed target loses its advantage as calibration increases, update the
  paper's cross-population boundary rather than selecting whichever budget is
  most favorable.

No claim may call this scene-level validation: the available manifest has one
unique image ID and no verified scene grouping.
