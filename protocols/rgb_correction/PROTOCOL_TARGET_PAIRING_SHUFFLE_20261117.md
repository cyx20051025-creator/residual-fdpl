# Target-pairing shuffle control

Frozen: 2026-11-17, before this control was run.

## Question

Does the source-free Ridge gain depend on the correspondence between each
calibration image's 117 statistics and its reference-derived RGB offset, or
can a similar gain arise from the marginal offset distribution alone?

## Design

- Endpoints: fine-tuned NAFNet seeds 62, 63 and 64.
- Population: SIDD-CC, 928 crops from 29 scenes.
- Calibration budget: 16 scenes; evaluation uses the remaining 13 scenes.
- Ten fixed partitions with split seed `20261216`.
- Ridge input: 117 image statistics plus intercept.
- Reference offsets and predictions receive the same zero-luma projection.
- Score: exact pre-clipping PSNR change from each crop's base MSE.

For each split, the actual model is compared with 20 shuffled controls. A
shuffled control preserves every calibration feature row and every target
offset, but randomly permutes the calibration-row correspondence between
them before fitting. The Ridge grid, split, recipient endpoint and
pre-clipping evaluation are otherwise identical.

## Decision

- The pairing interpretation is retained if the actual map beats the
  shuffled-control distribution in most splits and endpoints.
- A comparable shuffled result would indicate that the marginal distribution
  of offsets explains the gain, and the attribution to image-conditioned
  prediction would be removed.

This is a mechanism control without clipping or perceptual metrics. Repeated
splits reuse the same scenes.
