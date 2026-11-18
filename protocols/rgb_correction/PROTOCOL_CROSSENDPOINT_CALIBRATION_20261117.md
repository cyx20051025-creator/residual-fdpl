# Cross-endpoint calibration diagnostic

Status: exploratory diagnostic added after the primary selector results.

## Question

Does a target-Ridge map fitted on one frozen endpoint supply the right
image-statistic correction for another endpoint? This question asks how
strongly calibration should be conditioned on the checkpoint, rather than
using architecture as a proxy.

## Fixed data and split construction

- Population: the existing 928 paired SIDD-CC crops from 29 scenes.
- Split seed: `20261216`.
- Ten partitions, each with 16 calibration scenes and 13 evaluation scenes.
- All endpoints use the same row order, scene labels and partition scenes.
- Endpoints:
  - fine-tuned NAFNet seeds 62, 63 and 64;
  - released NAFNet;
  - fine-tuned SwinIR-M seed 62;
  - released Restormer.

## Procedure

For every ordered donor-recipient endpoint pair:

1. Fit a 117-statistic Ridge map to donor reference offsets on the 16
   calibration scenes.
2. Apply the fitted map to recipient features on the 13 evaluation scenes.
3. Project the predicted offset to zero luma.
4. Report both the frozen `0.02` cap and the uncapped prediction.
5. Score every crop with
   `base_mse - (target·offset)/6 + ||offset||^2/12`.

The same-checkpoint cells use the same procedure and serve as references.

## Validation

- The same-endpoint capped scores agree with the rendered source-feature
  ablation to within its expected clipping difference: `+0.6638/+0.6784/
  +0.6478` pre-clipping versus `+0.6659/+0.6806/+0.6500` rendered.
- All feature files contain 928 rows and 117 features in identical row and
  scene order.
- Every evaluated offset reconstructs a positive pre-clipping MSE.

## Interpretation boundary

These are pre-clipping PSNR diagnostics. They do not include clipping, SSIM,
LPIPS or DISTS and are not substitutes for rendered selector evaluation.
Repeated partitions reuse the same scenes and are not independent populations.
The matrix isolates map transfer under a fixed fitting budget; it does not
estimate a new deployment operating point.
