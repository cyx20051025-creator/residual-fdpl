# Selector with image-statistics-only Ridge

Protocol: `PROTOCOL.md`, frozen before the full run. Only the Ridge
input changed: the three source columns were replaced by zeros.
Candidates, thresholds, splits, projection and cap are unchanged.

## Validation

- Three endpoint files parsed with ten split records each.
- Calibration and evaluation scene lists exactly match the original
  features-plus-source selector files.
- All four selected-result metrics are finite.

## Endpoint results

| Seed | No-source ΔPSNR | Original ΔPSNR | Paired difference | SSIM | LPIPS | DISTS | Positive splits | Fixed target | Actions |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 62 | +1.0216 | +1.0170 | +0.0045 | +0.005502 | -0.006335 | -0.002589 | 10/10 | +0.2560 | target_ridge_capped:4 / target_ridge_uncapped:6 |
| 63 | +1.0404 | +1.0324 | +0.0080 | +0.005511 | -0.006416 | -0.002653 | 10/10 | +0.2674 | target_ridge_capped:4 / target_ridge_uncapped:6 |
| 64 | +1.0104 | +0.9560 | +0.0544 | +0.005506 | -0.006624 | -0.002633 | 10/10 | +0.2228 | target_ridge_capped:4 / target_ridge_uncapped:6 |

The three-endpoint paired difference is +0.0223 dB
(stratified split-bootstrap 95% interval [+0.0012, +0.0600]).

Selected actions are identical in 29/30 splits.
Changed actions: seed 64 split 3 (target_ridge_capped -> target_ridge_uncapped).

## Decision

The simplified selector is equivalent within the predeclared
`0.05 dB` practical margin and does not increase the negative-split
rate. Use the 117-feature-only Ridge in the deployable method and
retain the source predictor as a simple candidate/diagnostic.

These three endpoints share scenes and split assignments. They are not
independent populations.
