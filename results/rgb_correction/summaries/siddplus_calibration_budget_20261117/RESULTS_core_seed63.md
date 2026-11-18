# SIDD+ calibration-budget results

Endpoint: project fine-tuned NAFNet seed 63.

Protocol: `PROTOCOL.md`, frozen before running budgets beyond the
existing 16-image setting. This table validates all five budgets against
the same split seed and unchanged selector policy.

## Validation

- Five budget files each contain ten split records.
- Every partition contains the declared calibration budget and the
  disjoint remaining image-level evaluation set.
- The 16-image run reproduces the prior selected actions and split
  assignments. Rendered PSNR agrees within GPU nondeterminism
  (`2e-4 dB` tolerance).
- The budget subsets use matched random draws but are not forced to be
  nested, so differences by budget are descriptive.

## Selected results

| Budget | ΔPSNR | SSIM | LPIPS | DISTS | Positive splits | Mean severe-harm rate | Worst image | Selected actions |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | +0.5795 | +0.009597 | -0.002655 | +0.000276 | 10/10 | 12.0635% | -5.3106 | source_frozen:1 / source_scaled:5 / fixed_target:3 / target_ridge_capped:1 |
| 64 | +0.6005 | +0.010474 | -0.002660 | +0.000396 | 10/10 | 11.5104% | -4.3291 | source_frozen:2 / source_scaled:4 / fixed_target:4 |
| 128 | +0.7934 | +0.013738 | -0.004069 | +0.000030 | 10/10 | 12.3326% | -5.8428 | source_frozen:1 / fixed_target:5 / target_ridge_capped:4 |
| 256 | +0.8514 | +0.014514 | -0.004548 | -0.000257 | 10/10 | 11.8750% | -3.9127 | fixed_target:4 / target_ridge_capped:6 |
| 512 | +1.0314 | +0.019112 | -0.004024 | +0.000071 | 10/10 | 11.4844% | -5.4962 | fixed_target:1 / target_ridge_capped:1 / target_ridge_uncapped:8 |

## Always-on controls and selector gap

| Budget | Selector | Fixed target | Selector−fixed [95%] | Source frozen | Source scaled | Capped Ridge |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | +0.5795 | +0.8272 | -0.2476 [-0.3575, -0.1332] | +0.3143 | +0.4934 | +0.5762 |
| 64 | +0.6005 | +0.8555 | -0.2550 [-0.3986, -0.1094] | +0.3134 | +0.5056 | +0.3377 |
| 128 | +0.7934 | +0.8591 | -0.0657 [-0.1920, +0.0054] | +0.3118 | +0.5036 | +0.8161 |
| 256 | +0.8514 | +0.8700 | -0.0186 [-0.0364, -0.0015] | +0.3136 | +0.5067 | +0.8368 |
| 512 | +1.0314 | +0.8926 | +0.1388 [+0.0829, +0.1842] | +0.3167 | +0.5161 | +0.8610 |

## Interpretation

The highest selector mean occurs at budget 512 (+1.0314 dB), but budget choice is not a
new tuning decision: the fixed sweep is reported in full.

At budget 16 the selector is behind the fixed target by -0.2476 dB. The per-budget
selector-minus-fixed column should be read together with selected
actions and severe-harm rates. A selector that catches the fixed target
only by increasing calibration density but remains below it at the
original budget is still a harm-control result rather than a
mean-accuracy improvement.

No result at a larger budget changes the endpoint, source rule,
candidate family, thresholds, cap or split seed.

SIDD+ is image-level validation: the available manifest has no verified
scene grouping. Repeated splits reuse images and are not independent
populations.
