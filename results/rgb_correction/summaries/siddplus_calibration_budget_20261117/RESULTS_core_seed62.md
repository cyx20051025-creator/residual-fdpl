# SIDD+ calibration-budget results

Endpoint: project fine-tuned NAFNet seed 62.

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
| 16 | +0.5765 | +0.009554 | -0.002620 | +0.000266 | 10/10 | 11.9643% | -5.4276 | source_frozen:1 / source_scaled:5 / fixed_target:3 / target_ridge_capped:1 |
| 64 | +0.5962 | +0.010410 | -0.002607 | +0.000393 | 10/10 | 11.4792% | -4.3843 | source_frozen:2 / source_scaled:4 / fixed_target:4 |
| 128 | +0.7911 | +0.013649 | -0.004024 | +0.000008 | 10/10 | 12.3438% | -5.7596 | source_frozen:1 / fixed_target:5 / target_ridge_capped:4 |
| 256 | +0.8566 | +0.014741 | -0.004433 | -0.000121 | 10/10 | 12.4349% | -3.8325 | fixed_target:5 / target_ridge_capped:5 |
| 512 | +0.9927 | +0.018253 | -0.004073 | +0.000124 | 10/10 | 12.3047% | -5.3243 | fixed_target:3 / target_ridge_capped:1 / target_ridge_uncapped:6 |

## Always-on controls and selector gap

| Budget | Selector | Fixed target | Selector−fixed [95%] | Source frozen | Source scaled | Capped Ridge |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | +0.5765 | +0.8227 | -0.2462 [-0.3549, -0.1323] | +0.3121 | +0.4883 | +0.5766 |
| 64 | +0.5962 | +0.8524 | -0.2562 [-0.3994, -0.1106] | +0.3112 | +0.5009 | +0.3375 |
| 128 | +0.7911 | +0.8554 | -0.0643 [-0.1899, +0.0065] | +0.3096 | +0.4989 | +0.8139 |
| 256 | +0.8566 | +0.8670 | -0.0104 [-0.0223, +0.0021] | +0.3114 | +0.5021 | +0.8349 |
| 512 | +0.9927 | +0.8894 | +0.1034 [+0.0427, +0.1616] | +0.3146 | +0.5116 | +0.8596 |

## Interpretation

The highest selector mean occurs at budget 512 (+0.9927 dB), but budget choice is not a
new tuning decision: the fixed sweep is reported in full.

At budget 16 the selector is behind the fixed target by -0.2462 dB. The per-budget
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
