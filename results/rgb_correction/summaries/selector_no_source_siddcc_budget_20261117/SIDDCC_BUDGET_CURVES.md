# SIDD-CC source-free calibration-budget results

The selector and data are unchanged; only the three source columns in
the Ridge candidates are zero. Matched original-input results exist for
4/8/12/16 scenes.

| Seed | Scenes | ΔPSNR | Fixed target | Selected−fixed | ΔSSIM | ΔLPIPS | ΔDISTS | Positive | Worst | Actions | Original | Paired diff [95%] | Action agreement |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 62 | 4 | +0.1592 | +0.2033 | -0.0441 | +0.001105 | -0.000802 | -0.000471 | 8/10 | -0.0000 | no_correction:6 / source_scaled:1 / target_ridge_capped:2 / target_ridge_uncapped:1 | +0.1592 | +0.0001 [-0.0000, +0.0002] | 10/10 |
| 62 | 8 | +0.7420 | +0.3851 | +0.3569 | +0.002996 | -0.005148 | -0.002144 | 9/10 | -0.0000 | no_correction:2 / fixed_target:1 / target_ridge_capped:4 / target_ridge_uncapped:3 | +0.7419 | +0.0001 [-0.0000, +0.0004] | 10/10 |
| 62 | 12 | +0.7913 | +0.5393 | +0.2519 | +0.003044 | -0.005465 | -0.002237 | 10/10 | +0.0000 | no_correction:1 / source_frozen:1 / fixed_target:2 / target_ridge_capped:3 / target_ridge_uncapped:3 | +0.7871 | +0.0041 [-0.0000, +0.0100] | 10/10 |
| 62 | 16 | +1.0216 | +0.2560 | +0.7656 | +0.005502 | -0.006335 | -0.002589 | 10/10 | +0.6166 | target_ridge_capped:4 / target_ridge_uncapped:6 | +1.0170 | +0.0045 [-0.0002, +0.0138] | 10/10 |
| 63 | 4 | +0.1669 | +0.2173 | -0.0504 | +0.001115 | -0.000843 | -0.000495 | 4/10 | -0.0000 | no_correction:6 / source_scaled:1 / target_ridge_capped:2 / target_ridge_uncapped:1 | +0.1669 | +0.0000 [-0.0005, +0.0005] | 10/10 |
| 63 | 8 | +0.6858 | +0.4004 | +0.2854 | +0.002202 | -0.004720 | -0.001903 | 8/10 | -0.0000 | no_correction:2 / source_frozen:1 / fixed_target:1 / target_ridge_capped:4 / target_ridge_uncapped:2 | +0.6857 | +0.0001 [-0.0000, +0.0003] | 10/10 |
| 63 | 12 | +0.8177 | +0.5531 | +0.2646 | +0.003152 | -0.005624 | -0.002315 | 9/10 | -0.0000 | no_correction:1 / source_frozen:1 / fixed_target:1 / target_ridge_capped:4 / target_ridge_uncapped:3 | +0.8133 | +0.0044 [+0.0002, +0.0103] | 10/10 |
| 63 | 16 | +1.0404 | +0.2674 | +0.7730 | +0.005511 | -0.006416 | -0.002653 | 10/10 | +0.6364 | target_ridge_capped:4 / target_ridge_uncapped:6 | +1.0324 | +0.0080 [-0.0001, +0.0197] | 10/10 |
| 63 | 20 | +1.2133 | +0.1872 | +1.0261 | +0.008055 | -0.006337 | -0.002794 | 10/10 | +0.6109 | target_ridge_capped:1 / target_ridge_uncapped:9 | — | — | — |
| 63 | 24 | +1.3495 | +0.5727 | +0.7768 | +0.010090 | -0.006410 | -0.002941 | 10/10 | +0.8851 | target_ridge_uncapped:10 | — | — | — |
| 64 | 4 | +0.1559 | +0.1682 | -0.0123 | +0.001086 | -0.000869 | -0.000481 | 4/10 | -0.0000 | no_correction:6 / source_scaled:1 / target_ridge_capped:2 / target_ridge_uncapped:1 | +0.1558 | +0.0001 [-0.0000, +0.0002] | 10/10 |
| 64 | 8 | +0.7338 | +0.3515 | +0.3823 | +0.002997 | -0.005215 | -0.002148 | 8/10 | -0.0000 | no_correction:2 / fixed_target:1 / target_ridge_capped:4 / target_ridge_uncapped:3 | +0.7337 | +0.0001 [-0.0000, +0.0003] | 10/10 |
| 64 | 12 | +0.7804 | +0.5124 | +0.2679 | +0.003042 | -0.005567 | -0.002242 | 9/10 | -0.0000 | no_correction:1 / source_frozen:1 / fixed_target:2 / target_ridge_capped:3 / target_ridge_uncapped:3 | +0.7763 | +0.0040 [-0.0000, +0.0097] | 10/10 |
| 64 | 16 | +1.0104 | +0.2228 | +0.7876 | +0.005506 | -0.006624 | -0.002633 | 10/10 | +0.6032 | target_ridge_capped:4 / target_ridge_uncapped:6 | +0.9560 | +0.0544 [-0.0002, +0.1634] | 9/10 |

Not yet available: `[(62, 20), (62, 24), (64, 20), (64, 24)]`.

The 20/24-scene points have no matched original-input selector
runs and therefore extend only the source-free curve. Repeated
splits reuse the 29 scenes and are not independent populations.
