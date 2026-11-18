# Cross-backbone SIDD-CC calibration-budget results

All rows use the 117-feature-only Ridge input, the frozen selector
thresholds, ten scene partitions and split seed `20261216`.

| Endpoint | Scenes | ΔPSNR | Fixed target | Selected−fixed | ΔSSIM | ΔLPIPS | ΔDISTS | Positive | Worst split | Actions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| restormer | 4 | +0.3589 | +0.6911 | -0.3322 | +0.000941 | -0.003190 | -0.001577 | 4/10 | -0.0000 | no_correction:6 / fixed_target:4 |
| restormer | 8 | +0.8953 | +0.8651 | +0.0302 | +0.002445 | -0.008096 | -0.003309 | 9/10 | -0.0000 | no_correction:1 / fixed_target:2 / target_ridge_capped:5 / target_ridge_uncapped:2 |
| restormer | 12 | +1.0234 | +0.8945 | +0.1289 | +0.003354 | -0.009831 | -0.004078 | 10/10 | +0.7139 | fixed_target:7 / target_ridge_capped:1 / target_ridge_uncapped:2 |
| restormer | 16 | +1.0779 | +0.8250 | +0.2529 | +0.004007 | -0.012775 | -0.004491 | 10/10 | +0.6993 | fixed_target:4 / target_ridge_capped:2 / target_ridge_uncapped:4 |
| restormer | 20 | +1.1196 | +0.8522 | +0.2673 | +0.004923 | -0.010499 | -0.004204 | 10/10 | +0.7837 | fixed_target:5 / target_ridge_uncapped:5 |
| restormer | 24 | +1.2411 | +0.8699 | +0.3712 | +0.008347 | -0.013239 | -0.005245 | 10/10 | +0.6608 | fixed_target:3 / target_ridge_uncapped:7 |
| swinir_seed62 | 4 | +0.0003 | -0.0927 | +0.0930 | +0.000099 | -0.000394 | -0.000041 | 9/10 | -0.1384 | no_correction:8 / source_scaled:1 / target_ridge_capped:1 |

Not yet available: `[('swinir_seed62', 8), ('swinir_seed62', 12), ('swinir_seed62', 16), ('swinir_seed62', 20), ('swinir_seed62', 24)]`.

Interpretation guardrail: both architectures are evaluated on the
same 29 reused scenes. Repeated splits and calibration-budget draws
are not independent populations, and the two backbones are not
independent training seeds of one architecture.
