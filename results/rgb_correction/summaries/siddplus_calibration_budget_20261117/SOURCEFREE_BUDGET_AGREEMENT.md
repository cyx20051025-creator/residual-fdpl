# Source-free SIDD+ budget agreement

Rendered results are paired on the exact same calibration and evaluation
partitions. Intervals resample the ten split assignments and are
descriptive because the same images recur across splits.

| Seed | Budget | Source-free | Fixed target | Selector−fixed [95%] | Original | Difference [95%] | Identical actions | Actions |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 62 | 16 | +0.5765 | +0.8227 | -0.2462 [-0.3530, -0.1269] | +0.5765 | +0.0000 [+0.0000, +0.0000] | 10/10 | source_frozen:1 / source_scaled:5 / fixed_target:3 / target_ridge_capped:1 |
| 62 | 64 | +0.5962 | +0.8524 | -0.2562 [-0.3977, -0.1123] | +0.5962 | +0.0000 [+0.0000, +0.0000] | 10/10 | source_frozen:2 / source_scaled:4 / fixed_target:4 |
| 62 | 128 | +0.7911 | +0.8554 | -0.0643 [-0.1896, +0.0066] | +0.7911 | -0.0000 [-0.0000, +0.0000] | 10/10 | source_frozen:1 / fixed_target:5 / target_ridge_capped:4 |
| 62 | 256 | +0.8567 | +0.8671 | -0.0104 [-0.0222, +0.0017] | +0.8566 | +0.0001 [+0.0001, +0.0001] | 10/10 | fixed_target:5 / target_ridge_capped:5 |
| 62 | 512 | +0.9928 | +0.8895 | +0.1034 [+0.0443, +0.1624] | +0.9927 | +0.0001 [+0.0001, +0.0001] | 10/10 | fixed_target:3 / target_ridge_capped:1 / target_ridge_uncapped:6 |
| 63 | 16 | +0.5796 | +0.8272 | -0.2477 [-0.3600, -0.1315] | +0.5795 | +0.0000 [+0.0000, +0.0000] | 10/10 | source_frozen:1 / source_scaled:5 / fixed_target:3 / target_ridge_capped:1 |
| 63 | 64 | +0.6005 | +0.8555 | -0.2550 [-0.3950, -0.1164] | +0.6005 | +0.0000 [+0.0000, +0.0000] | 10/10 | source_frozen:2 / source_scaled:4 / fixed_target:4 |
| 63 | 128 | +0.7934 | +0.8591 | -0.0657 [-0.1903, +0.0054] | +0.7934 | -0.0000 [-0.0000, +0.0000] | 10/10 | source_frozen:1 / fixed_target:5 / target_ridge_capped:4 |
| 63 | 256 | +0.8514 | +0.8700 | -0.0186 [-0.0360, -0.0022] | +0.8514 | -0.0000 [-0.0000, +0.0000] | 10/10 | fixed_target:4 / target_ridge_capped:6 |
| 63 | 512 | +1.0314 | +0.8926 | +0.1388 [+0.0832, +0.1845] | +1.0314 | -0.0000 [-0.0000, -0.0000] | 10/10 | fixed_target:1 / target_ridge_capped:1 / target_ridge_uncapped:8 |
| 64 | 16 | +0.5690 | +0.8151 | -0.2460 [-0.3558, -0.1286] | — | — | — | source_frozen:1 / source_scaled:5 / fixed_target:3 / target_ridge_capped:1 |
| 64 | 64 | +0.6234 | +0.8432 | -0.2198 [-0.3690, -0.0730] | — | — | — | source_frozen:2 / source_scaled:3 / fixed_target:5 |
| 64 | 128 | +0.7872 | +0.8470 | -0.0598 [-0.1843, +0.0098] | — | — | — | source_frozen:1 / fixed_target:5 / target_ridge_capped:4 |
| 64 | 256 | +0.8376 | +0.8568 | -0.0192 [-0.0355, -0.0024] | — | — | — | fixed_target:3 / target_ridge_capped:7 |
| 64 | 512 | +1.0204 | +0.8801 | +0.1403 [+0.0884, +0.1854] | — | — | — | fixed_target:1 / target_ridge_capped:1 / target_ridge_uncapped:8 |

Interpretation guardrail: this is a paired ablation on reused
SIDD+ units, not an independent-population comparison.
