# NIND remaining-56 source-free calibration-budget results

All rows use the 117-statistic Ridge input, frozen selector
thresholds, ten scene partitions and split seed `20261219`.

| Seed | Scenes | ΔPSNR | Fixed target | Selected−fixed | ΔSSIM | ΔLPIPS | ΔDISTS | Positive | Worst split | Actions |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 62 | 4 | -0.0029 | -0.2883 | +0.2854 | +0.000105 | +0.000125 | +0.000033 | 2/10 | -0.0354 | no_correction:8 / source_frozen:1 / source_scaled:1 |
| 63 | 4 | -0.0043 | -0.2957 | +0.2914 | +0.000198 | +0.000211 | +0.000059 | 4/10 | -0.0317 | no_correction:7 / source_frozen:1 / source_scaled:2 |
| 63 | 8 | -0.0000 | -0.1294 | +0.1294 | -0.000000 | +0.000000 | +0.000000 | 3/10 | -0.0000 | no_correction:10 |
| 63 | 16 | -0.0046 | -0.1171 | +0.1126 | +0.000100 | +0.000144 | +0.000044 | 6/10 | -0.0473 | no_correction:8 / source_frozen:1 / source_scaled:1 |
| 63 | 28 | +0.0000 | -0.1509 | +0.1509 | -0.000000 | +0.000000 | -0.000000 | 6/10 | -0.0000 | no_correction:10 |
| 63 | 40 | -0.0000 | -0.1172 | +0.1172 | -0.000000 | +0.000000 | -0.000000 | 3/10 | -0.0000 | no_correction:10 |

Not yet available: `[(62, 8), (62, 16), (62, 28), (62, 40)]`.

Interpretation guardrail: NIND remaining-56 appeared in earlier
project work and is a pressure test, not a blind population.
Repeated splits reuse the same scenes. At 40 calibration scenes
only 16 outer-evaluation scenes remain, so that point is a
sensitivity diagnostic rather than a new operating point.
