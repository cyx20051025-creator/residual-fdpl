# Target-pairing shuffle control

Each shuffled replicate preserves the calibration features and offset
distribution but permutes their correspondence before fitting Ridge.
Scores are exact pre-clipping PSNR changes on the same 13 outer scenes.

| Endpoint | Actual map | Shuffled mean | Paired difference | Positive splits | Actual win rate vs shuffles |
|---:|---:|---:|---:|---:|---:|
| 62 | +1.2255 | -0.3434 | +1.5689 | 10/10 | 100.0% |
| 63 | +1.2456 | -0.3229 | +1.5685 | 10/10 | 100.0% |
| 64 | +1.2178 | -0.4039 | +1.6218 | 10/10 | 100.0% |

This is a mechanism control without clipping or perceptual
metrics. Repeated splits reuse the same 29 scenes.
