# Source-free selector: nine-way source-rule replay

For each endpoint and split, only the two target-Ridge summaries are
taken from the source-free GPU run. Candidate summaries that depend on
the source rule remain from that source-specific original run. The
frozen decision function is then replayed. Source-56 rows match the
actual source-free GPU records exactly.

| Seed | Source | Selected ΔPSNR | Original | Difference | Worst change | Split interval | Positive | Worst | Actions |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 62 | 56 | +1.0216 | +1.0170 | +0.0045 | -0.0010 | [+0.8226, +1.2228] | 10/10 | +0.6166 | target_ridge_capped:4 / target_ridge_uncapped:6 |
| 62 | 57 | +1.0216 | +1.0216 | -0.0000 | -0.0008 | [+0.8205, +1.2197] | 10/10 | +0.6166 | target_ridge_capped:4 / target_ridge_uncapped:6 |
| 62 | 58 | +1.0216 | +1.0134 | +0.0082 | -0.0006 | [+0.8199, +1.2186] | 10/10 | +0.6166 | target_ridge_capped:4 / target_ridge_uncapped:6 |
| 63 | 56 | +1.0404 | +1.0324 | +0.0080 | -0.0007 | [+0.8351, +1.2380] | 10/10 | +0.6364 | target_ridge_capped:4 / target_ridge_uncapped:6 |
| 63 | 57 | +1.0404 | +0.9837 | +0.0567 | -0.0009 | [+0.8397, +1.2367] | 10/10 | +0.6364 | target_ridge_capped:4 / target_ridge_uncapped:6 |
| 63 | 58 | +1.0404 | +1.0313 | +0.0091 | -0.0003 | [+0.8429, +1.2367] | 10/10 | +0.6364 | target_ridge_capped:4 / target_ridge_uncapped:6 |
| 64 | 56 | +1.0104 | +0.9560 | +0.0544 | -0.0007 | [+0.8029, +1.2115] | 10/10 | +0.6032 | target_ridge_capped:4 / target_ridge_uncapped:6 |
| 64 | 57 | +1.0104 | +1.0104 | -0.0000 | -0.0006 | [+0.8185, +1.2074] | 10/10 | +0.6032 | target_ridge_capped:4 / target_ridge_uncapped:6 |
| 64 | 58 | +1.0104 | +1.0017 | +0.0087 | -0.0003 | [+0.8163, +1.2031] | 10/10 | +0.6032 | target_ridge_capped:4 / target_ridge_uncapped:6 |

Across the nine applications, the selected mean ranges from `+1.0104` to `+1.0404 dB`; `9/9` applications and all 90 repeated split outcomes are positive.

These are repeated checkpoint/source applications over the same 29 SIDD-CC scenes, not nine independent populations.
