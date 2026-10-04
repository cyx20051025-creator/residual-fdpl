# Model Zoo

## Core Assets

The six final checkpoints are available in the public `v3.1-review` GitHub
Release. The executable source is tag `v3.1-review-8`. The Release contains ten
downloadable assets:

| Asset | Contents | Verification |
|---|---|---|
| `rcab3_seed42_fdpl_final.pth` | 3-RCAB, seed 42, Residual FDPL enabled | `CHECKPOINT_SHA256SUMS` |
| `rcab3_seed42_nofdpl_final.pth` | matched 3-RCAB, seed 42, no Residual FDPL | `CHECKPOINT_SHA256SUMS` |
| `rcab3_seed43_fdpl_final.pth` | 3-RCAB, seed 43, Residual FDPL enabled | `CHECKPOINT_SHA256SUMS` |
| `rcab3_seed43_nofdpl_final.pth` | matched 3-RCAB, seed 43, no Residual FDPL | `CHECKPOINT_SHA256SUMS` |
| `rcab3_seed44_fdpl_final.pth` | 3-RCAB, seed 44, Residual FDPL enabled | `CHECKPOINT_SHA256SUMS` |
| `rcab3_seed44_nofdpl_final.pth` | matched 3-RCAB, seed 44, no Residual FDPL | `CHECKPOINT_SHA256SUMS` |
| `CHECKPOINT_SHA256SUMS` | SHA-256 digests for the six checkpoints | direct inspection |
| `SIDD_QUICK_EVAL_SHA256SUMS` | SHA-256 digests for 2,048 extracted PNG files | direct inspection |
| `SIDD_QUICK_EVAL_ZIP_SHA256SUMS` | SHA-256 digest for the archive | direct inspection |
| `sidd_quick_eval_256.zip` | 1,024 noisy/ground-truth SIDD+ validation pairs | upstream notice applies |

No checkpoint is committed to ordinary Git history. Each checkpoint asset is
listed in `CHECKPOINT_SHA256SUMS`, distributed with the Release.

Historical per-image result JSON files and complete training histories are not
distributed. They can be regenerated only where the required checkpoints and
input data are available; the six Table 3 comparisons have a public evaluator.

## External Baselines

RIDNet, SwinIR, NAFNet, and other external model implementations must retain
their upstream attribution and license notices. Their pretrained weights are
not assumed to be redistributable.
