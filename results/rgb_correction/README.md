# RGB-Correction Results

This directory contains compact public summaries from the residual RGB
correction study:

- Markdown tables for reading and review;
- CSV files for exact aggregate values; and
- selected split, threshold-sensitivity, and pairing-control JSON files.

The summaries cover cross-endpoint calibration, calibration-budget sweeps,
source-free selector replay, source-rule replication, threshold sensitivity,
target-pairing shuffle controls, and NIND/SIDD-CC/SIDD+ boundary checks.
The fitted source-rule models used for those analyses are in
`source_rules/`, together with their checksums and loading notes.

These are **not** complete per-image result trees. Raw images, feature caches,
denoiser checkpoints, third-party weights, and large intermediate JSON files
remain outside Git. Numerical values must be interpreted with the matching
protocol in `protocols/rgb_correction/` and the mapping in
`docs/rgb_correction/PAPER_ARTIFACTS.md`.

In particular:

- pre-clipping PSNR diagnostics are not rendered post-clipping PSNR;
- repeated splits over the same scenes are not independent populations;
- a selector's mean gain must be read with its severe-loss and perceptual
  guards; and
- cross-checkpoint applications over one dataset are not independent datasets.
