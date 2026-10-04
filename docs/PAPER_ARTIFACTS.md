# Paper-to-Code and Result Mapping

This file maps the TVC manuscript artifacts to executable public code,
implementation components, and reference evidence. "Exact regeneration" means
that released inputs and a public script can reproduce the reported numerical
summary. "Implementation component" means that the relevant method or
operator is public, but checkpoints or the full historical pipeline for that
experiment are not all included.

| Paper artifact | Public executable path | Public output or implementation component | Historical reference |
|---|---|---|---|
| Table 3, main 3-RCAB comparison | `scripts/reproduce_table3.sh` with the six released checkpoints | `table3_reproduced.json`, `table3_reproduced.csv`, and per-image paired JSON | Historical paired JSON is not included |
| Table 4, component ablations | `src/cvfdpl/losses/fdpl.py`, `configs/sidd_stage1_64.json`, `configs/sidd_stage2_256.json` | implementation components; ablation checkpoints are not included | `paired_ablation_*` and `table4_detail_final_2026.json` |
| Table 5, sensitivity analysis | `src/cvfdpl/training/weight_map.py`, `src/cvfdpl/losses/fdpl.py` | implementation components; sensitivity checkpoints are not included | `paired_ablation_final_sliding.json`, `paired_new_20260909.json` |
| Table 6, RCAB capacity curve | `src/cvfdpl/models/rrdb_rcab.py`, `train.py` | implementation components; capacity variants are not included | `paired_rcab_redo_sliding.json`, `paired_rcab_redo_direct256.json` |
| Table 7, crop-50 detail metrics | public evaluator plus a separate metric-analysis script | evaluation implementation is public; crop metric script is not included | `table4_detail_final_2026.json`, `rcab3_detail_crops_2026.json`, `rcab3_pixel_pct_50_2026.json` |
| Fig. 5, main same-seed comparison | `scripts/reproduce_table3.sh` | six paired gains and per-image outputs needed for the plot | Figure-generation script is not included |
| Fig. 6, SIDD spectrum concentration | `src/cvfdpl/training/weight_map.py` | map-estimation implementation | `sidd_spectrum_stats.json` |
| Fig. 7, frequency-band mechanism | `src/cvfdpl/losses/fdpl.py` | spectral-loss implementation | `band_mse_decomposition.json`, `per_image_band_gain_corr_full1024.json`, `per_image_band_error_rcab3_full1024.json` |
| Fig. 8, visual and error comparison | `src/cvfdpl/evaluation.py` | direct and sliding evaluator | seed-42 paired results and crop-50 detail files |
| Efficiency context | `src/cvfdpl/models/rrdb_rcab.py`, `scripts/run_eval_all.sh` | model and evaluator implementation | `latency_memory_final_2026.json` |

## Table 3 Checkpoint Map

The following release assets correspond to the six rows in Table 3. Both
assets in a row use the same training seed and must be evaluated under the
listed protocol; the reported quantity is their same-seed paired PSNR gain.

| Table 3 row | Protocol | Seed | Matched no-RFDPL asset | Residual FDPL asset |
|---|---|---|---|---|
| 1 | sliding `128-tile / 32-overlap / stride-96` | 42 | `rcab3_seed42_nofdpl_final.pth` | `rcab3_seed42_fdpl_final.pth` |
| 2 | sliding `128-tile / 32-overlap / stride-96` | 43 | `rcab3_seed43_nofdpl_final.pth` | `rcab3_seed43_fdpl_final.pth` |
| 3 | sliding `128-tile / 32-overlap / stride-96` | 44 | `rcab3_seed44_nofdpl_final.pth` | `rcab3_seed44_fdpl_final.pth` |
| 4 | direct `256` | 42 | `rcab3_seed42_nofdpl_final.pth` | `rcab3_seed42_fdpl_final.pth` |
| 5 | direct `256` | 43 | `rcab3_seed43_nofdpl_final.pth` | `rcab3_seed43_fdpl_final.pth` |
| 6 | direct `256` | 44 | `rcab3_seed44_nofdpl_final.pth` | `rcab3_seed44_fdpl_final.pth` |

The same seed-44 checkpoint pair is evaluated under both sliding and direct
inference. Absolute PSNR values are not compared across seeds.

The released checkpoints and a verified SIDD+ validation copy can be evaluated
with one command:

```bash
CHECKPOINT_DIR=/path/to/checkpoints \
NOISY_DIR=/path/to/siddplus_valid_noisy_srgb \
GT_DIR=/path/to/siddplus_valid_gt_srgb \
OUTPUT_DIR=runs/reproduce_table3 \
bash scripts/reproduce_table3.sh
```

## Protocol Notes

- The main carrier is 4 x RRDB plus 3 x RCAB and has 197,819 parameters.
- Stage 0 uses Original FDPL; Stages 1 and 2 use Residual FDPL with target
  share `alpha = 0.30`.
- The FDPL maximum schedule weight is `0.08`, the VGG perceptual weight is
  `0.01`, and weight maps are computed from 15,000 SIDD pairs.
- Sliding inference uses 128 x 128 tiles with stride 96. Trailing tiles are
  clipped at the image boundary.
- This mapping follows the v3.1 manuscript artifact numbering. Figure 5 values
  can be reconstructed from the six rows emitted by
  `scripts/reproduce_table3.sh`; the paper plotting script itself is not
  included in this source release.
