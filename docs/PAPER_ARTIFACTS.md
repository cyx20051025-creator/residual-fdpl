# Paper-to-Code and Result Mapping

This file maps the paper artifacts to the public implementation, locked
configuration, and private reference results. Private JSON files are staged in
the companion full-release repository and are not committed to the public
source tree.

| Paper artifact | Public code or configuration | Private reference result |
|---|---|---|
| Table 3, main 3-RCAB comparison | `train.py`, `configs/sidd_stage1_64.json`, `configs/sidd_stage2_256.json`, `scripts/reproduce_table3.sh` | `paired_rcab_redo_sliding.json`, `paired_rcab_redo_direct256.json`, `paired_s44_sliding.json` |
| Table 4, component ablations | `src/cvfdpl/losses/fdpl.py`, `configs/sidd_stage1_64.json`, `configs/sidd_stage2_256.json` | `paired_ablation_abc_sliding.json`, `paired_ablation_df_sliding.json`, `paired_ablation_final_sliding.json`, `table4_detail_final_2026.json` |
| Table 5, sensitivity analysis | `src/cvfdpl/training/weight_map.py`, `src/cvfdpl/losses/fdpl.py` | `paired_ablation_final_sliding.json`, `paired_new_20260909.json` |
| Table 6, RCAB capacity curve | `src/cvfdpl/models/rrdb_rcab.py`, `train.py` | `paired_rcab_redo_sliding.json`, `paired_rcab_redo_direct256.json` |
| Table 7, crop-50 detail metrics | public evaluator plus private metric script | `table4_detail_final_2026.json`, `rcab3_detail_crops_2026.json`, `rcab3_pixel_pct_50_2026.json` |
| Fig. 5, main same-seed comparison | `scripts/run_eval_all.sh` | three paired-result JSON files listed under Table 3 |
| Fig. 6, SIDD spectrum concentration | `src/cvfdpl/training/weight_map.py` | `sidd_spectrum_stats.json` |
| Fig. 7, frequency-band mechanism | `src/cvfdpl/losses/fdpl.py` | `band_mse_decomposition.json`, `per_image_band_gain_corr_full1024.json`, `per_image_band_error_rcab3_full1024.json` |
| Fig. 8, visual and error comparison | `src/cvfdpl/evaluation.py` | seed-42 paired results and crop-50 detail files |
| Efficiency context | `src/cvfdpl/models/rrdb_rcab.py`, `scripts/run_eval_all.sh` | `latency_memory_final_2026.json` |

## Table 3 Checkpoint Map

The following release assets correspond to the five rows in Table 3. Both
assets in a row use the same training seed and must be evaluated under the
listed protocol; the reported quantity is their same-seed paired PSNR gain.

| Table 3 row | Protocol | Seed | Matched no-RFDPL asset | Residual FDPL asset |
|---|---|---|---|---|
| 1 | sliding `128-tile / 32-overlap / stride-96` | 42 | `rcab3_seed42_nofdpl_final.pth` | `rcab3_seed42_fdpl_final.pth` |
| 2 | sliding `128-tile / 32-overlap / stride-96` | 43 | `rcab3_seed43_nofdpl_final.pth` | `rcab3_seed43_fdpl_final.pth` |
| 3 | sliding `128-tile / 32-overlap / stride-96` | 44 | `rcab3_seed44_nofdpl_final.pth` | `rcab3_seed44_fdpl_final.pth` |
| 4 | direct `256` | 42 | `rcab3_seed42_nofdpl_final.pth` | `rcab3_seed42_fdpl_final.pth` |
| 5 | direct `256` | 43 | `rcab3_seed43_nofdpl_final.pth` | `rcab3_seed43_fdpl_final.pth` |

Seed 44 was trained and evaluated only under sliding inference. Absolute PSNR
values are not compared across seeds.

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
- Table and figure numbering must be checked against the final manuscript
  before release; this mapping describes the v3.1 artifact layout, not a
  substitute for the frozen table notes.
