# RGB-Correction Paper-to-Code Map

This table maps the major analysis components to the maintained API, frozen
protocols, and compact public results. The historical scripts are evidence and
replay aids; the maintained package is the stable interface.

| Analysis component | Maintained implementation | Frozen protocol | Compact result |
|---|---|---|---|
| Rec.601 zero-luma correction geometry | `src/rgb_correction/ops.py` | `PROTOCOL_TARGET_PAIRING_SHUFFLE_20261117.md` | `TARGET_PAIRING_SHUFFLE.md`, `TARGET_PAIRING_SHUFFLE.csv` |
| Exact pre-clipping MSE and PSNR change | `src/rgb_correction/ops.py` | `PROTOCOL_CROSSENDPOINT_CALIBRATION_20261117.md` | `CROSSENDPOINT_MATRIX.md`, `CROSSENDPOINT_MATRIX.csv` |
| Scene-balanced target Ridge | `src/rgb_correction/calibration.py` | `PROTOCOL_CROSSENDPOINT_CALIBRATION_20261117.md` | `CROSSENDPOINT_MATRIX.md` |
| Calibration-budget sensitivity | `src/rgb_correction/calibration.py`, replay scripts | `PROTOCOL_CROSSENDPOINT_SENSITIVITY_20261117.md` | `CROSSENDPOINT_SENSITIVITY.md`, `CROSSENDPOINT_SENSITIVITY.csv` |
| Cross-backbone calibration budgets | scene-balanced Ridge plus replay scripts | `PROTOCOL_CROSSBACKBONE_CALIBRATION_BUDGET_20261117.md` | `CROSSBACKBONE_BUDGET_SUMMARY.md`, `CROSSBACKBONE_BUDGET_SUMMARY.csv` |
| NIND calibration-budget pressure test | scene-balanced Ridge plus replay scripts | `PROTOCOL_NIND_CALIBRATION_BUDGET_20261117.md` | `NIND_BUDGET_SUMMARY.md`, `NIND_BUDGET_SUMMARY.csv` |
| SIDD-CC source-free selector curve | Ridge, risk policy, and replay scripts | `PROTOCOL_SIDDCC_SOURCE_FREE_BUDGET_20261117.md` | `SIDDCC_BUDGET_CURVES.md`, `SIDDCC_BUDGET_CURVES.csv` |
| SIDD+ calibration-budget boundary | Ridge and replay scripts | `PROTOCOL_POSTCORE_20261117.md` plus result-local protocols | `CORE_RESULTS_20261117.md`, seed and agreement summaries |
| Nine-way source-rule replay | `src/rgb_correction/selection.py` plus replay script | `PROTOCOL_NINEWAY_SOURCEFREE_REPLAY_20261117.md` | `NINEWAY_SUMMARY.md`, `NINEWAY_SUMMARY.csv` |
| Threshold sensitivity | `src/rgb_correction/selection.py` plus replay script | `PROTOCOL_NINEWAY_SOURCEFREE_REPLAY_20261117.md` | `threshold_sensitivity.md`, `threshold_grid.csv`, `threshold_sensitivity.json` |
| Target-pairing shuffle control | Ridge fitting plus replay script | `PROTOCOL_TARGET_PAIRING_SHUFFLE_20261117.md` | `TARGET_PAIRING_SHUFFLE.md`, `TARGET_PAIRING_SHUFFLE.csv` |
| RGB-offset pilot and external metrics | archived pilot pipeline | `scripts/rgb_correction/pipeline/PROTOCOL.md` | pilot results live in the archived package, not in the compact summary tree |

## Candidate Actions

The selector recognizes five correction actions:

- `source`: reuse the source-domain rule;
- `scaled_source`: reuse its direction with a calibrated scale;
- `fixed_target`: apply one target-population offset;
- `capped_ridge`: predict a per-image target offset with the frozen cap; and
- `uncapped_ridge`: predict a per-image target offset without that cap.

`no_correction` is the abstention result. The source and fixed-target actions are
intentionally kept as simple alternatives so the per-image predictor must earn
its additional complexity under the same held-out evidence.

## Statistical Units

The matching protocol is authoritative for whether the split unit is a scene,
image, or repeated checkpoint application. `validate_scene_split` checks only
that supplied calibration and evaluation units are disjoint; it cannot infer
the independence of repeated splits or recover missing grouping metadata.

## Result Scope

The files in `results/rgb_correction/summaries/` are compact aggregate
summaries. They are not the complete per-image result tree. Refer to the
matching protocol for the population, candidate definition, clipping status,
confidence interval, and interpretation guardrail.
