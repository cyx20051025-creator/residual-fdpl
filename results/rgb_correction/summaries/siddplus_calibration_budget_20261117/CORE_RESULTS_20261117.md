# SIDD+ core calibration-budget results

Status: complete before the requested 17:00 Europe/London checkpoint.

## Protocol

- Population: 1,024 image-level SIDD+ units.
- Endpoints: fine-tuned NAFNet seeds 62 and 63.
- Source predictor: seed 56.
- Budgets: 16, 64, 128, 256 and 512 calibration images.
- Ten matched, fixed splits per budget.
- Frozen candidates, risk thresholds, `0.02` cap, split seed and bootstrap
  settings.
- Calibration and evaluation partitions are disjoint.
- Budget draws are matched but not forced to be nested, so the budget
  comparison is descriptive rather than a causal sample-size experiment.

The 16-image seed-62 file reproduces the prior selected actions and split
assignments, and its rendered PSNR agrees within `2e-4 dB`.

## Two-seed summary

| Budget | Selected ΔPSNR | Fixed target | Selector−fixed | ΔSSIM | ΔLPIPS | ΔDISTS | Positive splits | Severe-harm rate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | +0.5780 | +0.8249 | −0.2469 | +0.009576 | −0.002637 | +0.000271 | 20/20 | 12.01% |
| 64 | +0.5984 | +0.8540 | −0.2556 | +0.010442 | −0.002633 | +0.000394 | 20/20 | 11.49% |
| 128 | +0.7922 | +0.8572 | −0.0650 | +0.013694 | −0.004047 | +0.000019 | 20/20 | 12.34% |
| 256 | +0.8540 | +0.8685 | −0.0145 | +0.014627 | −0.004491 | −0.000189 | 20/20 | 12.15% |
| 512 | +1.0121 | +0.8910 | **+0.1211** | +0.018683 | −0.004049 | +0.000097 | 20/20 | 11.89% |

Per-endpoint selector-minus-fixed at 512 images:

- Seed 62: `+0.1034 dB` [`+0.0427`, `+0.1616`].
- Seed 63: `+0.1388 dB` [`+0.0829`, `+0.1842`].

## Interpretation

The earlier 16-image SIDD+ deficit is not an immutable endpoint failure.
Increasing paired calibration density recovers the selector monotonically
after the 64-image point. At 128 images the gap narrows to about `0.065 dB`;
at 256 it is within `0.015 dB`; and at 512 the selector exceeds the fixed
target in both endpoints with intervals excluding zero.

This result supports an endpoint-calibration boundary: 16 images were
insufficient for this SIDD+ endpoint. It does not establish 512 images as a
universal requirement, nor does it turn the repeated splits into independent
populations.

## Artifacts

- Canonical files: `core_seed62/` and `core_seed63/`.
- Machine-readable summaries: `budget_summary_core_seed62.csv` and
  `budget_summary_core_seed63.csv`.
- Reading reports: `RESULTS_core_seed62.md` and `RESULTS_core_seed63.md`.
- 4090 execution logs: `remote_logs_4090/`.

SHA-256 of the ten canonical JSON files:

```text
9e7e0164241c7a5cde586c980edb9071743895d292258ba76ed7a158f9c9125e  core_seed62/risk_selector_sidd_valid_budget128.json
4f3fe4a3b05b1665ef14bc25cd1df217a240282594434dd3589b4fbde386da27  core_seed62/risk_selector_sidd_valid_budget16.json
e4092c1a1f92fa2ce656025ee29abfb8a9140d54761e7705d75c6fc6672cd271  core_seed62/risk_selector_sidd_valid_budget256.json
59cc56e00229329c4438ae5907b2cff1982a203c06c351a9a8ddc41ab98c2871  core_seed62/risk_selector_sidd_valid_budget512.json
a3df52e7d805df4e87ed9f751496906bc3c1c824a2007be259613db2060fce9f  core_seed62/risk_selector_sidd_valid_budget64.json
bf28dc4c3aa4892fc605de13b52497eb811e5a226460c186c7b579859f47368f  core_seed63/risk_selector_sidd_valid_budget128.json
c14cf689083b9cd7070f4613baa3eade8085877702f101f7b73188f8d5eccd1d  core_seed63/risk_selector_sidd_valid_budget16.json
7ef08fe85714ee294f6dd6e87c4eb99df8da56771d711931bdd1fa6626530e76  core_seed63/risk_selector_sidd_valid_budget256.json
4569a0418c273a8cc1406ee6fbca7d46fe81ab65c6a5f0d1de9c0bc16be15e03  core_seed63/risk_selector_sidd_valid_budget512.json
b089938f5b81034207d1c8698b5c23e5c46c4ae3d5bd5973c6fb989baef4fcec  core_seed63/risk_selector_sidd_valid_budget64.json
```
