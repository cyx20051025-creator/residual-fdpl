# RGB-Correction Reproducibility

## Verification Levels

| Level | Scope | Public inputs required |
|---|---|---|
| A: maintained code | Import, operators, Ridge fit, split validation, selection policy, CLI, source build | This Git repository |
| B: aggregate replay | Re-read compact CSV/Markdown summaries and inspect frozen split/policy decisions | This Git repository |
| C: historical analysis replay | Re-run fitting, scoring, bootstrap, and plotting scripts | Upstream images/features, frozen denoisers, and fitted regressors obtained under their own terms |
| D: end-to-end image rendering | Regenerate PSNR, SSIM, LPIPS, and DISTS from denoised outputs | All upstream images, checkpoints, model weights, and metric dependencies |

This release fully supports Level A. It supports Level B for the compact files
under `results/rgb_correction/`. Levels C and D are auditable from the archived
scripts and protocols, but are not a one-command public replay because the
required images, checkpoints, and feature caches are not redistributed.

## Locked Geometry

The correction acts on one spatially shared RGB offset per image. The public
implementation uses Rec.601 luma weights
`(0.299, 0.587, 0.114)` and projects every offset `d` to

`d <- d - (w^T d) w / (w^T w)`.

For a current mean error `e = reference - output` and an allowed offset `d`, the
exact pre-clipping channel-averaged MSE reduction is

`(2 e^T d - ||d||^2) / 3`.

This identity is implemented by `rgb_correction.ops.mse_reduction` and checked
against direct before/after MSE in the unit tests. The corresponding PSNR change
is `10 log10(MSE_before / MSE_after)`. It is not a post-clipping PSNR guarantee.

## Endpoint Calibration

`fit_zero_luma_ridge` fits one multi-output linear map from image statistics to
Ridge coefficients. Features are standardized with scene-balanced weights when
scene labels are supplied, so a scene with more crops does not receive more
total fitting weight. `ridge` is the penalty relative to the weighted fitting
row count.

The calibration workflow used by the paper is:

1. Partition calibration and evaluation units at the declared unit of
   independence.
2. Use clean references only to construct source or target offsets, or to score
   calibration candidates.
3. Compare source reuse, scaled source, fixed target, capped Ridge, and uncapped
   Ridge under identical held-out units.
4. Apply the predeclared risk policy and select one action, or abstain.
5. Freeze that selection before applying it to disjoint evaluation images.

The exact candidate thresholds are represented by `RiskPolicy`. They are not
retuned in the maintained API. `validate_scene_split` rejects overlap and can
enforce a complete partition.

## Public Commands

```bash
python -m pip install -e ".[dev]"
ruff check src/rgb_correction tests/test_rgb_correction_*.py
pytest tests/test_rgb_correction_*.py
PYTHONPATH=src python -m rgb_correction demo
```

To check a selection payload:

```bash
rgb-correction select --input candidates.json
```

The CLI JSON is an interface check, not a reproduction of paper numbers.
Historical result files remain the numerical record.

## Historical Replay

The scripts under `scripts/rgb_correction/` are preserved for inspection and
scholarly replay. Their paths were written for extracted experiment packages
and some expect a working directory, feature cache, or result directory that is
not present in ordinary Git. Users should read
`scripts/rgb_correction/pipeline/PROTOCOL.md` and the matching protocol under
`protocols/rgb_correction/` before running a script.

The repository deliberately excludes:

- source, calibration, and evaluation images;
- SIDD/SIDD+, PolyU, CC-noise, NIND, RENOIR, and DND datasets;
- denoiser checkpoints and third-party pretrained weights;
- feature caches and per-image tensors;
- cloud launch scripts and credentials; and
- large intermediate result trees.

## Reporting Rules

Use the calibration or evaluation unit stated by the source protocol. Do not
treat crops as independent scenes when the split was made over scenes. Do not
compare a post-clipping rendered metric with a pre-clipping diagnostic as if
they were the same quantity. Repeated splits over the same population provide
stability evidence, not additional independent populations. Exact values in
the aggregate summaries should be read together with their protocol guards and
population description.
