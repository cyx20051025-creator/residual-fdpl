# Residual RGB Correction

This directory documents the public software and evidence boundary for
*Residual Color Correction after Real-World Image Denoising: Transfer
Boundaries and Risk-Controlled Endpoint Calibration*.

The study asks three questions about a frozen denoiser:

1. How much of its remaining RGB channel-mean error is removable before image
   clipping?
2. Does a correction rule selected on one source population transfer to another
   population or to a different checkpoint?
3. With no clean reference at deployment, can held-out paired examples select a
   correction action that controls severe loss while retaining useful gain?

The maintained code is in `src/rgb_correction/`. It is separate from the earlier
Residual FDPL training code and does not retrain or modify a denoiser.

## Maintained API

| Module | Public responsibility |
|---|---|
| `rgb_correction.ops` | Rec.601 weights, zero-luma projection, exact pre-clipping MSE reduction, PSNR change, and clipped offset application |
| `rgb_correction.calibration` | Scene-balanced weights, standardized multi-output Ridge fitting, zero-luma prediction, and deterministic paired bootstrap intervals |
| `rgb_correction.selection` | Candidate scores, acceptance guards, simple-action-to-Ridge promotion, and abstention |
| `rgb_correction.validation` | Rejection of empty, overlapping, or incomplete calibration/evaluation scene splits |

The RGB module's own imports require only NumPy. It is covered by CPU tests
under `tests/test_rgb_correction_*.py`; installing the repository with the full
`cvfdpl` training dependencies still adds the project-level dependencies listed
in `pyproject.toml`.

## Quick Check

```bash
python -m pip install -e ".[dev]"
ruff check src/rgb_correction tests/test_rgb_correction_*.py
pytest tests/test_rgb_correction_*.py
rgb-correction demo
```

The `select` command expects a JSON object whose keys name candidate actions.
At least the required fields of `CandidateScore` should be provided:

```json
{
  "source": {
    "mean_psnr_delta": 0.20,
    "bootstrap_p05": 0.10,
    "severe_loss_rate": 0.05,
    "ssim_delta": 0.001,
    "lpips_delta": -0.001,
    "dists_delta": -0.001
  }
}
```

```bash
rgb-correction select --input candidates.json
```

The command reports the selected action, eligible actions, and the reason for
the decision. Metric deltas follow their natural direction: positive is better
for PSNR and SSIM; negative is better for LPIPS and DISTS.

## Public Research Files

- `protocols/rgb_correction/`: analysis questions and frozen split/policy rules.
- `scripts/rgb_correction/legacy/`: compact replay, aggregation, and plotting
  scripts copied from the paper package.
- `scripts/rgb_correction/pipeline/`: the archived RGB-offset pilot pipeline.
- `results/rgb_correction/summaries/`: compact Markdown and CSV summaries plus
  selected split and sensitivity JSON files.
- `results/rgb_correction/source_rules/`: the three fitted source-rule models,
  their SHA-256 manifest, and loading notes.
- `assets/rgb_correction/`: instructions for future release assets; no raw
  image, checkpoint, feature cache, or fitted third-party model is committed.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for what can be verified directly
from Git and what requires user-supplied upstream data.

## Interpretation Boundary

The package establishes reproducible operators and decision logic. It does not,
by itself, establish a new denoising result. Numerical claims depend on the
evaluation population, frozen checkpoint, split unit, clipping, and metric
reported alongside the result. The paper distinguishes:

- pre-clipping removable error from post-clipping image improvement;
- a better source-domain predictor from a correction rule that transfers;
- mean PSNR gain from severe-loss rate and perceptual-metric guards; and
- repeated splits over the same images from independent populations.

These distinctions should be retained in derivative uses of the code.
