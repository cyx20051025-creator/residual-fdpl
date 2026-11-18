# RGB offset pilot on frozen A denoisers

## Provenance and scope

This package reproduces the 15 November 2026 three-seed pilot that recorded a
+1.024220 dB CC-noise gain with Ridge. Ridge and MLP were fit locally on CPU
using cached source features. The external image inference and metric
evaluation used a single RTX 3060 12 GB. This is the 117→32→3 MLP pilot.
The later two-seed 117→128→64→3 capacity experiment is separate.

## Source fitting

- Seeds 56, 57 and 58 refer to the three existing frozen A endpoints.
- Each endpoint has 6,000 source crops and 160 scene identifiers in its cache.
  A fixed random permutation (seed 20261115) assigns 128 scenes / 4,812
  crops to fitting and 32 scenes / 1,188 crops to predictor holdout.
  This split controls predictor fitting. It makes no further claim about
  exclusion from the history of the frozen A checkpoint.
- Both regressors use identical 117-dimensional features extracted from the
  noisy input and frozen output, source-only feature standardization and
  feature clipping to [-10, 10]. The fitting target is the ground-truth
  minus output RGB channel mean, orthogonally projected to the zero-luma
  Rec.601 subspace with the vector (0.299, 0.587, 0.114).
- Ridge uses a closed-form solution with regularization n × 0.001 on the
  standardized feature coefficients, excluding the intercept. Coefficients
  are the ridge array in preprocessing_{seed}.npz.
- The MLP has 32 tanh hidden units and three outputs. It was fit by Adam for
  120 epochs at learning rate 0.001, using batches of approximately 128 and a
  source-fit RMS target normalization. The training script fixes the split,
  model sizes and schedules. GPU evaluation uses the bundled frozen weights.
- Source-holdout PSNR is computed before image clipping via the exact MSE
  change from adding a spatially constant offset. External PSNR is computed
  after rendering and clipping images to [0,1].

## External evaluation

- The manifest and five HDF5 files fix the samples, evaluation-unit IDs and
  file hashes. The evaluator verifies the selected input hashes.
- The reference for every reported gain is the same frozen A prediction
  clipped to [0,1]. The evaluator checks its PSNR against the archived
  original-rule records before computing paired differences.
- Original rule uses its archived per-image offsets as fixed control inputs.
  The six original-rule JSONs preserve those offsets and the original base
  PSNR. Ridge and MLP each apply their full fitted zero-luma prediction,
  with no extra shrinkage or gate. The original rule retains its existing
  calibration. Thus the Ridge–MLP comparison isolates the predictor family
  within this setup; the comparison with the original rule compares
  complete correction pipelines.
- The Oracle uses the clean reference to retrieve a diagnostic offset.
  Ridge and MLP use clean references only for fitting labels on the source
  and for scoring on the external datasets.
- Evaluation uses RGB PSNR, SSIM, LPIPS-Alex and PIQ DISTS. Compute per-image
  paired differences, average within the manifest's primary unit, then
  give evaluation units equal weight. The report averages seeds equally.
  Its 10,000-draw paired unit bootstrap conditions on the observed
  checkpoints and does not represent training-seed uncertainty.
- The five external sets are fixed diagnostic sets already used in this
  project. No refitting was carried out on those sets in this pilot.

## Results

| Population | Original rule | Ridge | MLP | Oracle |
|---|---:|---:|---:|---:|
| CC-noise | +0.551 | +1.024 | +0.671 | +1.198 |
| PolyU | +0.352 | +0.506 | +0.402 | +0.657 |
| NIND reserve 1 | +0.040 | −0.014 | +0.019 | +0.289 |
| NIND reserve 2 | +0.066 | +0.054 | +0.036 | +0.228 |
| SIDD benchmark | −0.031 | −0.232 | −0.137 | +0.587 |

Entries are three-seed mean paired PSNR changes in dB. For Ridge on
CC-noise, the per-seed gains were +1.130052, +0.990452 and +0.952156 dB.
The 15-image paired bootstrap interval was [+0.704201, +1.365199] dB.
The four-metric and alignment results are in external_summary.json.
