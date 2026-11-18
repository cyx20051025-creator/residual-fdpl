# Assets

This directory is reserved for the public download manifest. Checkpoint and
SIDD+ evaluation downloads point to the `v3.1-review` GitHub Release.

The `rgb_correction/` subdirectory documents the separate asset boundary for
the residual RGB-correction study. It currently commits no images,
checkpoints, feature caches, or fitted third-party weights.

No dataset, checkpoint, HDF5 cache, or generated figure is committed to the
public source repository. The `v3.1-review` Release contains six final
checkpoints, three checksum manifests, and `sidd_quick_eval_256.zip`. Full
datasets, derived training caches, historical per-image JSON, and complete
training histories remain outside normal Git history.

The authoritative asset inventory is maintained in
[`docs/MODEL_ZOO.md`](../docs/MODEL_ZOO.md), and the reproduction commands are
in [`README.md`](../README.md).

The quick-evaluation bundle contains 1,024 paired SIDD+ validation images and
expands to approximately 190 MB. A public
`docs/SIDD_QUICK_EVAL_SHA256SUMS` manifest records the exact expected files.
The archive includes its upstream source and full data notice; see
`docs/SIDD_PLUS_LICENSE_AND_NOTICE.md`.
