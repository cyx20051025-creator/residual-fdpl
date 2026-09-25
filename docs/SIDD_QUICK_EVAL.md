# SIDD Quick-Evaluation Manifest

The tables and figures in the v3.1 paper use the 1,024 paired SIDD+
validation images described here. The exact evaluation archive
`sidd_quick_eval_256.zip` is attached to the `v3.1-review.3` GitHub Release
under the upstream open-research/education notice. Git history stores only the
public manifest, not the images.

## Expected Layout

```text
valid/
|-- siddplus_valid_noisy_srgb/
|   |-- 0000.PNG
|   |-- 0001.PNG
|   `-- ...
`-- siddplus_valid_gt_srgb/
    |-- 0000.PNG
    |-- 0001.PNG
    `-- ...
```

The two directories contain matching names from `0000.PNG` through
`1023.PNG`, giving 1,024 pairs. Every file used in the reported evaluation is
an RGB PNG of size 256 x 256. The noisy and clean files must have the same
relative file name.

## Verification

Download and extract the Release asset:

```bash
shasum -a 256 -c SIDD_QUICK_EVAL_ZIP_SHA256SUMS
unzip sidd_quick_eval_256.zip -d /path/to/siddplus_valid
cd /path/to/siddplus_valid
```

Then verify every extracted image:

```bash
shasum -a 256 -c /path/to/residual-fdpl/docs/SIDD_QUICK_EVAL_SHA256SUMS
```

The manifest digest is:

```text
340801d0c0acda35aa04e86c80d97523d58c2dde473e0418ae00f86b3bc59cb5
```

After both directories verify, use `scripts/reproduce_table3.sh` with the
checkpoint directory and the verified image paths.

## Distribution Boundary

The upstream SIDD data terms remain in force and the stricter NTIRE SIDD+
challenge terms apply to this evaluation subset. The data are provided only
for open research and educational purposes. Redistribution must retain
[the complete notice](SIDD_PLUS_LICENSE_AND_NOTICE.md); the images are not
relicensed under Apache-2.0. Do not add these PNG files, derived HDF5 caches,
or the evaluation archive to normal Git history.
