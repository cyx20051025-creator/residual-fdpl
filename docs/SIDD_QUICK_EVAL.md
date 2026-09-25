# SIDD Quick-Evaluation Manifest

The tables and figures in the v3.1 paper use the 1,024 paired SIDD+
validation images described here. This repository does not redistribute the
image files or a derived archive. The public manifest contains only file names
and SHA-256 digests so a locally obtained copy can be verified before running
the released checkpoints.

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

From the directory that contains `siddplus_valid_noisy_srgb/` and
`siddplus_valid_gt_srgb/`, run:

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

The upstream SIDD data terms remain in force. Do not add these PNG files,
derived HDF5 caches, or a repacked quick-evaluation archive to normal Git
history. If the authors later confirm that redistribution is permitted, the
same manifest can be attached to a GitHub Release as a separate archive.
