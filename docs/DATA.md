# Data Preparation

## Distributed Scope

The Git source tree does not redistribute SIDD, Urban100, BSDS500, BSD68,
CBSD68, derived HDF5 caches, or any other upstream dataset.

The `v3.1-review` Release provides only the exact 1,024-pair SIDD+
validation subset used for paper evaluation. Training data and other datasets
must be obtained separately by the user.

For the exact evaluation archive and its data notice, see
[SIDD_QUICK_EVAL.md](SIDD_QUICK_EVAL.md) and
[SIDD_PLUS_LICENSE_AND_NOTICE.md](SIDD_PLUS_LICENSE_AND_NOTICE.md). Verify the
local copy with `SIDD_QUICK_EVAL_SHA256SUMS`.

## SIDD Directory Layout

The expected SIDD training layout is:

```text
train_crop/
|-- input_crops/
`-- gt_crops/
```

The validation layout is:

```text
valid/
|-- siddplus_valid_noisy_srgb/
`-- siddplus_valid_gt_srgb/
```

## Build The Main HDF5 Cache

The archived v3.1 main chain first converts the paired crops to a 256 x 256
bilinear cache:

```bash
python scripts/build_sidd_hdf5.py \
  --noisy-dir /path/to/train_crop/input_crops \
  --gt-dir /path/to/train_crop/gt_crops \
  --output /path/to/sidd_crops_256_bilinear.h5 \
  --crop-size 256 \
  --interpolation bilinear
```

During training, the dataset resizes each cached image to 64 x 64 in Stage 1
and 256 x 256 in Stage 2, both with bilinear interpolation.

## Optional Clean-Chain Cache

An original-resolution cache can be used with `--data-chain crop512`:

```bash
python scripts/build_sidd_hdf5.py \
  --noisy-dir /path/to/train_crop/input_crops \
  --gt-dir /path/to/train_crop/gt_crops \
  --output /path/to/sidd_crops_512.h5 \
  --crop-size 512
```

The generated HDF5 contains two datasets:

- `noisy`: `N x H x W x 3`, `uint8`
- `gt`: `N x H x W x 3`, `uint8`

## License Status

The upstream dataset terms remain in force. The provided evaluation archive
must retain its complete notice and may be used only for open research and
educational purposes. Derived training caches are not redistributed.
