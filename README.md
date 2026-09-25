# Residual FDPL

Reference implementation for **Residual Frequency-Domain Perceptual Loss
(Residual FDPL)**, a frequency-domain training objective for real-world image
denoising.

The canonical carrier used in the paper is a compact 3-RCAB model:

- 4 residual-in-residual dense blocks (RRDB)
- 3 lightweight residual channel attention blocks (RCAB)
- 197,819 parameters
- Stage 0 Gaussian pretraining with Original FDPL
- Stage 1 and Stage 2 SIDD fine-tuning with Residual FDPL
- target FDPL share `alpha = 0.30`

The canonical training and evaluation paths are present in a small, testable
codebase without changing the locked experiment protocol. The exact revision
prepared for review is tagged `v3.1-review`. Optional checkpoint and
evaluation-data assets are attached to the corresponding GitHub Release rather
than committed to ordinary Git history.

## Status

- Core model API: available
- Residual FDPL and Original FDPL loss APIs: available
- PSNR, SSIM, device, and reproducibility utilities: available
- SIDD paired-image and HDF5 datasets: available
- Synthetic CPU smoke test: available
- HDF5 build and inference CLI: available
- Canonical Stage 0 Gaussian trainer: available
- Canonical Stage 1 -> Stage 2 SIDD trainer: available
- EMA, fixed weight maps, calibration, checkpoints, and history JSON: available
- CPU-only regression tests: available
- Code license: Apache-2.0
- Core checkpoints: available in the public `v3.1-review` Release
- 1,024-pair SIDD+ validation bundle: available in the same Release under the
  upstream research/education notice
- Per-image result redistribution: retained privately for this review

## Installation

Python 3.10 or newer is required. Python 3.11 is the validated baseline.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Linux CUDA users should install the PyTorch build matching their CUDA driver
before installing this package. macOS users can use the standard PyPI wheels;
the validated local baseline used the MPS backend.

## Reviewer Quick Start

The fastest path is to download all ten assets from the `v3.1-review`
Release: the six checkpoints, the three SHA-256 manifests, and
`sidd_quick_eval_256.zip`.

```bash
pip install -r requirements.txt

shasum -a 256 -c CHECKPOINT_SHA256SUMS
shasum -a 256 -c SIDD_QUICK_EVAL_ZIP_SHA256SUMS
unzip sidd_quick_eval_256.zip -d /path/to/siddplus_valid

CHECKPOINT_DIR=/path/to/downloaded/checkpoints \
NOISY_DIR=/path/to/siddplus_valid/siddplus_valid_noisy_srgb \
GT_DIR=/path/to/siddplus_valid/siddplus_valid_gt_srgb \
OUTPUT_DIR=runs/reproduce_table3 \
bash scripts/reproduce_table3.sh
```

This evaluates each released checkpoint under the protocol required by its
Table 3 row and writes raw per-image JSON, `table3_reproduced.json`, and
`table3_reproduced.csv`. The reported quantity is the same-seed paired PSNR
gain; the script also reports the paired SSIM delta and PSNR normal
approximation. Add `--limit 1` for a fast CLI check on the first pair.
The archive carries
[the SIDD+ data notice](docs/SIDD_PLUS_LICENSE_AND_NOTICE.md); it is for open
research and educational use and is not relicensed under Apache-2.0.

For a single checkpoint and both inference protocols, use:

```bash
CHECKPOINT=/path/to/model.pth \
NOISY_DIR=/path/to/siddplus_valid_noisy_srgb \
GT_DIR=/path/to/siddplus_valid_gt_srgb \
OUTPUT_DIR=runs/single_checkpoint \
bash scripts/run_eval_all.sh
```

### Table 3 checkpoint map

Each Table 3 row is reproduced by evaluating the two matched checkpoints for
the stated seed under the stated protocol. A checkpoint trained for a seed is
used for both protocols when that protocol is available.

| Table 3 row | Protocol | Seed | Matched no-RFDPL asset | Residual FDPL asset | Reported gain |
|---|---|---|---|---|---|
| 1 | sliding `128-tile / 32-overlap / stride-96` | 42 | `rcab3_seed42_nofdpl_final.pth` | `rcab3_seed42_fdpl_final.pth` | `+0.6475` dB |
| 2 | sliding `128-tile / 32-overlap / stride-96` | 43 | `rcab3_seed43_nofdpl_final.pth` | `rcab3_seed43_fdpl_final.pth` | `+0.5985` dB |
| 3 | sliding `128-tile / 32-overlap / stride-96` | 44 | `rcab3_seed44_nofdpl_final.pth` | `rcab3_seed44_fdpl_final.pth` | `+0.4101` dB |
| 4 | direct `256` | 42 | `rcab3_seed42_nofdpl_final.pth` | `rcab3_seed42_fdpl_final.pth` | `+0.5995` dB |
| 5 | direct `256` | 43 | `rcab3_seed43_nofdpl_final.pth` | `rcab3_seed43_fdpl_final.pth` | `+0.6017` dB |

The seed-44 direct model was not trained, so Table 3 has no direct row for
seed 44. Table 3 gains are same-seed paired differences computed from the
per-image values; evaluate both assets for a row rather than comparing
absolute PSNR across seeds.

## Quick Smoke Test

The smoke test does not download data or weights. It instantiates the 3-RCAB
model, checks the parameter count, runs a forward and backward pass, and verifies
that the loss is finite.

```bash
python scripts/smoke_test.py
```

## Data Preparation

The complete SIDD training dataset is not bundled. The Release provides the
exact 1,024-pair validation subset used for evaluation, while the
data-preparation tooling below supports user-supplied training data.

```bash
python scripts/build_sidd_hdf5.py \
  --noisy-dir /path/to/input_crops \
  --gt-dir /path/to/gt_crops \
  --output /path/to/sidd_crops_256_bilinear.h5 \
  --crop-size 256 \
  --interpolation bilinear
```

This 512 -> 256 bilinear cache is the main v3.1 training chain. The trainer
then resizes patches to 64 or 256, matching the archived protocol.

See [docs/DATA.md](docs/DATA.md) for the expected SIDD layout and
[docs/SIDD_QUICK_EVAL.md](docs/SIDD_QUICK_EVAL.md) for the 1,024-pair
manifest and verification command.

## Evaluation

```bash
python evaluate.py \
  --checkpoint /path/to/model.pth \
  --noisy-dir /path/to/noisy \
  --gt-dir /path/to/gt \
  --protocol sliding
```

The evaluator accepts common Lightning-style checkpoint dictionaries with a
`model_state_dict` key and plain PyTorch state dictionaries.

## Training

The public entry point is `train.py`. Paths are always supplied by the user;
there are no machine-specific defaults.

### Stage 0: Urban100 Gaussian Pretraining

```bash
python train.py stage0 \
  --data-root /path/to/Urban100 \
  --output-dir runs/stage0_seed42 \
  --config configs/stage0_gaussian.json
```

This stage uses Original FDPL with the locked 50-epoch, 64 x 64 Gaussian
protocol and writes `stage0.pth`, `stage0_history.json`, and
`stage0_results.json`.

### Stage 1 -> Stage 2: SIDD Fine-Tuning

```bash
python train.py sidd \
  --h5 /path/to/sidd_crops_256_bilinear.h5 \
  --pretrained runs/stage0_seed42/stage0.pth \
  --output-dir runs/3rcab_seed42 \
  --stage1-config configs/sidd_stage1_64.json \
  --stage2-config configs/sidd_stage2_256.json
```

The two stages run in one process so the locked random state, EMA state, and
calibrated loss scaling are preserved. The run writes final, best, and EMA
checkpoints plus `loss_history.json` and `results.json`.

Use `--no-fdpl` for the matched controlled baseline:

```bash
python train.py sidd \
  --h5 /path/to/sidd_crops_256_bilinear.h5 \
  --pretrained runs/stage0_seed42/stage0.pth \
  --output-dir runs/3rcab_seed42_nofdpl \
  --stage1-config configs/sidd_stage1_64.json \
  --stage2-config configs/sidd_stage2_256.json \
  --no-fdpl
```

`--smoke N` limits the data and is intended only for installation checks. It
does not reproduce paper numbers.

VGG19 ImageNet weights are downloaded by torchvision on first use. For an
offline machine, pass a local torchvision VGG19 checkpoint to either training
command:

```bash
python train.py sidd ... --vgg-weights /path/to/vgg19-dcbb9e9d.pth
```

The archived training runs seeded all libraries but did not enable cuDNN
deterministic mode. The public trainer follows that default. Add
`--deterministic` when bitwise-repeatability is preferred over matching the
archived runtime behavior. Stage 0 checkpoints must match all model keys unless
`--allow-partial-pretrained` is explicitly supplied.

The optional `crop512` chain expects an original-resolution HDF5 cache and
random-crops it during training:

```bash
python train.py sidd \
  --h5 /path/to/sidd_crops_512.h5 \
  --data-chain crop512 \
  --pretrained runs/stage0_seed42/stage0.pth \
  --output-dir runs/3rcab_seed42_crop512
```

That variant is a clean-chain option, not the archived main result protocol.

## Repository Layout

```text
.
|-- train.py                # Canonical Stage 0 and SIDD training entry point
|-- evaluate.py             # Direct and sliding checkpoint evaluation
|-- configs/                # Locked protocol values
|-- docs/                   # Data, model, release, and reproducibility notes
|-- requirements.txt        # Thin compatibility entry point for dependencies
|-- scripts/                # One-click training/evaluation and data tools
|-- src/cvfdpl/
|   |-- data/               # SIDD and HDF5 datasets
|   |-- losses/             # Original FDPL and Residual FDPL
|   |-- metrics/            # PSNR and SSIM
|   |-- models/             # 3-RCAB carrier
|   |-- training/           # Trainers, EMA, weight maps, and configuration
|   `-- utils/              # Device and reproducibility helpers
|-- tests/                  # CPU-only unit and smoke tests
`-- results/                # Public placeholder; paper JSON stays private
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for ownership boundaries and
the public-file policy. Configuration JSON is validated strictly, so unknown
or internally inconsistent fields fail before training starts.

## Reproducibility

The canonical values currently reflected in the package are:

- direct, sliding, and Stage-0 protocols separated explicitly
- final checkpoints used for reported results
- seed 42, 43, and 44 sliding runs
- seed 42 and 43 direct runs
- sample-standard-deviation uncertainty for cross-seed summaries

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the migration
checklist and the locked protocol constants.

## Citation

Repository metadata is provided in [CITATION.cff](CITATION.cff). The paper
title, venue metadata, and DOI will be added after publication.

## License

Original source code and the six author-created 3-RCAB checkpoint files
distributed in this repository's review Release are licensed under the Apache
License 2.0, to the extent permitted by applicable law. See [LICENSE](LICENSE)
and [NOTICE](NOTICE). Third-party dependencies, external pretrained weights,
datasets, and architectural references are listed separately in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and remain under their own
terms.

The optional `sidd_quick_eval_256.zip` archive is third-party research data.
It is not Apache-2.0. Its source, upstream notice, redistribution conditions,
and citation are recorded in
[docs/SIDD_PLUS_LICENSE_AND_NOTICE.md](docs/SIDD_PLUS_LICENSE_AND_NOTICE.md).

## Review Availability

- [Code availability statement](docs/CODE_AVAILABILITY.md)
- [Paper-to-code and result mapping](docs/PAPER_ARTIFACTS.md)
- [Checkpoint SHA-256 manifest](docs/CHECKPOINT_SHA256SUMS)
- [SIDD quick-evaluation manifest](docs/SIDD_QUICK_EVAL.md)
- [SIDD+ data license and notice](docs/SIDD_PLUS_LICENSE_AND_NOTICE.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)
- [Release plan](docs/RELEASE_PLAN.md)

The review tag `v3.1-review` contains the canonical source code, locked
configurations, the Table 3 reproduction script, and the public checkpoint and
evaluation-data manifests. Checkpoint binaries and the 1,024-pair evaluation
archive are distributed in the Release; datasets are kept outside normal Git
history.

## Security and Assets

Do not commit credentials, private logs, datasets, checkpoints, HDF5 caches, or
paper material. See [SECURITY.md](SECURITY.md) and
[docs/MODEL_ZOO.md](docs/MODEL_ZOO.md). The prepared private asset manifest is
documented in [assets/README.md](assets/README.md).
