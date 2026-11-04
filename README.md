# Residual FDPL

Reference implementation for **Residual Frequency Domain Perception Loss
(Residual FDPL)**, a frequency-domain training objective for real-world image
denoising. This repository accompanies the manuscript:

> Yixuan Chen and Yufeng Qin. *Residual Frequency Domain Perception Loss for
> Real-World Image Denoising with Adaptive Spectral Supervision.* The Visual
> Computer, 2026 (accepted for publication).

Original FDPL was introduced in our earlier work:

> Yixuan Chen and Yufeng Qin. *A Lightweight Dual-Branch CNN with Frequency
> Domain Perception Loss for Image Denoising.* Computers, Materials &
> Continua, 2026, 89(2), 67. <https://doi.org/10.32604/cmc.2026.084385>

The CMC study used a lightweight dual-branch carrier and Gaussian benchmarks.
This repository implements the later TVC study, whose main carrier is a compact
3-RCAB model:

- 4 residual-in-residual dense blocks (RRDB)
- 3 lightweight residual channel attention blocks (RCAB)
- 197,819 parameters
- Stage 0 Gaussian pretraining with Original FDPL
- Stage 1 and Stage 2 SIDD fine-tuning with Residual FDPL
- target FDPL share `alpha = 0.30`

The training and evaluation paths are present in a small, testable codebase.
Use source tag `v3.1-review-8` for executable code. Checkpoint and
evaluation-data assets remain attached to the earlier `v3.1-review` GitHub
Release rather than committed to ordinary Git history.

The repository reproduces the TVC checkpoint-evaluation and Table 3
comparisons. It does not include the original CMC source tree, the CMC
Gaussian benchmark pipeline, the historical dual-branch checkpoints, or a
bitwise implementation-equivalence claim for the published CMC model.

## Status

The public repository provides:

- the 3-RCAB model, Original FDPL, and Residual FDPL implementations
- Stage 0 and Stage 1 -> Stage 2 training entry points
- direct and sliding checkpoint evaluators
- CPU regression tests and a data-free smoke test
- six final checkpoints and a 1,024-pair SIDD+ evaluation bundle in the
  `v3.1-review` Release
- a script that evaluates all six Table 3 comparisons and writes per-image
  results plus JSON and CSV summaries

Historical per-image JSON files, full training histories, and the original
CMC experiments are not included. The released checkpoints and evaluation
bundle allow users to regenerate the TVC per-image results. See
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the public scope and
verification boundaries.

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

## Reproduction Quick Start

Use source tag `v3.1-review-8` for the code and the `v3.1-review` Release for
the assets. Do not use the automatically attached source archive from the
`v3.1-review` Release page; that older snapshot does not contain the six-row
Table 3 reproduction script.

```bash
git clone --depth 1 --branch v3.1-review-8 \
  https://github.com/cyx20051025-creator/residual-fdpl.git
cd residual-fdpl
python -m pip install -r requirements.txt
```

Download all ten assets from
<https://github.com/cyx20051025-creator/residual-fdpl/releases/tag/v3.1-review>
into a directory such as `release_assets`: the six checkpoints, three SHA-256
manifests, and `sidd_quick_eval_256.zip`. Then verify and extract them:

```bash
mkdir -p release_assets
cd release_assets
shasum -a 256 -c CHECKPOINT_SHA256SUMS
shasum -a 256 -c SIDD_QUICK_EVAL_ZIP_SHA256SUMS
unzip sidd_quick_eval_256.zip -d /path/to/siddplus_valid
cd /path/to/siddplus_valid
shasum -a 256 -c /path/to/residual-fdpl/docs/SIDD_QUICK_EVAL_SHA256SUMS
cd /path/to/residual-fdpl

CHECKPOINT_DIR=/path/to/release_assets \
NOISY_DIR=/path/to/siddplus_valid/siddplus_valid_noisy_srgb \
GT_DIR=/path/to/siddplus_valid/siddplus_valid_gt_srgb \
OUTPUT_DIR=runs/reproduce_table3 \
bash scripts/reproduce_table3.sh
```

This evaluates each released checkpoint under the protocol required by its
Table 3 row and writes raw per-image JSON, `table3_reproduced.json`, and
`table3_reproduced.csv`. The reported quantity is the same-seed paired PSNR
gain; the script also reports the paired SSIM delta and a two-sided Student-t
PSNR p-value with `df = n - 1` (`df = 1023` for the full validation set). Add
`--limit 1` for a fast CLI check on the first pair.
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
| 6 | direct `256` | 44 | `rcab3_seed44_nofdpl_final.pth` | `rcab3_seed44_fdpl_final.pth` | `+0.3955` dB |

The seed-44 direct row uses the same matched final checkpoints as the seed-44
sliding row. Table 3 gains are same-seed paired differences computed from
the per-image values; evaluate both assets for a row rather than comparing
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
`-- results/                # Public placeholder; historical JSON is not distributed
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for ownership boundaries and
the public-file policy. Configuration JSON is validated strictly, so unknown
or internally inconsistent fields fail before training starts.

## Reproducibility

The main v3.1 protocol values reflected in the package are:

- direct, sliding, and Stage-0 protocols separated explicitly
- final checkpoints used for reported results
- seeds 42, 43, and 44 under sliding inference
- seeds 42, 43, and 44 under direct inference
- sample-standard-deviation uncertainty for cross-seed summaries

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the verification
scope and protocol constants.

## Citation

Repository metadata is provided in [CITATION.cff](CITATION.cff). It identifies
the software separately from the TVC manuscript and the earlier CMC paper. Add
the final TVC DOI once it is assigned; do not reuse the CMC DOI for the TVC
study.

## License

Original source code and the six author-created 3-RCAB checkpoint files
distributed with the `v3.1-review` asset Release are licensed under the
Apache License 2.0, to the extent permitted by applicable law. See
[LICENSE](LICENSE) and [NOTICE](NOTICE). Third-party dependencies, external
pretrained weights, datasets, and architectural references are listed
separately in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and remain under
their own terms.

The optional `sidd_quick_eval_256.zip` archive is third-party research data.
It is not Apache-2.0. Its source, upstream notice, redistribution conditions,
and citation are recorded in
[docs/SIDD_PLUS_LICENSE_AND_NOTICE.md](docs/SIDD_PLUS_LICENSE_AND_NOTICE.md).

## Documentation and Availability

- [Code availability statement](docs/CODE_AVAILABILITY.md)
- [Paper-to-code and result mapping](docs/PAPER_ARTIFACTS.md)
- [Checkpoint SHA-256 manifest](docs/CHECKPOINT_SHA256SUMS)
- [SIDD quick-evaluation manifest](docs/SIDD_QUICK_EVAL.md)
- [SIDD+ data license and notice](docs/SIDD_PLUS_LICENSE_AND_NOTICE.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)
- [Release plan](docs/RELEASE_PLAN.md)

The source tag `v3.1-review-8` contains the source code, configurations, the
six-row Table 3 reproduction script, and the public checkpoint and
evaluation-data manifests. Checkpoint binaries and the 1,024-pair evaluation
archive remain attached to the `v3.1-review` Release; datasets are kept outside
normal Git history.

## Security and Assets

Do not commit credentials, private logs, datasets, checkpoints, HDF5 caches, or
paper material. See [SECURITY.md](SECURITY.md) and
[docs/MODEL_ZOO.md](docs/MODEL_ZOO.md). The public asset inventory is
documented in [assets/README.md](assets/README.md).
