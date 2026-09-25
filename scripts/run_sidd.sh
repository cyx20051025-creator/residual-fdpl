#!/usr/bin/env bash
set -euo pipefail

: "${H5_PATH:?Set H5_PATH to sidd_crops_256_bilinear.h5}"
: "${PRETRAINED:?Set PRETRAINED to the Stage 0 checkpoint}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/3rcab_seed42}"

python train.py sidd \
  --h5 "$H5_PATH" \
  --pretrained "$PRETRAINED" \
  --output-dir "$OUTPUT_DIR" \
  --stage1-config configs/sidd_stage1_64.json \
  --stage2-config configs/sidd_stage2_256.json \
  "$@"
