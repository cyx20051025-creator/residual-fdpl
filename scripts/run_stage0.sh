#!/usr/bin/env bash
set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to the Urban100 directory}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/stage0_seed42}"

python train.py stage0 \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --config configs/stage0_gaussian.json \
  "$@"
