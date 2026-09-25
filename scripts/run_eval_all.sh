#!/usr/bin/env bash
set -euo pipefail

: "${CHECKPOINT:?Set CHECKPOINT to the model checkpoint}"
: "${NOISY_DIR:?Set NOISY_DIR to the validation noisy directory}"
: "${GT_DIR:?Set GT_DIR to the validation GT directory}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/evaluation}"

python evaluate.py \
  --checkpoint "$CHECKPOINT" \
  --noisy-dir "$NOISY_DIR" \
  --gt-dir "$GT_DIR" \
  --protocol direct \
  --output-json "$OUTPUT_DIR/direct256.json" \
  "$@"

python evaluate.py \
  --checkpoint "$CHECKPOINT" \
  --noisy-dir "$NOISY_DIR" \
  --gt-dir "$GT_DIR" \
  --protocol sliding \
  --tile-size 128 \
  --stride 96 \
  --output-json "$OUTPUT_DIR/sliding.json" \
  "$@"
