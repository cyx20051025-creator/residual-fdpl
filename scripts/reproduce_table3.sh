#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  PYTHON_BIN="python"
fi

ARGS=()
if [[ -n "${CHECKPOINT_DIR:-}" ]]; then
  ARGS+=(--checkpoint-dir "$CHECKPOINT_DIR")
fi
if [[ -n "${NOISY_DIR:-}" ]]; then
  ARGS+=(--noisy-dir "$NOISY_DIR")
fi
if [[ -n "${GT_DIR:-}" ]]; then
  ARGS+=(--gt-dir "$GT_DIR")
fi
if [[ -n "${OUTPUT_DIR:-}" ]]; then
  ARGS+=(--output-dir "$OUTPUT_DIR")
fi

if ((${#ARGS[@]})); then
  exec "$PYTHON_BIN" "$ROOT/scripts/reproduce_table3.py" "${ARGS[@]}" "$@"
else
  exec "$PYTHON_BIN" "$ROOT/scripts/reproduce_table3.py" "$@"
fi
