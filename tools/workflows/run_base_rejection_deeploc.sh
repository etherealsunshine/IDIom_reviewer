#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-base_rejection_10k_top2k}"
K="${K:-2000}"
MODEL="${MODEL:-Fast}"
DEVICE="${DEVICE:-cuda}"
UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}"
MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/idiom_mpl}"
export MPLCONFIGDIR

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

LOCAL_INPUT_DIR="data/processed/deeploc_custom_inputs/${RUN_NAME}"
LOCAL_OUTPUT_PARENT="data/processed/deeploc_outputs"
LOCAL_OUTPUT_DIR="${LOCAL_OUTPUT_PARENT}/${RUN_NAME}"
RESULTS_DIR="results/deeploc_validation/${RUN_NAME}"
mkdir -p "$LOCAL_OUTPUT_PARENT" "$RESULTS_DIR" logs

echo "== Base-IDiom rejection-sampling DeepLoc pilot =="
echo "run_name=${RUN_NAME}"
echo "k=${K}"
echo "model=${MODEL}"
echo "device=${DEVICE}"

echo "== Preparing selected base-model FASTAs =="
python3 tools/analysis/prepare_base_rejection_deeploc.py \
  --run-name "$RUN_NAME" \
  --k "$K"

echo "== Uploading selected FASTAs to Modal volume =="
UV_CACHE_DIR="$UV_CACHE_DIR" uv run --no-sync modal volume put \
  idiom-audit-data \
  "$LOCAL_INPUT_DIR" \
  "deeploc_custom_inputs/" \
  --force

echo "== Running DeepLoc on selected FASTAs =="
UV_CACHE_DIR="$UV_CACHE_DIR" uv run --no-sync modal run tools/modal/deeploc_score.py \
  --run-name "$RUN_NAME" \
  --model "$MODEL" \
  --device "$DEVICE" \
  --custom-inputs

echo "== Downloading DeepLoc outputs =="
rm -rf "$LOCAL_OUTPUT_DIR"
mkdir -p "$LOCAL_OUTPUT_PARENT"
UV_CACHE_DIR="$UV_CACHE_DIR" uv run --no-sync modal volume get \
  idiom-audit-data \
  "deeploc_outputs/${RUN_NAME}" \
  "$LOCAL_OUTPUT_PARENT" \
  --force

echo "== Analyzing rejection-sampling control =="
PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" python3 tools/analysis/analyze_base_rejection_deeploc.py \
  --results-dir "$LOCAL_OUTPUT_DIR" \
  --out-dir "$RESULTS_DIR"

echo "== Done =="
echo "Inputs:  $LOCAL_INPUT_DIR"
echo "Outputs: $LOCAL_OUTPUT_DIR"
echo "Results: $RESULTS_DIR"
