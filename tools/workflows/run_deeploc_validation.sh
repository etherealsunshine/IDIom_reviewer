#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-deeploc_pilot_2k}"
PER_SOURCE_LIMIT="${PER_SOURCE_LIMIT:-2000}"
DISPROT_LIMIT="${DISPROT_LIMIT:-0}"
MODEL="${MODEL:-Fast}"
DEVICE="${DEVICE:-cuda}"
DEEPLOC_PACKAGE="${DEEPLOC_PACKAGE:-}"
INSTALL_SMOKE_ONLY="${INSTALL_SMOKE_ONLY:-0}"
UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}"
MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/idiom_mpl}"
export MPLCONFIGDIR

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

LOCAL_OUTPUT_PARENT="data/processed/deeploc_outputs"
LOCAL_OUTPUT_DIR="${LOCAL_OUTPUT_PARENT}/${RUN_NAME}"
RESULTS_DIR="results/deeploc_validation/${RUN_NAME}"
mkdir -p "$LOCAL_OUTPUT_PARENT" "$RESULTS_DIR" logs

echo "== DeepLoc independent validation =="
echo "run_name=${RUN_NAME}"
echo "per_source_limit=${PER_SOURCE_LIMIT}"
echo "disprot_limit=${DISPROT_LIMIT} (0 means all)"
echo "model=${MODEL}"
echo "device=${DEVICE}"
if [[ -n "$DEEPLOC_PACKAGE" ]]; then
  echo "deeploc_package=${DEEPLOC_PACKAGE}"
fi

if [[ -n "$DEEPLOC_PACKAGE" ]]; then
  if [[ ! -f "$DEEPLOC_PACKAGE" && ! -d "$DEEPLOC_PACKAGE" ]]; then
    echo "DeepLoc package path does not exist: $DEEPLOC_PACKAGE" >&2
    exit 1
  fi
  echo "== Uploading DeepLoc package to Modal volume =="
  UV_CACHE_DIR="$UV_CACHE_DIR" uv run --no-sync modal volume put idiom-deeploc-cache "$DEEPLOC_PACKAGE" packages/ --force
fi

if [[ "$INSTALL_SMOKE_ONLY" == "1" ]]; then
  echo "== Running DeepLoc install smoke on Modal =="
  UV_CACHE_DIR="$UV_CACHE_DIR" uv run --no-sync modal run tools/modal/deeploc_score.py --install-smoke
  exit 0
fi

echo "== Running DeepLoc on Modal =="
UV_CACHE_DIR="$UV_CACHE_DIR" uv run --no-sync modal run tools/modal/deeploc_score.py \
  --run-name "$RUN_NAME" \
  --per-source-limit "$PER_SOURCE_LIMIT" \
  --disprot-limit "$DISPROT_LIMIT" \
  --model "$MODEL" \
  --device "$DEVICE"

echo "== Downloading DeepLoc outputs =="
rm -rf "$LOCAL_OUTPUT_DIR"
mkdir -p "$LOCAL_OUTPUT_PARENT"
UV_CACHE_DIR="$UV_CACHE_DIR" uv run --no-sync modal volume get idiom-audit-data "deeploc_outputs/${RUN_NAME}" "$LOCAL_OUTPUT_PARENT" --force

echo "== Analyzing DeepLoc outputs =="
UV_CACHE_DIR="$UV_CACHE_DIR" uv run --no-sync idiom-audit deeploc-analyze \
  --results-dir "$LOCAL_OUTPUT_DIR" \
  --out-dir "$RESULTS_DIR"

echo "== Done =="
echo "DeepLoc outputs:"
echo "  $LOCAL_OUTPUT_DIR"
echo "Analysis:"
echo "  $RESULTS_DIR"
