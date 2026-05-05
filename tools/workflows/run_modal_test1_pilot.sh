#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-test1_pilot_1k}"
BASE_LIMIT="${BASE_LIMIT:-1000}"
RL_LIMIT_PER_TARGET="${RL_LIMIT_PER_TARGET:-1000}"
SCRAMBLES_PER_TYPE="${SCRAMBLES_PER_TYPE:-3}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SEED="${SEED:-33402}"
UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

mkdir -p data/processed "results/test1_scramble/${RUN_NAME}" logs

echo "== Test 1 Modal pilot =="
echo "run_name=${RUN_NAME}"
echo "base_limit=${BASE_LIMIT}"
echo "rl_limit_per_target=${RL_LIMIT_PER_TARGET}"
echo "scrambles_per_type=${SCRAMBLES_PER_TYPE} (full + block)"
echo "batch_size=${BATCH_SIZE}"

echo "== Checking Modal data volume layout =="
UV_CACHE_DIR="$UV_CACHE_DIR" uv run --no-sync modal run tools/modal/download_test1_fastas.py --action layout

echo "== Scoring originals and scrambles on Modal =="
UV_CACHE_DIR="$UV_CACHE_DIR" uv run --no-sync modal run tools/modal/test1_score.py \
  --run-name "$RUN_NAME" \
  --base-limit "$BASE_LIMIT" \
  --rl-limit-per-target "$RL_LIMIT_PER_TARGET" \
  --scrambles-per-type "$SCRAMBLES_PER_TYPE" \
  --batch-size "$BATCH_SIZE" \
  --seed "$SEED"

echo "== Downloading score CSVs from Modal volume =="
UV_CACHE_DIR="$UV_CACHE_DIR" uv run --no-sync modal volume get idiom-audit-data "processed/${RUN_NAME}_originals_protgps_scores.csv" data/processed --force
UV_CACHE_DIR="$UV_CACHE_DIR" uv run --no-sync modal volume get idiom-audit-data "processed/${RUN_NAME}_scrambles_protgps_scores.csv" data/processed --force

echo "== Running local Test 1 analysis =="
UV_CACHE_DIR="$UV_CACHE_DIR" uv run --no-sync idiom-audit test1 \
  --originals "data/processed/${RUN_NAME}_originals_protgps_scores.csv" \
  --scrambles "data/processed/${RUN_NAME}_scrambles_protgps_scores.csv" \
  --groups "data/processed/${RUN_NAME}_originals_protgps_scores.csv" "data/processed/${RUN_NAME}_scrambles_protgps_scores.csv" \
  --out-dir "results/test1_scramble/${RUN_NAME}"

echo "== Done =="
echo "Scores:"
echo "  data/processed/${RUN_NAME}_originals_protgps_scores.csv"
echo "  data/processed/${RUN_NAME}_scrambles_protgps_scores.csv"
echo "Results:"
echo "  results/test1_scramble/${RUN_NAME}"
