#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-test2_full}"
BASE_LIMIT="${BASE_LIMIT:-10000}"
RL_LIMIT_PER_TARGET="${RL_LIMIT_PER_TARGET:-10000}"
TRAINING_LIMIT="${TRAINING_LIMIT:-10000}"
DISPROT_LIMIT="${DISPROT_LIMIT:-0}"
CATH_LIMIT="${CATH_LIMIT:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SEED="${SEED:-33402}"
UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}"
MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/idiom_mpl}"
export MPLCONFIGDIR

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

mkdir -p data/processed "results/test2_feature_probe/${RUN_NAME}" logs

echo "== Test 2 RandomForest shallow-feature probe =="
echo "run_name=${RUN_NAME}"
echo "base_limit=${BASE_LIMIT}"
echo "rl_limit_per_target=${RL_LIMIT_PER_TARGET}"
echo "training_limit=${TRAINING_LIMIT}"
echo "disprot_limit=${DISPROT_LIMIT} (0 means all)"
echo "cath_limit=${CATH_LIMIT} (0 means all)"
echo "batch_size=${BATCH_SIZE}"

echo "== Downloading/confirming Test 2 source FASTAs in Modal volume =="
UV_CACHE_DIR="$UV_CACHE_DIR" uv run --no-sync modal run tools/modal/test2_score_pool.py --action download

echo "== Assembling and scoring Test 2 pool on Modal =="
UV_CACHE_DIR="$UV_CACHE_DIR" uv run --no-sync modal run tools/modal/test2_score_pool.py \
  --action score \
  --run-name "$RUN_NAME" \
  --base-limit "$BASE_LIMIT" \
  --rl-limit-per-target "$RL_LIMIT_PER_TARGET" \
  --training-limit "$TRAINING_LIMIT" \
  --disprot-limit "$DISPROT_LIMIT" \
  --cath-limit "$CATH_LIMIT" \
  --batch-size "$BATCH_SIZE" \
  --seed "$SEED"

echo "== Downloading scored pool from Modal volume =="
UV_CACHE_DIR="$UV_CACHE_DIR" uv run --no-sync modal volume get idiom-audit-data "processed/${RUN_NAME}_pool_protgps_scores.csv" data/processed --force

echo "== Featurizing scored pool =="
UV_CACHE_DIR="$UV_CACHE_DIR" uv run --no-sync idiom-audit featurize \
  --input "data/processed/${RUN_NAME}_pool_protgps_scores.csv" \
  --output "data/processed/${RUN_NAME}_features.csv"

echo "== Running RandomForest feature probe =="
UV_CACHE_DIR="$UV_CACHE_DIR" uv run --no-sync idiom-audit test2 \
  --features "data/processed/${RUN_NAME}_features.csv" \
  --scores "data/processed/${RUN_NAME}_pool_protgps_scores.csv" \
  --out-dir "results/test2_feature_probe/${RUN_NAME}" \
  --seed "$SEED"

echo "== Done =="
echo "Scores:"
echo "  data/processed/${RUN_NAME}_pool_protgps_scores.csv"
echo "Features:"
echo "  data/processed/${RUN_NAME}_features.csv"
echo "Results:"
echo "  results/test2_feature_probe/${RUN_NAME}"
