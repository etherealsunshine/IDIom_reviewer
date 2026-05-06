#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-test1_pilot_1k}"
N_PER_TARGET="${N_PER_TARGET:-1000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SHARD_SIZE="${SHARD_SIZE:-512}"
SEED="${SEED:-33402}"
UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}"
MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/idiom_mpl}"
DISPROT_SCORES="${DISPROT_SCORES:-}"
export MPLCONFIGDIR

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

ORIGINALS="data/processed/${RUN_NAME}_originals_protgps_scores.csv"
SCRAMBLES="data/processed/${RUN_NAME}_scrambles_protgps_scores.csv"
BASELINES="data/processed/${RUN_NAME}_amended_baselines.csv"
BASELINES_SCORES="data/processed/${RUN_NAME}_amended_baselines_protgps_scores.csv"
MOTIF_SUMMARY="results/amended_baselines/${RUN_NAME}/rl_motif_summary.csv"
OUT_DIR="results/amended_baselines/${RUN_NAME}"

mkdir -p data/processed "$OUT_DIR" logs

if [[ ! -s "$ORIGINALS" ]]; then
  echo "Missing originals score CSV: $ORIGINALS" >&2
  exit 1
fi
if [[ ! -s "$SCRAMBLES" ]]; then
  echo "Missing scrambles score CSV: $SCRAMBLES" >&2
  exit 1
fi

echo "== Generating amended baselines =="
echo "run_name=${RUN_NAME}"
echo "n_per_target=${N_PER_TARGET}"
UV_CACHE_DIR="$UV_CACHE_DIR" uv run --no-sync idiom-audit amended-baselines \
  --originals "$ORIGINALS" \
  --output "$BASELINES" \
  --motif-summary-output "$MOTIF_SUMMARY" \
  --n-per-target "$N_PER_TARGET" \
  --seed "$SEED"

echo "== Scoring amended baselines on Modal =="
UV_CACHE_DIR="$UV_CACHE_DIR" uv run --no-sync modal run tools/modal/protgps_score_csv.py \
  --input-csv "$BASELINES" \
  --output-csv "$BASELINES_SCORES" \
  --limit 0 \
  --shard-size "$SHARD_SIZE" \
  --batch-size "$BATCH_SIZE"

echo "== Comparing conditions =="
SCORE_ARGS=("$ORIGINALS" "$SCRAMBLES" "$BASELINES_SCORES")
if [[ -n "$DISPROT_SCORES" ]]; then
  SCORE_ARGS+=("$DISPROT_SCORES")
fi

UV_CACHE_DIR="$UV_CACHE_DIR" uv run --no-sync idiom-audit amended-baseline-compare \
  --scores "${SCORE_ARGS[@]}" \
  --out-dir "$OUT_DIR"

echo "== Done =="
echo "Baselines:"
echo "  $BASELINES"
echo "Baseline scores:"
echo "  $BASELINES_SCORES"
echo "Motif summary:"
echo "  $MOTIF_SUMMARY"
echo "Comparison:"
echo "  ${OUT_DIR}/amended_baseline_summary.csv"
