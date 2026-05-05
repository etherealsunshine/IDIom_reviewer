# Tools

Small non-package entry points live here. Keep importable analysis code in
`idiom_audit/`, and keep heavier one-off helpers grouped by what they touch.

```text
tools/
  data/        dataset/download-plan helpers
  modal/       Modal GPU scoring pilot and CSV scorer
  protgps/     ProtGPS scoring wrappers
  workflows/   shell workflows that compose the audit CLI
```

Use the workspace environment when running these:

```bash
UV_CACHE_DIR=.uv-cache uv run python tools/protgps/score_csv.py --help
UV_CACHE_DIR=.uv-cache uv run bash tools/workflows/prepare_audit_inputs.sh
```

Modal commands should be run from the project root:

```bash
UV_CACHE_DIR=.uv-cache uv run modal run tools/modal/download_test1_fastas.py --action inspect
UV_CACHE_DIR=.uv-cache uv run modal run tools/modal/download_test1_fastas.py --action download

UV_CACHE_DIR=.uv-cache uv run modal run tools/modal/protgps_score_csv.py \
  --input-csv results/smoke/cheap.csv \
  --output-csv results/smoke/cheap_modal_scores.csv \
  --limit 12 \
  --shard-size 12 \
  --batch-size 8
```

Unattended Test 1 pilot:

```bash
bash tools/workflows/run_modal_test1_pilot.sh 2>&1 | tee logs/test1_pilot_1k.log
```

The Test 1 pilot creates both full amino-acid scrambles and block scrambles.
Use `SCRAMBLES_PER_TYPE=2` or `3` to control how many replicates of each type are
generated per RL sequence.
