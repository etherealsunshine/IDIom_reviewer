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

Amended cheap-baseline probe:

```bash
bash tools/workflows/run_amended_baseline_probe.sh 2>&1 | tee logs/amended_baseline_test1_pilot_1k.log
```

Full Test 2 RandomForest probe:

```bash
bash tools/workflows/run_test2_random_forest.sh 2>&1 | tee logs/test2_random_forest_full.log
```

DeepLoc validation:

```bash
bash tools/workflows/run_deeploc_validation.sh 2>&1 | tee logs/deeploc_pilot_2k.log
```

If the official DeepLoc package has not been uploaded to the Modal cache Volume
yet, run:

```bash
DEEPLOC_PACKAGE=/path/to/deeploc-2.1.All.tar.gz INSTALL_SMOKE_ONLY=1 \
  bash tools/workflows/run_deeploc_validation.sh
```
