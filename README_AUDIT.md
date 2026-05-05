# IDiom RL Post-Training Audit Scaffold

This workspace contains a lightweight audit package for the IDiom RL/ProtGPS question.
It does not download the heavy HuggingFace model bundle or the 186 GB dataset bundle.
The public IDiom code repo is cloned at `idiom_repo/` so the ProtGPS wrapper can use
the authors' own inference utilities later.

## What Is Implemented

- Test 1 composition-matched scrambling: exact composition/length-preserving shuffles,
  paired Wilcoxon statistics, scatter plots, and group box plots.
- Test 2 shallow feature probe: sequence features, RandomForestRegressor per ProtGPS
  compartment, metrics, feature importances, and predicted-vs-actual plots.
- Test 3 motif-spiked cheap baselines: IDR-like random backgrounds with compartment
  motif stuffing.
- Test 4 off-target/specificity: target-minus-best-off-target specificity,
  12x12 score correlation, and mean score profile plots.
- Test 7 diversity summary: unique fraction, per-sequence entropy, and sampled
  pairwise identity summary.
- ProtGPS scoring wrapper: `tools/protgps/score_csv.py`, using the cloned
  `idiom_repo/rewards/protgps/scripts/inference.py` code after checkpoints exist.

Tests 5, 6, 8, 9, and 10 are not fully automated yet because they require external
predictors, MMseqs2, or curated natural-positive data. The package is structured so
their resulting score CSVs can be folded into the same plotting/statistics layer.

## Install

From `/Users/utkarsh/IDIom`:

```bash
uv sync
```

Then run commands through the workspace environment:

```bash
UV_CACHE_DIR=.uv-cache uv run idiom-audit --help
```

## Data Layout

Expected local files after later HuggingFace downloads:

```text
idiom_repo/datasets/idr_datasets/generated_sequences/generated_idps/generated_full.fasta
idiom_repo/datasets/idr_datasets/generated_sequences/generated_idrs/generated_full.fasta
idiom_repo/datasets/idr_datasets/generated_sequences/generated_protgps/generated_nucleolus/generated_full.fasta
idiom_repo/datasets/idr_datasets/generated_sequences/generated_protgps/generated_chromosome/generated_full.fasta
idiom_repo/datasets/idr_datasets/generated_sequences/generated_protgps/generated_p-body/generated_full.fasta
idiom_repo/datasets/idr_datasets/generated_sequences/generated_protgps/generated_stress_granule/generated_full.fasta
idiom_repo/datasets/idr_datasets/training_sequences/AFDB_IDR_90_FIM_512_idrs.fasta
```

The IDiom repo README says the full model bundle is about 26 GB and the full dataset
bundle is about 186 GB. Do those only when you are ready.

## Prepare Sequence Tables

Edit paths in `tools/workflows/prepare_audit_inputs.sh`, then run:

```bash
bash tools/workflows/prepare_audit_inputs.sh
```

This creates:

```text
data/processed/all_sequences.csv
data/processed/all_scrambles.csv
data/processed/all_scrambles_for_scoring.fasta
data/processed/cheap_baselines.csv
data/processed/cheap_baselines_for_scoring.fasta
data/processed/all_features.csv
```

## Score With ProtGPS

### Modal Pilot

For GPU scoring without waiting on local hardware, use the Modal scorer. The
one-time model asset setup was built around a persistent Modal Volume named
`idiom-protgps-models`, and downloads only the ProtGPS checkpoint plus the small
ESM2 assets needed by the reward model.

To download the Test 1 generated FASTAs into a separate persistent Modal Volume
named `idiom-audit-data`:

```bash
UV_CACHE_DIR=.uv-cache uv run modal run tools/modal/download_test1_fastas.py --action inspect
UV_CACHE_DIR=.uv-cache uv run modal run tools/modal/download_test1_fastas.py --action download
UV_CACHE_DIR=.uv-cache uv run modal run tools/modal/download_test1_fastas.py --action layout
```

The downloaded files live under `/audit_data` inside Modal:

```text
/audit_data/idr_datasets/generated_sequences/generated_idps/generated_idrs.fasta
/audit_data/idr_datasets/generated_sequences/generated_idps/generated_full.fasta
/audit_data/idr_datasets/generated_sequences/generated_protgps/generated_nucleolus/generated_idrs.fasta
/audit_data/idr_datasets/generated_sequences/generated_protgps/generated_chromosome/generated_idrs.fasta
/audit_data/idr_datasets/generated_sequences/generated_protgps/generated_p-body/generated_idrs.fasta
/audit_data/idr_datasets/generated_sequences/generated_protgps/generated_stress_granule/generated_idrs.fasta
```

Small smoke run:

```bash
UV_CACHE_DIR=.uv-cache uv run modal run tools/modal/protgps_score_csv.py \
  --input-csv results/smoke/cheap.csv \
  --output-csv results/smoke/cheap_modal_scores.csv \
  --limit 12 \
  --shard-size 12 \
  --batch-size 8
```

Pilot-scale run:

```bash
UV_CACHE_DIR=.uv-cache uv run modal run tools/modal/protgps_score_csv.py \
  --input-csv data/processed/all_sequences.csv \
  --output-csv data/processed/all_sequences_protgps_scores_modal_pilot.csv \
  --limit 5000 \
  --shard-size 512 \
  --batch-size 32
```

The Modal function currently allows up to 4 L40S containers via `max_containers=4`.
Increase that in `tools/modal/protgps_score_csv.py` only after a pilot confirms
cost and throughput look sane.

For a longer unattended Test 1 pilot that prepares originals/scrambles from the
Modal FASTA Volume, scores them, downloads the score CSVs, and runs the local
Test 1 analysis:

```bash
tmux new -s idiom-test1
bash tools/workflows/run_modal_test1_pilot.sh 2>&1 | tee logs/test1_pilot_1k.log
```

By default this makes `3` full composition scrambles and `3` block scrambles
per RL sequence. Block scrambles cut sequences into 10-20 residue chunks and
shuffle those chunks, preserving local motifs while disrupting longer-range
arrangement.

Override limits if desired:

```bash
RUN_NAME=test1_pilot_2k BASE_LIMIT=2000 RL_LIMIT_PER_TARGET=2000 BATCH_SIZE=64 \
  bash tools/workflows/run_modal_test1_pilot.sh 2>&1 | tee logs/test1_pilot_2k.log
```

Override scramble depth:

```bash
SCRAMBLES_PER_TYPE=2 RUN_NAME=test1_pilot_1k_s2 \
  bash tools/workflows/run_modal_test1_pilot.sh 2>&1 | tee logs/test1_pilot_1k_s2.log
```

### Local Scoring

After downloading the ProtGPS checkpoint and ESM directory:

```bash
UV_CACHE_DIR=.uv-cache uv run python tools/protgps/score_csv.py \
  --idiom-repo idiom_repo \
  --model-path idiom_repo/models/protgps/protgps/32bf44b16a4e770a674896b81dfb3729epoch=26.ckpt \
  --esm-dir idiom_repo/models/protgps/esm_models/esm2 \
  --input data/processed/all_sequences.csv \
  --output data/processed/all_sequences_protgps_scores.csv \
  --batch-size 32
```

Run the same command for `all_scrambles.csv` and `cheap_baselines.csv`.
The wrapper writes score columns named `protgps_<compartment>` for all 12 ProtGPS
outputs, using this order from the authors' code:

```text
nuclear_speckle, p-body, pml-bdoy, post_synaptic_density, stress_granule,
chromosome, nucleolus, nuclear_pore_complex, cajal_body, rna_granule,
cell_junction, transcriptional
```

## Run Analyses

Test 1:

```bash
python -m idiom_audit.cli test1 \
  --originals data/processed/all_sequences_protgps_scores.csv \
  --scrambles data/processed/all_scrambles_protgps_scores.csv \
  --groups data/processed/all_sequences_protgps_scores.csv data/processed/all_scrambles_protgps_scores.csv \
  --out-dir results/test1_scramble
```

Test 2:

```bash
python -m idiom_audit.cli test2 \
  --features data/processed/all_features.csv \
  --scores data/processed/all_sequences_protgps_scores.csv \
  --out-dir results/test2_feature_probe
```

Test 4:

```bash
python -m idiom_audit.cli test4 \
  --scores data/processed/all_sequences_protgps_scores.csv \
  --out-dir results/test4_specificity
```

Test 7:

```bash
python -m idiom_audit.cli test7 \
  --input data/processed/all_sequences.csv \
  --out-dir results/test7_diversity
```

## Score CSV Contract

Analysis scripts expect:

```text
sequence_id, sequence, source, compartment_target, protgps_nucleolus, ...
```

For scrambles, include:

```text
original_sequence_id, scramble_replicate
```

Every intermediate table is saved as CSV so ProtGPS inference can be cached and
reused. Seed `33402` is used by default for scrambling, sampling, cheap baselines,
and train/test splits.
