#!/usr/bin/env bash
set -euo pipefail

# Edit these paths after downloading the generated FASTAs.
BASE_IDP="idiom_repo/datasets/idr_datasets/generated_sequences/generated_idps/generated_full.fasta"
RL_NUC="idiom_repo/datasets/idr_datasets/generated_sequences/generated_protgps/generated_nucleolus/generated_full.fasta"
RL_CHR="idiom_repo/datasets/idr_datasets/generated_sequences/generated_protgps/generated_chromosome/generated_full.fasta"
RL_PBD="idiom_repo/datasets/idr_datasets/generated_sequences/generated_protgps/generated_p-body/generated_full.fasta"
RL_SG="idiom_repo/datasets/idr_datasets/generated_sequences/generated_protgps/generated_stress_granule/generated_full.fasta"

python -m idiom_audit.cli load-fastas \
  --fasta "$BASE_IDP:base_idp" \
  --fasta "$RL_NUC:rl_nucleolus:nucleolus" \
  --fasta "$RL_CHR:rl_chromosome:chromosome" \
  --fasta "$RL_PBD:rl_p-body:p-body" \
  --fasta "$RL_SG:rl_stress_granule:stress_granule" \
  --output data/processed/all_sequences.csv

python -m idiom_audit.cli make-scrambles \
  --input data/processed/all_sequences.csv \
  --output data/processed/all_scrambles.csv \
  --fasta-output data/processed/all_scrambles_for_scoring.fasta

python -m idiom_audit.cli cheap-baselines \
  --output data/processed/cheap_baselines.csv \
  --fasta-output data/processed/cheap_baselines_for_scoring.fasta

python -m idiom_audit.cli featurize \
  --input data/processed/all_sequences.csv \
  --output data/processed/all_features.csv
