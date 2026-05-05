# IDiom Audit Download Plan

This is intentionally a plan, not an automatic download script. It avoids pulling
the 26 GB model repo or 186 GB dataset bundle until you opt in.

Code repo:

```bash
git clone --depth 1 https://github.com/rotskoff-group/idiom idiom_repo
```

Heavy model checkpoint bundle, later:

```bash
cd idiom_repo
hf download jxliu2/idiom --local-dir ./models
```

Dataset bundle, later:

```bash
cd idiom_repo
hf download jxliu2/idiom-datasets --repo-type=dataset --local-dir ./datasets
```

Training IDR FASTA only, still large:

```bash
cd idiom_repo
hf download jxliu2/idiom-datasets \
  idr_datasets/training_sequences/AFDB_IDR_90_FIM_512_idrs.fasta \
  --repo-type=dataset \
  --local-dir ./datasets
```

Likely generated FASTA paths in the dataset, based on the cloned repo:

```text
datasets/idr_datasets/generated_sequences/generated_idps/generated_full.fasta
datasets/idr_datasets/generated_sequences/generated_idrs/generated_full.fasta
datasets/idr_datasets/generated_sequences/generated_protgps/generated_chromosome/generated_full.fasta
datasets/idr_datasets/generated_sequences/generated_protgps/generated_nucleolus/generated_full.fasta
datasets/idr_datasets/generated_sequences/generated_protgps/generated_p-body/generated_full.fasta
datasets/idr_datasets/generated_sequences/generated_protgps/generated_stress_granule/generated_full.fasta
```
