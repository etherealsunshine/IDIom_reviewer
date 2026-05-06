#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

PLAN = """# IDiom Audit Download Plan

This file is intentionally a plan, not an automatic download script.
It avoids pulling the 26 GB model repo or 186 GB dataset bundle until you opt in.

Code repo:
  git clone --depth 1 https://github.com/rotskoff-group/idiom idiom_repo

Heavy model checkpoint bundle, later:
  cd idiom_repo
  hf download jxliu2/idiom --local-dir ./models

Dataset bundle, later:
  cd idiom_repo
  hf download jxliu2/idiom-datasets --repo-type=dataset --local-dir ./datasets

Training IDR FASTA only, still large:
  cd idiom_repo
  hf download jxliu2/idiom-datasets \\
    idr_datasets/training_sequences/AFDB_IDR_90_FIM_512_idrs.fasta \\
    --repo-type=dataset \\
    --local-dir ./datasets

Likely generated IDR FASTA paths in the dataset:
  datasets/idr_datasets/generated_sequences/generated_idps/generated_idrs.fasta
  datasets/idr_datasets/generated_sequences/generated_idrs/generated_idrs.fasta
  datasets/idr_datasets/generated_sequences/generated_protgps/generated_nucleolus/generated_idrs.fasta
  datasets/idr_datasets/generated_sequences/generated_protgps/generated_chromosome/generated_idrs.fasta
  datasets/idr_datasets/generated_sequences/generated_protgps/generated_p-body/generated_idrs.fasta
  datasets/idr_datasets/generated_sequences/generated_protgps/generated_stress_granule/generated_idrs.fasta
"""


def main() -> None:
    out = Path("data/DOWNLOAD_PLAN.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(PLAN)
    print(out)


if __name__ == "__main__":
    main()
