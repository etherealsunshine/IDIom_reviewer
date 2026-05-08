#!/usr/bin/env python3
"""Analyze DeepLoc outputs for base-IDiom rejection-sampling controls."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from idiom_audit.deeploc import pair_classifier, read_deeploc_outputs


PAIRS = [
    ("base_reject_spec_pbody", "base_reject_spec_sg", "spec_pbody", "spec_sg"),
    ("base_reject_raw_pbody", "base_reject_raw_sg", "raw_pbody", "raw_sg"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_deeploc_outputs(args.results_dir)
    df.to_csv(out_dir / "base_rejection_deeploc_predictions.csv", index=False)

    rows = []
    for source_a, source_b, label_a, label_b in PAIRS:
        result = pair_classifier(df, source_a, source_b, label_a, label_b)
        cm = result.pop("confusion_matrix")
        result["confusion_matrix"] = cm.tolist()
        rows.append(result)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "base_rejection_deeploc_classifier_metrics.csv", index=False)

    print(metrics.to_string(index=False))
    print(f"Wrote {out_dir / 'base_rejection_deeploc_classifier_metrics.csv'}")


if __name__ == "__main__":
    main()
