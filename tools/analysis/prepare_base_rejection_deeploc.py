#!/usr/bin/env python3
"""Prepare base-IDiom rejection-sampling FASTAs for DeepLoc scoring."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def write_fasta(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in df.itertuples(index=False):
            seq_id = str(row.sequence_id).replace("|", "__")
            seq = str(row.sequence)
            handle.write(f">{seq_id}\n")
            for start in range(0, len(seq), 80):
                handle.write(seq[start : start + 80] + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", default="data/processed/test2_full_pool_protgps_scores.csv")
    parser.add_argument("--run-name", default="base_rejection_10k_top2k")
    parser.add_argument("--k", type=int, default=2000)
    parser.add_argument("--out-root", default="data/processed/deeploc_custom_inputs")
    args = parser.parse_args()

    scores = pd.read_csv(args.scores)
    base = scores[scores["source"].eq("base_idp")].copy()
    if base.empty:
        raise SystemExit("No source == base_idp rows found.")

    base["pbody_specific"] = base["protgps_p-body"] - base["protgps_stress_granule"]
    base["sg_specific"] = base["protgps_stress_granule"] - base["protgps_p-body"]

    out_dir = Path(args.out_root) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Primary selection: target specificity. These sets are non-overlapping by construction.
    spec_pbody = base.nlargest(args.k, "pbody_specific").copy()
    spec_sg = base.nlargest(args.k, "sg_specific").copy()
    write_fasta(spec_pbody, out_dir / "base_reject_spec_pbody.fasta")
    write_fasta(spec_sg, out_dir / "base_reject_spec_sg.fasta")

    # Secondary selection: raw target score, excluding overlaps so labels are not duplicated.
    raw_p_ids = set(base.nlargest(args.k, "protgps_p-body")["sequence_id"])
    raw_s_ids = set(base.nlargest(args.k, "protgps_stress_granule")["sequence_id"])
    overlap = raw_p_ids & raw_s_ids
    raw_pbody = base[base["sequence_id"].isin(raw_p_ids - overlap)].copy()
    raw_sg = base[base["sequence_id"].isin(raw_s_ids - overlap)].copy()
    write_fasta(raw_pbody, out_dir / "base_reject_raw_pbody.fasta")
    write_fasta(raw_sg, out_dir / "base_reject_raw_sg.fasta")

    manifest = pd.concat(
        [
            spec_pbody.assign(rejection_source="base_reject_spec_pbody"),
            spec_sg.assign(rejection_source="base_reject_spec_sg"),
            raw_pbody.assign(rejection_source="base_reject_raw_pbody"),
            raw_sg.assign(rejection_source="base_reject_raw_sg"),
        ],
        ignore_index=True,
    )
    manifest_path = out_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    summary = pd.DataFrame(
        [
            {"source": "base_reject_spec_pbody", "n": len(spec_pbody), "selection": "top pbody_specific"},
            {"source": "base_reject_spec_sg", "n": len(spec_sg), "selection": "top sg_specific"},
            {
                "source": "base_reject_raw_pbody",
                "n": len(raw_pbody),
                "selection": f"top protgps_p-body excluding {len(overlap)} overlapping raw-top sequences",
            },
            {
                "source": "base_reject_raw_sg",
                "n": len(raw_sg),
                "selection": f"top protgps_stress_granule excluding {len(overlap)} overlapping raw-top sequences",
            },
        ]
    )
    summary_path = out_dir / "selection_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Wrote FASTAs to {out_dir}")
    print(summary.to_string(index=False))
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
