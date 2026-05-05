#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

COMPARTMENTS = [
    "nuclear_speckle",
    "p-body",
    "pml-bdoy",
    "post_synaptic_density",
    "stress_granule",
    "chromosome",
    "nucleolus",
    "nuclear_pore_complex",
    "cajal_body",
    "rna_granule",
    "cell_junction",
    "transcriptional",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Score CSV sequences with the IDiom-bundled ProtGPS model.")
    parser.add_argument("--idiom-repo", default="idiom_repo")
    parser.add_argument("--model-path", required=True, help="Path to ProtGPS .ckpt")
    parser.add_argument("--esm-dir", required=True, help="Path to ESM2 directory used by ProtGPS")
    parser.add_argument("--input", required=True, help="CSV with at least sequence_id and sequence")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sequence-col", default="sequence")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None, help="Default: cuda if available, else cpu")
    args = parser.parse_args()

    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "ProtGPS scoring requires torch and the IDiom/ProtGPS dependencies. "
            "Install those only when you are ready to score with the checkpoint."
        ) from exc

    protgps_root = Path(args.idiom_repo) / "rewards" / "protgps"
    sys.path.insert(0, str(protgps_root.resolve()))
    from scripts.inference import load_model, predict_condensates

    df = pd.read_csv(args.input)
    if args.sequence_col not in df.columns:
        raise SystemExit(f"Missing sequence column: {args.sequence_col}")

    valid = df[args.sequence_col].astype(str).str.len().lt(1800)
    scored = df.loc[valid].copy()
    sequences = scored[args.sequence_col].astype(str).str.upper().tolist()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_path, args.esm_dir)
    model.eval().to(torch.device(device))
    scores = predict_condensates(model, sequences, batch_size=args.batch_size, round=False).numpy()
    for i, compartment in enumerate(COMPARTMENTS):
        scored[f"protgps_{compartment}"] = scores[:, i]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out, index=False)


if __name__ == "__main__":
    main()
