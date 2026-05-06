from __future__ import annotations

import random
import time
from pathlib import Path

import modal

APP_NAME = "idiom-test2-score-pool"
DATA_ROOT = "/audit_data"
MODEL_ROOT = "/models"
IDIOM_ROOT = "/opt/idiom"

data_volume = modal.Volume.from_name("idiom-audit-data", create_if_missing=True)
model_volume = modal.Volume.from_name("idiom-protgps-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "pandas",
        "numpy",
        "torch==2.4.0",
        "pytorch-lightning==1.9.5",
        "torchmetrics<1.0",
        "tqdm",
        "biopython",
        "fair-esm",
        "huggingface_hub",
    )
    .run_commands(
        f"git clone --depth 1 https://github.com/rotskoff-group/idiom {IDIOM_ROOT}"
    )
    .env(
        {
            "PYTHONPATH": f"{IDIOM_ROOT}/src:{IDIOM_ROOT}/rewards/protgps",
        }
    )
)

app = modal.App(APP_NAME)

EXTRA_TEST2_PATTERNS = [
    "idr_datasets/generated_sequences/generated_idps/generated_idrs.fasta",
    "idr_datasets/generated_sequences/generated_protgps/generated_nucleolus/generated_idrs.fasta",
    "idr_datasets/generated_sequences/generated_protgps/generated_chromosome/generated_idrs.fasta",
    "idr_datasets/generated_sequences/generated_protgps/generated_p-body/generated_idrs.fasta",
    "idr_datasets/generated_sequences/generated_protgps/generated_stress_granule/generated_idrs.fasta",
    "idr_datasets/reference_sequences/CATH/cath-domain-seqs-S60_1000.fa",
    "idr_datasets/reference_sequences/DisProt/disprot_idrs.fasta",
    "idr_datasets/training_sequences/AFDB_IDR_90_FIM_512_idrs.fasta",
]

TARGET_FASTAS = {
    "nucleolus": "idr_datasets/generated_sequences/generated_protgps/generated_nucleolus/generated_idrs.fasta",
    "chromosome": "idr_datasets/generated_sequences/generated_protgps/generated_chromosome/generated_idrs.fasta",
    "p-body": "idr_datasets/generated_sequences/generated_protgps/generated_p-body/generated_idrs.fasta",
    "stress_granule": "idr_datasets/generated_sequences/generated_protgps/generated_stress_granule/generated_idrs.fasta",
}
BASE_IDP_FASTA = "idr_datasets/generated_sequences/generated_idps/generated_idrs.fasta"
DISPROT_FASTA = "idr_datasets/reference_sequences/DisProt/disprot_idrs.fasta"
CATH_FASTA = "idr_datasets/reference_sequences/CATH/cath-domain-seqs-S60_1000.fa"
TRAINING_IDR_FASTA = "idr_datasets/training_sequences/AFDB_IDR_90_FIM_512_idrs.fasta"


def _clean_sequence(seq: str) -> str:
    allowed = set("ACDEFGHIKLMNPQRSTVWY")
    return "".join(aa for aa in seq.upper().replace("*", "") if aa in allowed)


def _read_fasta(path: Path, source: str, target: str | None = None, limit: int = 0) -> list[dict]:
    rows = []
    seq_id = None
    chunks: list[str] = []

    def flush():
        if seq_id is None:
            return
        seq = _clean_sequence("".join(chunks))
        if seq:
            rows.append(
                {
                    "sequence_id": f"{source}|{seq_id}",
                    "sequence": seq,
                    "source": source,
                    "compartment_target": target,
                }
            )

    with path.open() as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                if limit and len(rows) >= limit:
                    return rows
                seq_id = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
    flush()
    return rows[:limit] if limit else rows


def _reservoir_sample_fasta(path: Path, source: str, limit: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    sample: list[dict] = []
    seen = 0
    seq_id = None
    chunks: list[str] = []

    def maybe_add():
        nonlocal seen
        if seq_id is None:
            return
        seq = _clean_sequence("".join(chunks))
        if not seq:
            return
        row = {
            "sequence_id": f"{source}|{seq_id}",
            "sequence": seq,
            "source": source,
            "compartment_target": None,
        }
        seen += 1
        if len(sample) < limit:
            sample.append(row)
        else:
            idx = rng.randrange(seen)
            if idx < limit:
                sample[idx] = row

    with path.open() as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                maybe_add()
                seq_id = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
    maybe_add()
    return sample


def _score_rows(rows: list[dict], batch_size: int):
    import sys
    from pathlib import Path

    import torch

    sys.path.insert(0, f"{IDIOM_ROOT}/rewards/protgps")
    from scripts.inference import COMPARTMENTS, load_model, predict_condensates

    model_path = Path(MODEL_ROOT) / "protgps/protgps/32bf44b16a4e770a674896b81dfb3729epoch=26.ckpt"
    esm_dir = Path(MODEL_ROOT) / "protgps/esm_models/esm2"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(str(model_path), str(esm_dir))
    model.eval().to(device)

    valid_rows = []
    invalid_rows = []
    sequences = []
    for row in rows:
        seq = str(row.get("sequence", "")).upper()
        if 0 < len(seq) < 1800:
            clean = dict(row)
            clean["sequence"] = seq
            valid_rows.append(clean)
            sequences.append(seq)
        else:
            bad = dict(row)
            bad["protgps_error"] = f"invalid_length_{len(seq)}"
            invalid_rows.append(bad)

    scored = []
    if sequences:
        scores = predict_condensates(model, sequences, batch_size=batch_size, round=False)
        for row, score_vec in zip(valid_rows, scores.tolist()):
            for compartment, score in zip(COMPARTMENTS, score_vec):
                row[f"protgps_{compartment}"] = float(score)
            scored.append(row)
    return scored + invalid_rows


@app.function(
    image=image,
    timeout=6 * 60 * 60,
    volumes={DATA_ROOT: data_volume},
)
def download_test2_sources() -> dict:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="jxliu2/idiom-datasets",
        repo_type="dataset",
        local_dir=DATA_ROOT,
        allow_patterns=EXTRA_TEST2_PATTERNS,
    )
    data_volume.commit()
    root = Path(DATA_ROOT)
    return {
        path: {
            "exists": (root / path).exists(),
            "size_bytes": (root / path).stat().st_size if (root / path).exists() else None,
        }
        for path in EXTRA_TEST2_PATTERNS
    }


@app.function(
    image=image,
    gpu="L40S",
    timeout=8 * 60 * 60,
    volumes={DATA_ROOT: data_volume, MODEL_ROOT: model_volume},
)
def assemble_and_score_test2_pool(
    run_name: str = "test2_full",
    base_limit: int = 10000,
    rl_limit_per_target: int = 10000,
    training_limit: int = 10000,
    disprot_limit: int = 0,
    cath_limit: int = 0,
    batch_size: int = 64,
    seed: int = 33402,
) -> dict:
    import pandas as pd

    start = time.monotonic()
    root = Path(DATA_ROOT)
    processed = root / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    rows = []
    rows.extend(_read_fasta(root / BASE_IDP_FASTA, "base_idp", limit=base_limit))
    for target, rel_path in TARGET_FASTAS.items():
        rows.extend(_read_fasta(root / rel_path, f"rl_{target}", target=target, limit=rl_limit_per_target))
    rows.extend(_read_fasta(root / DISPROT_FASTA, "disprot_idr", limit=disprot_limit))
    rows.extend(_read_fasta(root / CATH_FASTA, "cath_s60", limit=cath_limit))
    rows.extend(_reservoir_sample_fasta(root / TRAINING_IDR_FASTA, "training_idr", training_limit, seed))

    meta_path = processed / f"{run_name}_pool.csv"
    scores_path = processed / f"{run_name}_pool_protgps_scores.csv"
    pd.DataFrame(rows).to_csv(meta_path, index=False)

    score_start = time.monotonic()
    scored = _score_rows(rows, batch_size=batch_size)
    score_seconds = time.monotonic() - score_start
    pd.DataFrame(scored).to_csv(scores_path, index=False)
    data_volume.commit()

    counts = pd.DataFrame(rows)["source"].value_counts().to_dict()
    return {
        "run_name": run_name,
        "n_rows": len(rows),
        "n_scored": len(scored),
        "source_counts": counts,
        "batch_size": batch_size,
        "score_seconds": score_seconds,
        "wall_seconds": time.monotonic() - start,
        "remote_outputs": {
            "pool": str(meta_path.relative_to(root)),
            "scores": str(scores_path.relative_to(root)),
        },
    }


@app.local_entrypoint()
def main(
    action: str = "score",
    run_name: str = "test2_full",
    base_limit: int = 10000,
    rl_limit_per_target: int = 10000,
    training_limit: int = 10000,
    disprot_limit: int = 0,
    cath_limit: int = 0,
    batch_size: int = 64,
    seed: int = 33402,
):
    if action == "download":
        print(download_test2_sources.remote())
    elif action == "score":
        print(
            assemble_and_score_test2_pool.remote(
                run_name=run_name,
                base_limit=base_limit,
                rl_limit_per_target=rl_limit_per_target,
                training_limit=training_limit,
                disprot_limit=disprot_limit,
                cath_limit=cath_limit,
                batch_size=batch_size,
                seed=seed,
            )
        )
    else:
        raise ValueError("action must be one of: download, score")
