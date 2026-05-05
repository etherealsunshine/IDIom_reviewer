from __future__ import annotations

import time
from pathlib import Path

import modal

APP_NAME = "idiom-test1-score"
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

TARGET_FASTAS = {
    "nucleolus": "idr_datasets/generated_sequences/generated_protgps/generated_nucleolus/generated_idrs.fasta",
    "chromosome": "idr_datasets/generated_sequences/generated_protgps/generated_chromosome/generated_idrs.fasta",
    "p-body": "idr_datasets/generated_sequences/generated_protgps/generated_p-body/generated_idrs.fasta",
    "stress_granule": "idr_datasets/generated_sequences/generated_protgps/generated_stress_granule/generated_idrs.fasta",
}
BASE_FASTA = "idr_datasets/generated_sequences/generated_idps/generated_idrs.fasta"


def _read_fasta(path: Path, source: str, target: str | None, limit: int) -> list[dict]:
    rows = []
    seq_id = None
    chunks: list[str] = []

    def flush():
        if seq_id is None:
            return
        seq = "".join(chunks).upper().replace("*", "")
        seq = "".join(aa for aa in seq if aa.isalpha())
        if seq:
            rows.append(
                {
                    "sequence_id": seq_id,
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


def _full_scramble(seq: str, rng):
    import numpy as np

    arr = np.array(list(seq))
    rng.shuffle(arr)
    return "".join(arr.tolist())


def _block_scramble(seq: str, rng, min_block: int = 10, max_block: int = 20) -> str:
    if len(seq) <= min_block:
        return seq
    chunks = []
    pos = 0
    while pos < len(seq):
        remaining = len(seq) - pos
        if remaining <= max_block:
            chunks.append(seq[pos:])
            break
        size = int(rng.integers(min_block, max_block + 1))
        chunks.append(seq[pos : pos + size])
        pos += size
    if len(chunks) > 1:
        rng.shuffle(chunks)
    return "".join(chunks)


def _scramble_rows(rows: list[dict], scrambles_per_type: int, seed: int):
    import numpy as np

    rng = np.random.default_rng(seed)
    out = []
    for row in rows:
        seq = row["sequence"]
        for scramble_type, scramble_fn in (
            ("full", _full_scramble),
            ("block", _block_scramble),
        ):
            for rep in range(scrambles_per_type):
                scrambled = dict(row)
                scrambled["sequence_id"] = f"{row['sequence_id']}__{scramble_type}_scramble{rep + 1}"
                scrambled["sequence"] = scramble_fn(seq, rng)
                scrambled["source"] = f"scrambled_{scramble_type}_" + str(row["source"])
                scrambled["original_sequence_id"] = row["sequence_id"]
                scrambled["scramble_replicate"] = rep + 1
                scrambled["scramble_type"] = scramble_type
                out.append(scrambled)
    return out


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
    sequences = []
    invalid_rows = []
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
    gpu="L40S",
    timeout=8 * 60 * 60,
    volumes={DATA_ROOT: data_volume, MODEL_ROOT: model_volume},
)
def run_test1_scoring(
    run_name: str,
    base_limit: int = 1000,
    rl_limit_per_target: int = 1000,
    scrambles_per_type: int = 3,
    batch_size: int = 64,
    seed: int = 33402,
) -> dict:
    import pandas as pd

    start = time.monotonic()
    root = Path(DATA_ROOT)
    processed = root / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    base_rows = _read_fasta(root / BASE_FASTA, "base_idp", None, base_limit)
    rl_rows = []
    for target, rel_path in TARGET_FASTAS.items():
        rl_rows.extend(
            _read_fasta(
                root / rel_path,
                source=f"rl_{target}",
                target=target,
                limit=rl_limit_per_target,
            )
        )

    scramble_rows = _scramble_rows(rl_rows, scrambles_per_type, seed)
    originals = base_rows + rl_rows

    originals_meta_path = processed / f"{run_name}_originals.csv"
    scrambles_meta_path = processed / f"{run_name}_scrambles.csv"
    originals_scores_path = processed / f"{run_name}_originals_protgps_scores.csv"
    scrambles_scores_path = processed / f"{run_name}_scrambles_protgps_scores.csv"

    pd.DataFrame(originals).to_csv(originals_meta_path, index=False)
    pd.DataFrame(scramble_rows).to_csv(scrambles_meta_path, index=False)

    score_start = time.monotonic()
    scored_originals = _score_rows(originals, batch_size=batch_size)
    originals_score_seconds = time.monotonic() - score_start
    pd.DataFrame(scored_originals).to_csv(originals_scores_path, index=False)

    score_start = time.monotonic()
    scored_scrambles = _score_rows(scramble_rows, batch_size=batch_size)
    scrambles_score_seconds = time.monotonic() - score_start
    pd.DataFrame(scored_scrambles).to_csv(scrambles_scores_path, index=False)

    data_volume.commit()

    elapsed = time.monotonic() - start
    return {
        "run_name": run_name,
        "base_limit": base_limit,
        "rl_limit_per_target": rl_limit_per_target,
        "scrambles_per_type": scrambles_per_type,
        "scramble_types": ["full", "block"],
        "batch_size": batch_size,
        "n_base": len(base_rows),
        "n_rl": len(rl_rows),
        "n_scrambles": len(scramble_rows),
        "n_scored_originals": len(scored_originals),
        "n_scored_scrambles": len(scored_scrambles),
        "originals_score_seconds": originals_score_seconds,
        "scrambles_score_seconds": scrambles_score_seconds,
        "wall_seconds": elapsed,
        "remote_outputs": {
            "originals_meta": str(originals_meta_path.relative_to(root)),
            "scrambles_meta": str(scrambles_meta_path.relative_to(root)),
            "originals_scores": str(originals_scores_path.relative_to(root)),
            "scrambles_scores": str(scrambles_scores_path.relative_to(root)),
        },
    }


@app.local_entrypoint()
def main(
    run_name: str = "test1_pilot_1k",
    base_limit: int = 1000,
    rl_limit_per_target: int = 1000,
    scrambles_per_type: int = 3,
    batch_size: int = 64,
    seed: int = 33402,
):
    result = run_test1_scoring.remote(
        run_name=run_name,
        base_limit=base_limit,
        rl_limit_per_target=rl_limit_per_target,
        scrambles_per_type=scrambles_per_type,
        batch_size=batch_size,
        seed=seed,
    )
    print(result)
