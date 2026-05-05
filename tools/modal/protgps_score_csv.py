from __future__ import annotations

import math
import time
from pathlib import Path

import modal

APP_NAME = "idiom-protgps"
MODEL_ROOT = "/models"
IDIOM_ROOT = "/opt/idiom"

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
        "openpyxl",
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

app = modal.App(f"{APP_NAME}-score-csv")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_project_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return _project_root() / candidate


def _chunks(rows: list[dict], size: int):
    for start in range(0, len(rows), size):
        yield start, rows[start : start + size]


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60,
    volumes={MODEL_ROOT: model_volume},
    max_containers=4,
)
def score_rows_shard(rows: list[dict], batch_size: int = 32, sequence_col: str = "sequence") -> dict:
    import sys
    import time
    from pathlib import Path

    import torch

    sys.path.insert(0, f"{IDIOM_ROOT}/rewards/protgps")
    from scripts.inference import COMPARTMENTS, load_model, predict_condensates

    model_path = Path(MODEL_ROOT) / "protgps/protgps/32bf44b16a4e770a674896b81dfb3729epoch=26.ckpt"
    esm_dir = Path(MODEL_ROOT) / "protgps/esm_models/esm2"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    start = time.monotonic()
    model = load_model(str(model_path), str(esm_dir))
    model.eval().to(device)
    load_seconds = time.monotonic() - start

    valid_rows = []
    invalid_rows = []
    sequences = []
    for row in rows:
        seq = str(row.get(sequence_col, "")).upper()
        if 0 < len(seq) < 1800:
            clean_row = dict(row)
            clean_row[sequence_col] = seq
            valid_rows.append(clean_row)
            sequences.append(seq)
        else:
            bad_row = dict(row)
            bad_row["protgps_error"] = f"invalid_length_{len(seq)}"
            invalid_rows.append(bad_row)

    score_start = time.monotonic()
    scored_rows = []
    if sequences:
        scores = predict_condensates(model, sequences, batch_size=batch_size, round=False)
        for row, score_vec in zip(valid_rows, scores.tolist()):
            for compartment, score in zip(COMPARTMENTS, score_vec):
                row[f"protgps_{compartment}"] = float(score)
            scored_rows.append(row)
    score_seconds = time.monotonic() - score_start

    return {
        "rows": scored_rows + invalid_rows,
        "n_input": len(rows),
        "n_scored": len(scored_rows),
        "n_invalid": len(invalid_rows),
        "load_seconds": load_seconds,
        "score_seconds": score_seconds,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


@app.local_entrypoint()
def main(
    input_csv: str,
    output_csv: str,
    limit: int = 0,
    shard_size: int = 512,
    batch_size: int = 32,
    sequence_col: str = "sequence",
):
    import pandas as pd

    input_path = _resolve_project_path(input_csv)
    output_path = _resolve_project_path(output_csv)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    df = pd.read_csv(input_path)
    if sequence_col not in df.columns:
        raise ValueError(f"Missing sequence column {sequence_col!r} in {input_path}")
    if limit and limit > 0:
        df = df.head(limit)

    rows = df.to_dict(orient="records")
    if not rows:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"No rows to score. Wrote empty CSV to {output_path}")
        return

    n_shards = math.ceil(len(rows) / shard_size)
    print(
        f"Scoring {len(rows)} rows from {input_path} as {n_shards} shard(s): "
        f"shard_size={shard_size}, batch_size={batch_size}, max_containers=4"
    )

    start = time.monotonic()
    shard_inputs = [chunk for _, chunk in _chunks(rows, shard_size)]
    results = list(
        score_rows_shard.map(
            shard_inputs,
            kwargs={"batch_size": batch_size, "sequence_col": sequence_col},
            order_outputs=True,
            return_exceptions=False,
        )
    )
    elapsed = time.monotonic() - start

    scored_rows = []
    n_scored = 0
    n_invalid = 0
    remote_score_seconds = 0.0
    remote_load_seconds = 0.0
    for result in results:
        scored_rows.extend(result["rows"])
        n_scored += result["n_scored"]
        n_invalid += result["n_invalid"]
        remote_score_seconds += result["score_seconds"]
        remote_load_seconds += result["load_seconds"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(scored_rows).to_csv(output_path, index=False)

    rows_per_second = n_scored / elapsed if elapsed else float("nan")
    print(f"Wrote {len(scored_rows)} rows to {output_path}")
    print(
        "Summary: "
        f"n_scored={n_scored}, n_invalid={n_invalid}, wall_seconds={elapsed:.2f}, "
        f"rows_per_wall_second={rows_per_second:.2f}, "
        f"remote_load_seconds_sum={remote_load_seconds:.2f}, "
        f"remote_score_seconds_sum={remote_score_seconds:.2f}"
    )
