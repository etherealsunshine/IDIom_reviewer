from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import DEFAULT_SEED, SOURCE_SCRAMBLED_PREFIX
from .fasta import SequenceRecord, read_fasta, write_fasta


def load_fasta_table(items: Iterable[tuple[str, str, str | None]]) -> pd.DataFrame:
    records: list[SequenceRecord] = []
    for path, source, target in items:
        records.extend(read_fasta(path, source=source, target=target))
    return pd.DataFrame([r.__dict__ for r in records])


def sample_by_source(df: pd.DataFrame, per_source: int | None, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    if per_source is None:
        return df.copy()
    rng = np.random.default_rng(seed)
    chunks = []
    for _, group in df.groupby("source", sort=False):
        if len(group) <= per_source:
            chunks.append(group)
        else:
            chunks.append(group.sample(per_source, random_state=int(rng.integers(0, 2**31 - 1))))
    return pd.concat(chunks, ignore_index=True)


def make_scrambles(
    df: pd.DataFrame,
    replicates: int = 5,
    seed: int = DEFAULT_SEED,
    sequence_col: str = "sequence",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for row in df.itertuples(index=False):
        row_dict = row._asdict()
        seq = row_dict[sequence_col]
        for rep in range(replicates):
            arr = np.array(list(seq))
            rng.shuffle(arr)
            out = dict(row_dict)
            out["sequence_id"] = f"{row_dict['sequence_id']}__scramble{rep + 1}"
            out["sequence"] = "".join(arr.tolist())
            out["source"] = SOURCE_SCRAMBLED_PREFIX + str(row_dict["source"])
            out["scramble_replicate"] = rep + 1
            out["original_sequence_id"] = row_dict["sequence_id"]
            rows.append(out)
    return pd.DataFrame(rows)


def write_scoring_fasta(df: pd.DataFrame, out_path: str | Path) -> None:
    records = [
        SequenceRecord(
            sequence_id=str(row.sequence_id),
            sequence=str(row.sequence),
            source=str(row.source),
            compartment_target=getattr(row, "compartment_target", None),
        )
        for row in df.itertuples(index=False)
    ]
    write_fasta(records, out_path)
