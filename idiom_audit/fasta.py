from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from .constants import AA_SET


@dataclass(frozen=True)
class SequenceRecord:
    sequence_id: str
    sequence: str
    source: str
    compartment_target: str | None = None


def clean_sequence(seq: str) -> str:
    seq = seq.upper().replace(" ", "").replace("*", "")
    return "".join(aa for aa in seq if aa in AA_SET)


def read_fasta(path: str | Path, source: str, target: str | None = None) -> Iterator[SequenceRecord]:
    path = Path(path)
    name: str | None = None
    chunks: list[str] = []
    with path.open() as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    seq = clean_sequence("".join(chunks))
                    if seq:
                        yield SequenceRecord(name, seq, source, target)
                name = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
    if name is not None:
        seq = clean_sequence("".join(chunks))
        if seq:
            yield SequenceRecord(name, seq, source, target)


def write_fasta(records: Iterable[SequenceRecord], path: str | Path, width: int = 80) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for rec in records:
            handle.write(f">{rec.sequence_id}\n")
            seq = rec.sequence
            for start in range(0, len(seq), width):
                handle.write(seq[start : start + width] + "\n")


def records_to_frame(records: Iterable[SequenceRecord]):
    import pandas as pd

    return pd.DataFrame([rec.__dict__ for rec in records])
