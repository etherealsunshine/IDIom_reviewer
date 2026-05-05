from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .constants import AMINO_ACIDS, DEFAULT_SEED, SOURCE_CHEAP_PREFIX, TARGET_COMPARTMENTS

DEFAULT_IDR_WEIGHTS = {
    "A": 0.08,
    "C": 0.005,
    "D": 0.075,
    "E": 0.095,
    "F": 0.015,
    "G": 0.10,
    "H": 0.018,
    "I": 0.02,
    "K": 0.075,
    "L": 0.035,
    "M": 0.012,
    "N": 0.055,
    "P": 0.09,
    "Q": 0.055,
    "R": 0.055,
    "S": 0.105,
    "T": 0.065,
    "V": 0.025,
    "W": 0.003,
    "Y": 0.017,
}

MOTIFS = {
    "nucleolus": ["KKKRK", "PKKKRKV", "KRKR"],
    "chromosome": ["SP", "TP", "SPK", "SPE"],
    "p-body": ["RGG", "RGGG", "FGG", "YGG"],
    "stress_granule": ["RGG", "SYG"],
}


def frequencies_from_sequences(sequences: list[str]) -> dict[str, float]:
    counts = Counter()
    for seq in sequences:
        counts.update(seq)
    total = sum(counts[aa] for aa in AMINO_ACIDS)
    if total == 0:
        return DEFAULT_IDR_WEIGHTS.copy()
    return {aa: counts[aa] / total for aa in AMINO_ACIDS}


def _draw_background(rng: np.random.Generator, length: int, weights: dict[str, float]) -> list[str]:
    probs = np.array([weights.get(aa, 0.0) for aa in AMINO_ACIDS], dtype=float)
    probs = probs / probs.sum()
    return rng.choice(np.array(AMINO_ACIDS), size=length, p=probs).tolist()


def _insert(seq: list[str], motif: str, rng: np.random.Generator) -> None:
    if len(motif) >= len(seq):
        seq[: len(motif)] = list(motif[: len(seq)])
        return
    pos = int(rng.integers(0, len(seq) - len(motif) + 1))
    seq[pos : pos + len(motif)] = list(motif)


def _boost_fraction(seq: list[str], chars: str, target_fraction: float, rng: np.random.Generator) -> None:
    target = int(round(target_fraction * len(seq)))
    current = sum(aa in chars for aa in seq)
    if current >= target:
        return
    candidates = [i for i, aa in enumerate(seq) if aa not in chars]
    rng.shuffle(candidates)
    for idx in candidates[: target - current]:
        seq[idx] = chars[int(rng.integers(0, len(chars)))]


def motif_spiked_sequence(
    target: str,
    length: int,
    rng: np.random.Generator,
    weights: dict[str, float] | None = None,
) -> str:
    weights = weights or DEFAULT_IDR_WEIGHTS
    seq = _draw_background(rng, length, weights)
    if target == "nucleolus":
        for _ in range(int(rng.integers(3, 5))):
            _insert(seq, rng.choice(MOTIFS[target]).item(), rng)
        _boost_fraction(seq, "KR", 0.20, rng)
    elif target == "chromosome":
        for _ in range(int(rng.integers(8, 11))):
            _insert(seq, rng.choice(["SP", "TP"]).item(), rng)
        for _ in range(int(rng.integers(2, 4))):
            _insert(seq, rng.choice(["SPK", "SPE"]).item(), rng)
        _boost_fraction(seq, "ST", 0.20, rng)
    elif target == "p-body":
        for _ in range(int(rng.integers(5, 9))):
            _insert(seq, rng.choice(["RGG", "RGGG"]).item(), rng)
        for _ in range(int(rng.integers(2, 4))):
            _insert(seq, rng.choice(["FGG", "YGG"]).item(), rng)
    elif target == "stress_granule":
        for _ in range(int(rng.integers(4, 7))):
            _insert(seq, "RGG", rng)
        for _ in range(int(rng.integers(2, 4))):
            _insert(seq, "SYG", rng)
        _boost_fraction(seq, "G", 0.15, rng)
    else:
        raise ValueError(f"Unknown target: {target}")
    return "".join(seq)


def generate_motif_spiked(
    n: int = 1000,
    length: int = 100,
    seed: int = DEFAULT_SEED,
    weights: dict[str, float] | None = None,
    targets: tuple[str, ...] = TARGET_COMPARTMENTS,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for target in targets:
        for i in range(n):
            rows.append(
                {
                    "sequence_id": f"cheap_{target}_{i:05d}",
                    "sequence": motif_spiked_sequence(target, length, rng, weights),
                    "source": SOURCE_CHEAP_PREFIX + target,
                    "compartment_target": target,
                }
            )
    return pd.DataFrame(rows)
