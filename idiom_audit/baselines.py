from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd

from .constants import AMINO_ACIDS, DEFAULT_SEED, TARGET_COMPARTMENTS

TARGET_MOTIFS = {
    "nucleolus": {
        "motifs": ("KKKRK", "PKKKRKV", "KRKR"),
        "regexes": (r"KKKRK", r"PKKKRKV", r"KRKR", r"[KR]{3,}"),
    },
    "chromosome": {
        "motifs": ("SP", "TP", "SPK", "SPE"),
        "regexes": (r"SP", r"TP", r"SPK", r"SPE"),
    },
    "p-body": {
        "motifs": ("RGG", "RGGG", "FGG", "YGG"),
        "regexes": (r"RGG", r"RGGG", r"[FY]GG"),
    },
    "stress_granule": {
        "motifs": ("RGG", "SYG"),
        "regexes": (r"RGG", r"SYG"),
    },
}


def aa_frequencies(sequences: list[str]) -> dict[str, float]:
    counts = Counter()
    for seq in sequences:
        counts.update(aa for aa in seq if aa in AMINO_ACIDS)
    total = sum(counts.values())
    if total == 0:
        return {aa: 1.0 / len(AMINO_ACIDS) for aa in AMINO_ACIDS}
    return {aa: counts[aa] / total for aa in AMINO_ACIDS}


def sample_iid_sequence(length: int, freqs: dict[str, float], rng: np.random.Generator) -> str:
    probs = np.array([freqs.get(aa, 0.0) for aa in AMINO_ACIDS], dtype=float)
    probs = probs / probs.sum()
    return "".join(rng.choice(np.array(AMINO_ACIDS), size=length, p=probs).tolist())


def motif_count(seq: str, target: str) -> int:
    regexes = TARGET_MOTIFS[target]["regexes"]
    return int(sum(len(re.findall(pattern, seq)) for pattern in regexes))


def motif_choice_weights(sequences: list[str], target: str) -> dict[str, float]:
    motifs = TARGET_MOTIFS[target]["motifs"]
    counts = Counter()
    for seq in sequences:
        for motif in motifs:
            counts[motif] += seq.count(motif)
    total = sum(counts.values())
    if total == 0:
        return {motif: 1.0 / len(motifs) for motif in motifs}
    return {motif: counts[motif] / total for motif in motifs}


def insert_motif(seq: str, motif: str, rng: np.random.Generator) -> str:
    if not seq:
        return motif
    if len(motif) >= len(seq):
        return motif[: len(seq)]
    pos = int(rng.integers(0, len(seq) - len(motif) + 1))
    return seq[:pos] + motif + seq[pos + len(motif) :]


def motif_calibrated_sequence(
    length: int,
    freqs: dict[str, float],
    target: str,
    motif_events: int,
    motif_weights: dict[str, float],
    rng: np.random.Generator,
) -> str:
    seq = sample_iid_sequence(length, freqs, rng)
    motifs = np.array(list(motif_weights.keys()))
    probs = np.array([motif_weights[motif] for motif in motifs], dtype=float)
    probs = probs / probs.sum()
    for motif in rng.choice(motifs, size=max(0, motif_events), p=probs):
        seq = insert_motif(seq, str(motif), rng)
    return seq


def aromatic_spaced_sequence(length: int, freqs: dict[str, float], rng: np.random.Generator) -> str:
    seq = sample_iid_sequence(length, freqs, rng)
    aromatics = [aa for aa in seq if aa in "FYW"]
    others = [aa for aa in seq if aa not in "FYW"]
    if len(aromatics) <= 1:
        return seq

    out: list[str | None] = [None] * length
    gap = int(rng.integers(8, 11))
    offset = int(rng.integers(0, min(gap, length)))
    positions = list(range(offset, length, gap))
    if len(positions) < len(aromatics):
        free = [idx for idx in range(length) if idx not in set(positions)]
        if free:
            extra_idx = np.linspace(0, len(free) - 1, len(aromatics) - len(positions), dtype=int)
            positions.extend([free[i] for i in extra_idx])
    positions = sorted(set(positions))[: len(aromatics)]
    rng.shuffle(aromatics)
    for pos, aa in zip(positions, aromatics):
        out[pos] = aa
    rng.shuffle(others)
    other_iter = iter(others)
    for idx, aa in enumerate(out):
        if aa is None:
            out[idx] = next(other_iter)
    return "".join(aa for aa in out if aa is not None)


def summarize_rl_motifs(rl_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in TARGET_COMPARTMENTS:
        group = rl_df[rl_df["compartment_target"] == target]
        counts = [motif_count(seq, target) for seq in group["sequence"].astype(str)]
        if not counts:
            continue
        rows.append(
            {
                "target": target,
                "n": len(counts),
                "motif_count_mean": float(np.mean(counts)),
                "motif_count_std": float(np.std(counts)),
                "motif_count_median": float(np.median(counts)),
                "motif_count_p95": float(np.quantile(counts, 0.95)),
            }
        )
    return pd.DataFrame(rows)


def generate_amended_baselines(
    scored_originals: pd.DataFrame,
    n_per_target: int = 1000,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rl_df = scored_originals[
        scored_originals["compartment_target"].isin(TARGET_COMPARTMENTS)
        & scored_originals["source"].astype(str).str.startswith("rl_")
    ].copy()
    motif_summary = summarize_rl_motifs(rl_df)
    rows = []
    for target in TARGET_COMPARTMENTS:
        group = rl_df[rl_df["compartment_target"] == target]
        seqs = group["sequence"].astype(str).tolist()
        if not seqs:
            continue
        freqs = aa_frequencies(seqs)
        lengths = np.array([len(seq) for seq in seqs], dtype=int)
        motif_counts = np.array([motif_count(seq, target) for seq in seqs], dtype=int)
        motif_weights = motif_choice_weights(seqs, target)

        for i in range(n_per_target):
            length = int(rng.choice(lengths))
            rows.append(
                {
                    "sequence_id": f"comp_random_{target}_{i:05d}",
                    "sequence": sample_iid_sequence(length, freqs, rng),
                    "source": f"comp_random_{target}",
                    "compartment_target": target,
                    "baseline_type": "composition_matched_random",
                }
            )

        for i in range(n_per_target):
            length = int(rng.choice(lengths))
            events = int(rng.choice(motif_counts)) if len(motif_counts) else 0
            rows.append(
                {
                    "sequence_id": f"motif_calibrated_{target}_{i:05d}",
                    "sequence": motif_calibrated_sequence(length, freqs, target, events, motif_weights, rng),
                    "source": f"motif_calibrated_{target}",
                    "compartment_target": target,
                    "baseline_type": "motif_calibrated_random",
                    "sampled_motif_events": events,
                }
            )

        if target == "p-body":
            for i in range(n_per_target):
                length = int(rng.choice(lengths))
                rows.append(
                    {
                        "sequence_id": f"pbody_aromatic_spaced_{i:05d}",
                        "sequence": aromatic_spaced_sequence(length, freqs, rng),
                        "source": "pbody_aromatic_spaced",
                        "compartment_target": target,
                        "baseline_type": "pbody_aromatic_spaced",
                    }
                )

    return pd.DataFrame(rows), motif_summary
