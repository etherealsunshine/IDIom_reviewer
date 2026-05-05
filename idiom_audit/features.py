from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import pandas as pd

from .constants import AMINO_ACIDS

HYDROPHOBIC = set("AVILMFW")
AROMATIC = set("FYW")
CHARGED = set("KRDE")


def _fraction(counter: Counter[str], chars: set[str] | str, length: int) -> float:
    if length == 0:
        return 0.0
    return sum(counter[c] for c in chars) / length


def shannon_entropy(seq: str) -> float:
    if not seq:
        return 0.0
    counts = Counter(seq)
    n = len(seq)
    return -sum((count / n) * math.log(count / n) for count in counts.values())


def sliding_window_entropy(seq: str, window: int = 12) -> float:
    if not seq:
        return 0.0
    if len(seq) <= window:
        return shannon_entropy(seq)
    vals = [shannon_entropy(seq[i : i + window]) for i in range(0, len(seq) - window + 1)]
    return float(np.mean(vals))


def sequence_features(seq: str) -> dict[str, float]:
    seq = seq.upper()
    n = len(seq)
    counts = Counter(seq)
    feats: dict[str, float] = {"length": float(n)}
    for aa in AMINO_ACIDS:
        feats[f"frac_{aa}"] = counts[aa] / n if n else 0.0
    feats["net_charge_per_residue"] = _fraction(counts, "KR", n) - _fraction(counts, "DE", n)
    feats["fcr"] = _fraction(counts, CHARGED, n)
    feats["frac_hydrophobic"] = _fraction(counts, HYDROPHOBIC, n)
    feats["frac_aromatic"] = _fraction(counts, AROMATIC, n)
    feats["frac_glycine"] = counts["G"] / n if n else 0.0
    feats["frac_proline"] = counts["P"] / n if n else 0.0
    feats["count_RGG"] = float(len(re.findall(r"RGG", seq)))
    feats["count_FYGG"] = float(len(re.findall(r"[FY]GG", seq)))
    feats["count_SYG"] = float(len(re.findall(r"SYG", seq)))
    feats["count_basic_clusters"] = float(len(re.findall(r"[KR]{3,}", seq)))
    feats["count_SP_TP"] = float(len(re.findall(r"SP|TP", seq)))
    feats["entropy"] = shannon_entropy(seq)
    feats["low_complexity_entropy_w12"] = sliding_window_entropy(seq, 12)
    return feats


def featurize_frame(df: pd.DataFrame, sequence_col: str = "sequence") -> pd.DataFrame:
    features = pd.DataFrame([sequence_features(seq) for seq in df[sequence_col].astype(str)])
    meta_cols = [c for c in ("sequence_id", "source", "compartment_target") if c in df.columns]
    return pd.concat([df[meta_cols].reset_index(drop=True), features], axis=1)
