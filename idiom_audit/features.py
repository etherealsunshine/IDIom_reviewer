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


def _positions_regex(seq: str, pattern: str) -> list[int]:
    return [match.start() for match in re.finditer(pattern, seq)]


def _positions_chars(seq: str, chars: set[str]) -> list[int]:
    return [idx for idx, aa in enumerate(seq) if aa in chars]


def _gap_features(positions: list[int], prefix: str) -> dict[str, float]:
    if len(positions) < 2:
        return {
            f"{prefix}_mean_gap": 0.0,
            f"{prefix}_std_gap": 0.0,
            f"{prefix}_min_gap": 0.0,
            f"{prefix}_max_gap": 0.0,
            f"{prefix}_gap_cv": 0.0,
            f"{prefix}_evenness_score": 0.0,
        }
    gaps = np.diff(sorted(positions)).astype(float)
    mean_gap = float(np.mean(gaps))
    std_gap = float(np.std(gaps))
    return {
        f"{prefix}_mean_gap": mean_gap,
        f"{prefix}_std_gap": std_gap,
        f"{prefix}_min_gap": float(np.min(gaps)),
        f"{prefix}_max_gap": float(np.max(gaps)),
        f"{prefix}_gap_cv": std_gap / mean_gap if mean_gap else 0.0,
        f"{prefix}_evenness_score": 1.0 / (1.0 + (std_gap / mean_gap if mean_gap else 0.0)),
    }


def _max_regex_match_len(seq: str, pattern: str) -> int:
    matches = [match.group(0) for match in re.finditer(pattern, seq)]
    if not matches:
        return 0
    return max(len(match) for match in matches)


def _per_100(count: float, length: int) -> float:
    return (count / length) * 100.0 if length else 0.0


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
    feats["RGG_per_100"] = _per_100(feats["count_RGG"], n)
    feats["FYGG_per_100"] = _per_100(feats["count_FYGG"], n)
    feats["SYG_per_100"] = _per_100(feats["count_SYG"], n)
    feats["basic_cluster_per_100"] = _per_100(feats["count_basic_clusters"], n)
    feats["SP_TP_per_100"] = _per_100(feats["count_SP_TP"], n)
    feats["max_basic_cluster_len"] = float(_max_regex_match_len(seq, r"[KR]{3,}"))
    feats["longest_same_aa_run"] = float(_max_regex_match_len(seq, r"([A-Z])\1*"))
    feats.update(_gap_features(_positions_chars(seq, AROMATIC), "aromatic"))
    feats.update(_gap_features(_positions_regex(seq, r"RGG"), "RGG"))
    feats.update(_gap_features(_positions_regex(seq, r"SP|TP"), "SP_TP"))
    feats["entropy"] = shannon_entropy(seq)
    feats["low_complexity_entropy_w12"] = sliding_window_entropy(seq, 12)
    return feats


def featurize_frame(df: pd.DataFrame, sequence_col: str = "sequence") -> pd.DataFrame:
    features = pd.DataFrame([sequence_features(seq) for seq in df[sequence_col].astype(str)])
    meta_cols = [c for c in ("sequence_id", "source", "compartment_target") if c in df.columns]
    return pd.concat([df[meta_cols].reset_index(drop=True), features], axis=1)
