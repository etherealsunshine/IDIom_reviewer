from __future__ import annotations

from pathlib import Path

import pandas as pd


META_COLUMNS = {
    "sequence_id",
    "sequence",
    "source",
    "compartment_target",
    "original_sequence_id",
    "scramble_replicate",
}


def read_scores(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "sequence_id" not in df.columns:
        raise ValueError(f"{path} must contain sequence_id")
    return df


def score_columns(df: pd.DataFrame, prefix: str = "protgps_") -> list[str]:
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        numeric = df.select_dtypes("number").columns.tolist()
        cols = [c for c in numeric if c not in META_COLUMNS]
    return cols


def target_score_column(target: str, df: pd.DataFrame, prefix: str = "protgps_") -> str:
    normalized = target.replace("_", "-") if target == "p_body" else target
    candidates = [
        f"{prefix}{target}",
        f"{prefix}{target.replace('-', '_')}",
        f"{prefix}{target.replace('_', '-')}",
        f"{prefix}{normalized}",
        f"{normalized.upper()}_Score",
        f"{target.upper()}_Score",
        target,
        normalized,
    ]
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"No score column found for target {target!r}; tried {candidates}")


def merge_scores(metadata: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    drop = [c for c in scores.columns if c in metadata.columns and c != "sequence_id"]
    return metadata.merge(scores.drop(columns=drop), on="sequence_id", how="left", validate="one_to_one")


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
