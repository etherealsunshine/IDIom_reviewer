from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, wilcoxon
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from .scoring_io import score_columns


def paired_scramble_stats(
    originals: pd.DataFrame,
    scrambles: pd.DataFrame,
    target: str,
    score_col: str,
    scramble_type: str | None = None,
) -> dict[str, float | str | int]:
    original_scores = originals.drop_duplicates("sequence_id").set_index("sequence_id")[score_col]
    scr = scrambles[scrambles["compartment_target"] == target].copy()
    if scramble_type is not None and "scramble_type" in scr.columns:
        scr = scr[scr["scramble_type"] == scramble_type].copy()
    paired = scr.join(original_scores.rename("original_score"), on="original_sequence_id")
    paired = paired.dropna(subset=["original_score", score_col])
    scramble_mean = paired.groupby("original_sequence_id")[score_col].mean()
    common_original = original_scores.reindex(scramble_mean.index).dropna()
    scramble_mean = scramble_mean.reindex(common_original.index)
    try:
        stat = wilcoxon(common_original.to_numpy(), scramble_mean.to_numpy(), zero_method="zsplit")
        p_value = float(stat.pvalue)
    except ValueError:
        p_value = np.nan
    return {
        "target": target,
        "scramble_type": scramble_type or "all",
        "n": int(len(common_original)),
        "original_mean": float(common_original.mean()),
        "original_std": float(common_original.std()),
        "original_median": float(common_original.median()),
        "scramble_mean": float(scramble_mean.mean()),
        "scramble_std": float(scramble_mean.std()),
        "scramble_median": float(scramble_mean.median()),
        "wilcoxon_p": p_value,
    }


def specificity(df: pd.DataFrame, target_col: str = "compartment_target") -> pd.DataFrame:
    cols = score_columns(df)
    out = df.copy()
    values = []
    for _, row in out.iterrows():
        row_dict = row.to_dict()
        target = row_dict.get(target_col)
        if pd.isna(target):
            values.append(np.nan)
            continue
        target_score_name = f"protgps_{target}"
        if target_score_name not in cols and target == "p_body":
            target_score_name = "protgps_p-body"
        if target_score_name not in cols:
            values.append(np.nan)
            continue
        others = [row_dict[c] for c in cols if c != target_score_name]
        values.append(row_dict[target_score_name] - max(others))
    out["specificity"] = values
    return out


def score_correlation(df: pd.DataFrame) -> pd.DataFrame:
    return df[score_columns(df)].corr(method="pearson")


def train_feature_probe(
    feature_df: pd.DataFrame,
    score_df: pd.DataFrame,
    feature_cols: list[str],
    stratify_col: str = "source",
    seed: int = 33402,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged = feature_df.merge(score_df, on="sequence_id", suffixes=("", "_score"))
    scores = score_columns(merged)
    if not scores:
        raise ValueError("No ProtGPS score columns found.")
    stratify = merged[stratify_col] if stratify_col in merged and merged[stratify_col].nunique() > 1 else None
    train_idx, test_idx = train_test_split(
        merged.index,
        test_size=0.2,
        random_state=seed,
        stratify=stratify,
    )
    x_train = merged.loc[train_idx, feature_cols]
    x_test = merged.loc[test_idx, feature_cols]
    metrics = []
    importances = []
    predictions = merged.loc[test_idx, ["sequence_id", "source"]].copy()
    for score_col in scores:
        y_train = merged.loc[train_idx, score_col]
        y_test = merged.loc[test_idx, score_col]
        model = RandomForestRegressor(
            n_estimators=500,
            max_depth=15,
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        metrics.append(
            {
                "compartment": score_col.replace("protgps_", ""),
                "score_col": score_col,
                "r2": float(r2_score(y_test, pred)),
                "mae": float(mean_absolute_error(y_test, pred)),
                "spearman_rho": float(spearmanr(y_test, pred).correlation),
            }
        )
        predictions[f"pred_{score_col}"] = pred
        predictions[f"actual_{score_col}"] = y_test.to_numpy()
        for name, value in zip(feature_cols, model.feature_importances_):
            importances.append(
                {
                    "compartment": score_col.replace("protgps_", ""),
                    "feature": name,
                    "importance": float(value),
                }
            )
    return pd.DataFrame(metrics), pd.DataFrame(importances), predictions


def pairwise_identity(a: str, b: str) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return sum(x == y for x, y in zip(a[:n], b[:n])) / n


def diversity_summary(df: pd.DataFrame, sample_n: int = 500, seed: int = 33402) -> pd.DataFrame:
    from .features import shannon_entropy

    rng = np.random.default_rng(seed)
    rows = []
    for source, group in df.groupby("source"):
        seqs = group["sequence"].astype(str).tolist()
        sample = seqs if len(seqs) <= sample_n else list(rng.choice(seqs, size=sample_n, replace=False))
        ids = []
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                ids.append(pairwise_identity(sample[i], sample[j]))
        rows.append(
            {
                "source": source,
                "n": len(seqs),
                "unique_fraction": len(set(seqs)) / len(seqs) if seqs else np.nan,
                "mean_sequence_entropy": float(np.mean([shannon_entropy(s) for s in seqs])) if seqs else np.nan,
                "mean_pairwise_identity": float(np.mean(ids)) if ids else np.nan,
                "std_pairwise_identity": float(np.std(ids)) if ids else np.nan,
                "p95_pairwise_identity": float(np.quantile(ids, 0.95)) if ids else np.nan,
            }
        )
    return pd.DataFrame(rows)


def cross_score_pearson(df: pd.DataFrame) -> pd.DataFrame:
    cols = score_columns(df)
    rows = []
    for a in cols:
        for b in cols:
            valid = df[[a, b]].dropna()
            r = pearsonr(valid[a], valid[b]).statistic if len(valid) > 1 else np.nan
            rows.append({"score_a": a, "score_b": b, "pearson_r": float(r)})
    return pd.DataFrame(rows)
