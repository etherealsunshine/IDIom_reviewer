#!/usr/bin/env python3
"""Pool-wise shallow-feature probe for the IDiom audit.

This reruns the Test 2 RandomForest analysis with whole sequence sources held
out, instead of doing a row-wise split where every source appears in train and
test. The goal is to test whether shallow sequence features generalize across
sequence pools.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


TARGET_SCORE_COLS = [
    "protgps_p-body",
    "protgps_stress_granule",
    "protgps_nucleolus",
    "protgps_chromosome",
]

RL_SOURCES = [
    "rl_nucleolus",
    "rl_chromosome",
    "rl_p-body",
    "rl_stress_granule",
]


def evaluate_split(
    merged: pd.DataFrame,
    feature_cols: list[str],
    split_name: str,
    train_mask: pd.Series,
    test_mask: pd.Series,
    score_cols: list[str],
    n_estimators: int,
    max_depth: int,
    seed: int,
) -> list[dict]:
    x_train = merged.loc[train_mask, feature_cols]
    x_test = merged.loc[test_mask, feature_cols]
    rows: list[dict] = []

    if x_train.empty or x_test.empty:
        print(f"Skipping {split_name}: empty train or test split")
        return rows

    train_sources = sorted(merged.loc[train_mask, "source"].unique())
    test_sources = sorted(merged.loc[test_mask, "source"].unique())
    print(f"{split_name}: train={len(x_train):,} test={len(x_test):,}")

    for score_col in score_cols:
        y_train = merged.loc[train_mask, score_col]
        y_test = merged.loc[test_mask, score_col]
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=seed,
            n_jobs=-1,
        )
        t0 = time.time()
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        rho = spearmanr(y_test, pred).correlation
        row = {
            "split_name": split_name,
            "train_sources": ",".join(train_sources),
            "test_sources": ",".join(test_sources),
            "n_train": int(len(x_train)),
            "n_test": int(len(x_test)),
            "compartment": score_col.replace("protgps_", ""),
            "score_col": score_col,
            "r2": float(r2_score(y_test, pred)),
            "mae": float(mean_absolute_error(y_test, pred)),
            "spearman_rho": float(rho) if rho == rho else np.nan,
            "train_score_min": float(y_train.min()),
            "train_score_max": float(y_train.max()),
            "test_score_min": float(y_test.min()),
            "test_score_max": float(y_test.max()),
            "seconds": round(time.time() - t0, 3),
        }
        rows.append(row)
        print(
            f"  {row['compartment']:<15} "
            f"R2={row['r2']:>7.3f}  rho={row['spearman_rho']:>6.3f}  "
            f"MAE={row['mae']:.3f}"
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/processed/test2_full_features.csv")
    parser.add_argument("--scores", default="data/processed/test2_full_pool_protgps_scores.csv")
    parser.add_argument("--out-dir", default="results/test2_feature_probe/test2_full")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=15)
    parser.add_argument("--seed", type=int, default=33402)
    args = parser.parse_args()

    features = pd.read_csv(args.features)
    scores = pd.read_csv(args.scores)
    merged = features.merge(scores, on="sequence_id", suffixes=("", "_score"))

    feature_cols = [
        c
        for c in features.columns
        if c not in {"sequence_id", "source", "compartment_target"}
    ]
    merged[feature_cols] = merged[feature_cols].fillna(0.0)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []

    for source in RL_SOURCES:
        all_rows.extend(
            evaluate_split(
                merged=merged,
                feature_cols=feature_cols,
                split_name=f"leave_one_rl_source_out:{source}",
                train_mask=merged["source"].ne(source),
                test_mask=merged["source"].eq(source),
                score_cols=TARGET_SCORE_COLS,
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                seed=args.seed,
            )
        )

    all_rows.extend(
        evaluate_split(
            merged=merged,
            feature_cols=feature_cols,
            split_name="train_non_rl_test_all_rl",
            train_mask=~merged["source"].isin(RL_SOURCES),
            test_mask=merged["source"].isin(RL_SOURCES),
            score_cols=TARGET_SCORE_COLS,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            seed=args.seed,
        )
    )

    metrics = pd.DataFrame(all_rows)
    out_path = out_dir / "feature_probe_poolwise_blog_relevant_metrics.csv"
    metrics.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")

    pivot = metrics.pivot_table(
        index="split_name",
        columns="compartment",
        values="r2",
        aggfunc="mean",
    )
    pivot_path = out_dir / "feature_probe_poolwise_r2_pivot.csv"
    pivot.to_csv(pivot_path)
    print(f"Wrote {pivot_path}")
    print("\nR2 pivot:")
    print(pivot.round(3).to_string())


if __name__ == "__main__":
    main()
