from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .scoring_io import score_columns, target_score_column

sns.set_theme(style="whitegrid", context="talk")


def _save(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_scramble_scatter(paired_df: pd.DataFrame, out_path: str | Path) -> None:
    row = "scramble_type" if "scramble_type" in paired_df.columns and paired_df["scramble_type"].nunique() > 1 else None
    g = sns.FacetGrid(
        paired_df,
        row=row,
        col="compartment_target",
        col_wrap=None if row else 2,
        height=4,
        sharex=False,
        sharey=False,
    )
    g.map_dataframe(sns.scatterplot, x="original_score", y="scramble_score", alpha=0.25, s=14)
    for ax in g.axes.flat:
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    g.set_axis_labels("ProtGPS original", "ProtGPS scrambled")
    _save(out_path)


def plot_group_box(df: pd.DataFrame, target: str, out_path: str | Path) -> None:
    col = target_score_column(target, df)
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x="source", y=col, showfliers=False)
    sns.stripplot(data=df.sample(min(len(df), 3000), random_state=33402), x="source", y=col, color="0.2", alpha=0.15, size=2)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel(f"ProtGPS {target}")
    _save(out_path)


def plot_feature_importances(importances: pd.DataFrame, out_dir: str | Path, targets: list[str] | None = None) -> None:
    out_dir = Path(out_dir)
    targets = targets or sorted(importances["compartment"].unique())
    for target in targets:
        top = importances[importances["compartment"] == target].nlargest(10, "importance")
        if top.empty:
            continue
        plt.figure(figsize=(8, 5))
        sns.barplot(data=top, y="feature", x="importance", color="#4C78A8")
        plt.title(target)
        _save(out_dir / f"feature_importance_{target}.png")


def plot_predicted_vs_actual(predictions: pd.DataFrame, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    for pred_col in [c for c in predictions.columns if c.startswith("pred_protgps_")]:
        score = pred_col.replace("pred_", "")
        actual = f"actual_{score}"
        target = score.replace("protgps_", "")
        plt.figure(figsize=(5, 5))
        sns.scatterplot(data=predictions, x=actual, y=pred_col, hue="source", s=12, alpha=0.4, legend=False)
        lo = min(predictions[actual].min(), predictions[pred_col].min())
        hi = max(predictions[actual].max(), predictions[pred_col].max())
        plt.plot([lo, hi], [lo, hi], color="black", linewidth=1)
        plt.title(target)
        plt.xlabel("Actual ProtGPS")
        plt.ylabel("Predicted ProtGPS")
        _save(out_dir / f"predicted_vs_actual_{target}.png")


def plot_correlation_heatmap(corr: pd.DataFrame, out_path: str | Path) -> None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, vmin=-1, vmax=1, cmap="vlag", square=True)
    _save(out_path)


def plot_specificity(df: pd.DataFrame, out_path: str | Path) -> None:
    plt.figure(figsize=(11, 5))
    sns.boxplot(data=df, x="source", y="specificity", showfliers=False)
    plt.xticks(rotation=35, ha="right")
    _save(out_path)


def plot_mean_score_radarish(df: pd.DataFrame, out_path: str | Path) -> None:
    cols = score_columns(df)
    means = df.groupby("source")[cols].mean().reset_index().melt(id_vars="source", var_name="compartment", value_name="mean_score")
    means["compartment"] = means["compartment"].str.replace("protgps_", "", regex=False)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=means, x="compartment", y="mean_score", hue="source")
    plt.xticks(rotation=45, ha="right")
    _save(out_path)


def plot_target_condition_box(df: pd.DataFrame, target: str, condition_col: str, out_path: str | Path) -> None:
    col = target_score_column(target, df)
    sub = df[df["compartment_target"] == target].copy()
    if sub.empty:
        return
    plt.figure(figsize=(13, 5))
    sns.boxplot(data=sub, x=condition_col, y=col, showfliers=False)
    sample = sub.sample(min(len(sub), 4000), random_state=33402)
    sns.stripplot(data=sample, x=condition_col, y=col, color="0.2", alpha=0.12, size=1.8)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel(f"ProtGPS {target}")
    _save(out_path)


def plot_deeploc_barplot(summary: pd.DataFrame, out_path: str | Path) -> None:
    class_cols = [c for c in summary.columns if c.startswith("mean_")]
    plot_df = summary.melt(id_vars="source", value_vars=class_cols, var_name="class", value_name="mean_probability")
    plot_df["class"] = plot_df["class"].str.replace("mean_", "", regex=False)
    plt.figure(figsize=(14, 6))
    sns.barplot(data=plot_df, x="class", y="mean_probability", hue="source")
    plt.xticks(rotation=45, ha="right")
    _save(out_path)


def plot_confusion_matrix(cm, labels: list[str], out_path: str | Path) -> None:
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    _save(out_path)
