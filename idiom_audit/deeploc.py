from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

DEEPLOC_CLASSES = [
    "Nucleus",
    "Cytoplasm",
    "Extracellular",
    "Mitochondrion",
    "Cell membrane",
    "Endoplasmic reticulum",
    "Chloroplast",
    "Golgi apparatus",
    "Lysosome/Vacuole",
    "Peroxisome",
]


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


CLASS_ALIASES = {
    "Nucleus": {"nucleus", "nuclear"},
    "Cytoplasm": {"cytoplasm", "cytosol", "cytoplasmic"},
    "Extracellular": {"extracellular", "secreted"},
    "Mitochondrion": {"mitochondrion", "mitochondria", "mitochondrial"},
    "Cell membrane": {"cellmembrane", "membrane", "plasmamembrane"},
    "Endoplasmic reticulum": {"endoplasmicreticulum", "er"},
    "Chloroplast": {"chloroplast", "plastid"},
    "Golgi apparatus": {"golgiapparatus", "golgi"},
    "Lysosome/Vacuole": {"lysosomevacuole", "lysosome", "vacuole"},
    "Peroxisome": {"peroxisome"},
}

SIGNAL_ALIASES = {
    "NLS": {"nls", "nuclearlocalizationsignal", "nuclearlocalisationsignal"},
}


def find_class_columns(df: pd.DataFrame) -> dict[str, str]:
    normalized = {_norm(col): col for col in df.columns}
    out = {}
    for cls in DEEPLOC_CLASSES:
        aliases = {_norm(cls), *CLASS_ALIASES.get(cls, set())}
        for alias in aliases:
            if alias in normalized:
                out[cls] = normalized[alias]
                break
    if len(out) != len(DEEPLOC_CLASSES):
        missing = [cls for cls in DEEPLOC_CLASSES if cls not in out]
        raise ValueError(f"Could not identify DeepLoc class columns for {missing}; columns={df.columns.tolist()}")
    return out


def find_nls_column(df: pd.DataFrame) -> str | None:
    normalized = {_norm(col): col for col in df.columns}
    for alias in SIGNAL_ALIASES["NLS"]:
        if alias in normalized:
            return normalized[alias]
    candidates = [col for col in df.columns if "nls" in _norm(col)]
    return candidates[0] if candidates else None


def read_deeploc_outputs(results_dir: str | Path) -> pd.DataFrame:
    results_dir = Path(results_dir)
    frames = []
    for path in sorted(results_dir.rglob("*")):
        if path.suffix.lower() not in {".csv", ".tsv"}:
            continue
        try:
            df = pd.read_csv(path, sep=None, engine="python")
        except Exception:
            continue
        try:
            class_cols = find_class_columns(df)
        except ValueError:
            continue
        source = path.parent.name
        renamed = df.rename(columns={v: k for k, v in class_cols.items()}).copy()
        if "source" not in renamed.columns:
            renamed["source"] = source
        nls_col = find_nls_column(df)
        if nls_col and nls_col in df.columns:
            renamed["NLS"] = df[nls_col]
        frames.append(renamed)
    if not frames:
        raise FileNotFoundError(f"No parseable DeepLoc CSV/TSV outputs found under {results_dir}")
    return pd.concat(frames, ignore_index=True, sort=False)


def add_deeploc_top_class(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["deeploc_top_class"] = out[DEEPLOC_CLASSES].idxmax(axis=1)
    return out


def basic_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = add_deeploc_top_class(df)
    rows = []
    for source, group in df.groupby("source"):
        row = {"source": source, "n": len(group)}
        for cls in DEEPLOC_CLASSES:
            row[f"mean_{cls}"] = float(group[cls].mean())
        row["frac_top_Nucleus"] = float(group["deeploc_top_class"].eq("Nucleus").mean())
        row["frac_top_Cytoplasm"] = float(group["deeploc_top_class"].eq("Cytoplasm").mean())
        rows.append(row)
    return pd.DataFrame(rows)


def sorting_signal_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, group in df.groupby("source"):
        row = {"source": source, "n": len(group)}
        if "NLS" in group.columns:
            vals = group["NLS"]
            if vals.dtype == object:
                truth = vals.astype(str).str.lower().isin({"true", "1", "yes", "y", "nls"})
            else:
                truth = vals.astype(float) > 0.5
            row["frac_NLS"] = float(truth.mean())
        else:
            pred_cols = [c for c in group.columns if "Predicted" in c or "Signal" in c or "signal" in c]
            if pred_cols:
                signals = group[pred_cols].fillna("").map(str).agg(" ".join, axis=1)
                row["frac_NLS"] = float(signals.str.contains(r"\bNLS\b|nuclear localization signal", case=False, regex=True).mean())
            else:
                row["frac_NLS"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def pair_classifier(df: pd.DataFrame, source_a: str, source_b: str, label_a: str, label_b: str) -> dict:
    sub = df[df["source"].isin([source_a, source_b])].dropna(subset=DEEPLOC_CLASSES).copy()
    if sub.empty or sub["source"].nunique() != 2:
        raise ValueError(f"Need both sources {source_a}, {source_b}; available={sorted(df['source'].unique())}")
    x = sub[DEEPLOC_CLASSES].to_numpy(dtype=float)
    y = sub["source"].map({source_a: 0, source_b: 1}).to_numpy()
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    min_class_count = int(pd.Series(y).value_counts().min())
    if min_class_count < 2:
        raise ValueError(f"Need at least 2 examples per class; class_counts={pd.Series(y).value_counts().to_dict()}")
    cv = StratifiedKFold(n_splits=min(5, min_class_count), shuffle=True, random_state=33402)
    pred = cross_val_predict(model, x, y, cv=cv, method="predict")
    prob = cross_val_predict(model, x, y, cv=cv, method="predict_proba")[:, 1]
    cm = confusion_matrix(y, pred)
    means = sub.groupby("source")[DEEPLOC_CLASSES].mean()
    distance = float(np.linalg.norm(means.loc[source_a].to_numpy() - means.loc[source_b].to_numpy()))
    return {
        "source_a": source_a,
        "source_b": source_b,
        "label_a": label_a,
        "label_b": label_b,
        "n": int(len(sub)),
        "accuracy": float(accuracy_score(y, pred)),
        "auc": float(roc_auc_score(y, prob)),
        "euclidean_mean_vector_distance": distance,
        "confusion_matrix": cm,
    }
