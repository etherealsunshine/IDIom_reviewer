from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .constants import DEFAULT_SEED, TARGET_COMPARTMENTS
from .baselines import generate_amended_baselines
from .deeploc import basic_summary, pair_classifier, read_deeploc_outputs, sorting_signal_summary
from .datasets import load_fasta_table, make_scrambles, write_scoring_fasta
from .features import featurize_frame
from .scoring_io import read_scores, save_csv, score_columns, target_score_column
from .stats import diversity_summary, paired_scramble_stats, score_correlation, specificity, train_feature_probe
from .synthetic import generate_motif_spiked


def _fasta_items(values: list[str]) -> list[tuple[str, str, str | None]]:
    items = []
    for value in values:
        parts = value.split(":", 2)
        if len(parts) == 2:
            path, source = parts
            target = None
        elif len(parts) == 3:
            path, source, target = parts
        else:
            raise SystemExit("--fasta entries must be PATH:SOURCE or PATH:SOURCE:TARGET")
        items.append((path, source, target))
    return items


def cmd_load_fastas(args: argparse.Namespace) -> None:
    df = load_fasta_table(_fasta_items(args.fasta))
    save_csv(df, args.output)


def cmd_make_scrambles(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.input)
    scr = make_scrambles(df, replicates=args.replicates, seed=args.seed)
    save_csv(scr, args.output)
    if args.fasta_output:
        write_scoring_fasta(scr, args.fasta_output)


def cmd_featurize(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.input)
    save_csv(featurize_frame(df), args.output)


def cmd_cheap_baselines(args: argparse.Namespace) -> None:
    weights = None
    if args.frequency_csv:
        freq_df = pd.read_csv(args.frequency_csv)
        weights = dict(zip(freq_df["aa"], freq_df["frequency"]))
    df = generate_motif_spiked(n=args.n, length=args.length, seed=args.seed, weights=weights)
    save_csv(df, args.output)
    if args.fasta_output:
        write_scoring_fasta(df, args.fasta_output)


def cmd_amended_baselines(args: argparse.Namespace) -> None:
    originals = read_scores(args.originals)
    baselines, motif_summary = generate_amended_baselines(
        originals,
        n_per_target=args.n_per_target,
        seed=args.seed,
    )
    save_csv(baselines, args.output)
    save_csv(motif_summary, args.motif_summary_output)
    if args.fasta_output:
        write_scoring_fasta(baselines, args.fasta_output)


def _condition_label(df: pd.DataFrame) -> pd.Series:
    source = df["source"].astype(str)
    condition = pd.Series(source, index=df.index, dtype="object")
    condition[source.eq("base_idp")] = "Base IDPs"
    condition[source.str.startswith("rl_")] = "RL original"
    if "scramble_type" in df.columns:
        condition[df["scramble_type"].astype(str).eq("block")] = "RL block-shuffled"
        condition[df["scramble_type"].astype(str).eq("full")] = "RL fully scrambled"
    condition[source.str.startswith("comp_random_")] = "Composition-matched random"
    condition[source.str.startswith("motif_calibrated_")] = "Motif-calibrated random"
    condition[source.eq("pbody_aromatic_spaced")] = "P-body aromatic-spaced"
    condition[source.str.contains("disprot", case=False, na=False)] = "DisProt natural"
    return condition


def cmd_amended_baseline_compare(args: argparse.Namespace) -> None:
    from .plots import plot_target_condition_box

    out_dir = Path(args.out_dir)
    frames = []
    for path in args.scores:
        frames.append(read_scores(path))
    df = pd.concat(frames, ignore_index=True, sort=False)
    df["condition"] = _condition_label(df)

    rows = []
    for target in TARGET_COMPARTMENTS:
        try:
            score_col = target_score_column(target, df)
        except KeyError:
            continue
        sub = df[df["compartment_target"].eq(target) | df["source"].eq("base_idp")].copy()
        for condition, group in sub.groupby("condition", dropna=False):
            vals = group[score_col].dropna()
            if vals.empty:
                continue
            rows.append(
                {
                    "target": target,
                    "condition": condition,
                    "n": len(vals),
                    "mean": float(vals.mean()),
                    "std": float(vals.std()),
                    "median": float(vals.median()),
                    "p25": float(vals.quantile(0.25)),
                    "p75": float(vals.quantile(0.75)),
                }
            )
        plot_target_condition_box(sub, target, "condition", out_dir / f"amended_baseline_box_{target}.png")

    summary = pd.DataFrame(rows)
    rl_means = summary[summary["condition"].eq("RL original")][["target", "mean"]].rename(columns={"mean": "rl_mean"})
    summary = summary.merge(rl_means, on="target", how="left")
    summary["gap_vs_rl"] = summary["rl_mean"] - summary["mean"]
    summary["threshold_call"] = ""
    trivial = summary["condition"].isin(["Composition-matched random", "Motif-calibrated random", "P-body aromatic-spaced"])
    summary.loc[trivial & summary["gap_vs_rl"].le(0.05), "threshold_call"] = "within_0.05_trivial_baseline"
    summary.loc[trivial & summary["gap_vs_rl"].gt(0.15), "threshold_call"] = "gap_gt_0.15_nontrivial"
    summary.loc[trivial & summary["gap_vs_rl"].gt(0.05) & summary["gap_vs_rl"].le(0.15), "threshold_call"] = "ambiguous"
    save_csv(summary, out_dir / "amended_baseline_summary.csv")


def cmd_deeploc_analyze(args: argparse.Namespace) -> None:
    from .plots import plot_confusion_matrix, plot_deeploc_barplot

    out_dir = Path(args.out_dir)
    df = read_deeploc_outputs(args.results_dir)
    save_csv(df, out_dir / "deeploc_all_predictions.csv")

    summary = basic_summary(df)
    save_csv(summary, out_dir / "deeploc_summary_table.csv")
    plot_deeploc_barplot(summary, out_dir / "deeploc_barplot.png")

    signals = sorting_signal_summary(df)
    save_csv(signals, out_dir / "deeploc_sorting_signals_summary.csv")

    pairs = [
        ("rl_p-body", "rl_stress_granule", "pbody", "stress_granule", "deeploc_pbody_sg_confusion.png", "deeploc_pbody_vs_sg_classifier_accuracy.txt"),
        ("rl_nucleolus", "rl_chromosome", "nucleolus", "chromosome", "deeploc_nucleolus_chromosome_confusion.png", "deeploc_nucleolus_vs_chromosome_classifier_accuracy.txt"),
    ]
    rows = []
    for source_a, source_b, label_a, label_b, png_name, txt_name in pairs:
        try:
            result = pair_classifier(df, source_a, source_b, label_a, label_b)
        except ValueError as exc:
            (out_dir / txt_name).parent.mkdir(parents=True, exist_ok=True)
            (out_dir / txt_name).write_text(str(exc) + "\n")
            continue
        rows.append({k: v for k, v in result.items() if k != "confusion_matrix"})
        plot_confusion_matrix(result["confusion_matrix"], [label_a, label_b], out_dir / png_name)
        (out_dir / txt_name).parent.mkdir(parents=True, exist_ok=True)
        (out_dir / txt_name).write_text(
            "\n".join(
                [
                    f"source_a={source_a}",
                    f"source_b={source_b}",
                    f"accuracy={result['accuracy']:.4f}",
                    f"auc={result['auc']:.4f}",
                    f"euclidean_mean_vector_distance={result['euclidean_mean_vector_distance']:.6f}",
                    f"confusion_matrix={result['confusion_matrix'].tolist()}",
                ]
            )
            + "\n"
        )
    if rows:
        save_csv(pd.DataFrame(rows), out_dir / "deeploc_pair_classifier_summary.csv")


def _paired_scramble_long(originals: pd.DataFrame, scrambles: pd.DataFrame) -> pd.DataFrame:
    if "scramble_type" not in scrambles.columns:
        scrambles = scrambles.copy()
        scrambles["scramble_type"] = "full"
    rows = []
    for target in sorted(originals["compartment_target"].dropna().unique()):
        score_col = target_score_column(target, originals)
        original_scores = (
            originals[originals["compartment_target"] == target]
            .drop_duplicates("sequence_id")
            .set_index("sequence_id")[score_col]
        )
        sub = scrambles[scrambles["compartment_target"] == target]
        for _, row in sub.iterrows():
            row_dict = row.to_dict()
            original_score = original_scores.get(row_dict["original_sequence_id"])
            if pd.notna(original_score) and pd.notna(row_dict.get(score_col)):
                rows.append(
                    {
                        "compartment_target": target,
                        "scramble_type": row_dict.get("scramble_type", "full"),
                        "original_sequence_id": row_dict["original_sequence_id"],
                        "original_score": original_score,
                        "scramble_score": row_dict[score_col],
                    }
                )
    return pd.DataFrame(rows)


def cmd_test1(args: argparse.Namespace) -> None:
    from .plots import plot_group_box, plot_scramble_scatter

    out_dir = Path(args.out_dir)
    originals = read_scores(args.originals)
    scrambles = read_scores(args.scrambles)
    if "scramble_type" not in scrambles.columns:
        scrambles = scrambles.copy()
        scrambles["scramble_type"] = "full"
    rows = []
    for target in sorted(originals["compartment_target"].dropna().unique()):
        col = target_score_column(target, originals)
        for scramble_type in sorted(scrambles["scramble_type"].dropna().unique()):
            rows.append(
                paired_scramble_stats(
                    originals[originals["compartment_target"] == target],
                    scrambles,
                    target,
                    col,
                    scramble_type=scramble_type,
                )
            )
    stats = pd.DataFrame(rows)
    save_csv(stats, out_dir / "composition_matched_scramble_stats.csv")
    paired = _paired_scramble_long(originals, scrambles)
    save_csv(paired, out_dir / "composition_matched_scramble_pairs.csv")
    plot_scramble_scatter(paired, out_dir / "scramble_original_vs_scrambled.png")
    if args.groups:
        groups = pd.concat([read_scores(p) for p in args.groups], ignore_index=True)
        for target in TARGET_COMPARTMENTS:
            try:
                plot_group_box(groups, target, out_dir / f"box_target_score_{target}.png")
            except KeyError:
                continue


def cmd_test2(args: argparse.Namespace) -> None:
    from .plots import plot_feature_importances, plot_predicted_vs_actual

    out_dir = Path(args.out_dir)
    features = pd.read_csv(args.features)
    scores = read_scores(args.scores)
    feature_cols = [c for c in features.columns if c not in {"sequence_id", "source", "compartment_target"}]
    metrics, importances, predictions = train_feature_probe(features, scores, feature_cols, seed=args.seed)
    save_csv(metrics, out_dir / "feature_probe_metrics.csv")
    save_csv(importances, out_dir / "feature_probe_importances.csv")
    save_csv(predictions, out_dir / "feature_probe_predictions.csv")
    plot_feature_importances(importances, out_dir)
    plot_predicted_vs_actual(predictions, out_dir)


def cmd_test4(args: argparse.Namespace) -> None:
    from .plots import plot_correlation_heatmap, plot_mean_score_radarish, plot_specificity

    out_dir = Path(args.out_dir)
    df = read_scores(args.scores)
    spec = specificity(df)
    save_csv(spec, out_dir / "specificity_scores.csv")
    corr = score_correlation(df)
    save_csv(corr.reset_index(names="score"), out_dir / "cross_compartment_score_correlation.csv")
    plot_specificity(spec, out_dir / "specificity_boxplot.png")
    plot_correlation_heatmap(corr, out_dir / "cross_compartment_score_correlation.png")
    plot_mean_score_radarish(df, out_dir / "mean_scores_by_source.png")


def cmd_test7(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.input)
    save_csv(diversity_summary(df, sample_n=args.sample_n, seed=args.seed), Path(args.out_dir) / "diversity_summary.csv")


def cmd_write_scoring_fasta(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.input)
    write_scoring_fasta(df, args.output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="idiom-audit")
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("load-fastas")
    p.add_argument("--fasta", action="append", required=True, help="PATH:SOURCE or PATH:SOURCE:TARGET")
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_load_fastas)

    p = sub.add_parser("make-scrambles")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--fasta-output")
    p.add_argument("--replicates", type=int, default=5)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.set_defaults(func=cmd_make_scrambles)

    p = sub.add_parser("featurize")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_featurize)

    p = sub.add_parser("cheap-baselines")
    p.add_argument("--output", required=True)
    p.add_argument("--fasta-output")
    p.add_argument("--frequency-csv")
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--length", type=int, default=100)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.set_defaults(func=cmd_cheap_baselines)

    p = sub.add_parser("amended-baselines")
    p.add_argument("--originals", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--motif-summary-output", required=True)
    p.add_argument("--fasta-output")
    p.add_argument("--n-per-target", type=int, default=1000)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.set_defaults(func=cmd_amended_baselines)

    p = sub.add_parser("amended-baseline-compare")
    p.add_argument("--scores", nargs="+", required=True)
    p.add_argument("--out-dir", default="results/amended_baselines")
    p.set_defaults(func=cmd_amended_baseline_compare)

    p = sub.add_parser("deeploc-analyze")
    p.add_argument("--results-dir", required=True)
    p.add_argument("--out-dir", default="results/deeploc_validation")
    p.set_defaults(func=cmd_deeploc_analyze)

    p = sub.add_parser("write-scoring-fasta")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_write_scoring_fasta)

    p = sub.add_parser("test1")
    p.add_argument("--originals", required=True)
    p.add_argument("--scrambles", required=True)
    p.add_argument("--groups", nargs="*", help="Optional score CSVs for boxplots, e.g. base/original/scrambled.")
    p.add_argument("--out-dir", default="results/test1_scramble")
    p.set_defaults(func=cmd_test1)

    p = sub.add_parser("test2")
    p.add_argument("--features", required=True)
    p.add_argument("--scores", required=True)
    p.add_argument("--out-dir", default="results/test2_feature_probe")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.set_defaults(func=cmd_test2)

    p = sub.add_parser("test4")
    p.add_argument("--scores", required=True)
    p.add_argument("--out-dir", default="results/test4_specificity")
    p.set_defaults(func=cmd_test4)

    p = sub.add_parser("test7")
    p.add_argument("--input", required=True)
    p.add_argument("--out-dir", default="results/test7_diversity")
    p.add_argument("--sample-n", type=int, default=500)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.set_defaults(func=cmd_test7)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
