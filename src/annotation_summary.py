#!/usr/bin/env python
"""
Summary visualizations for:
  1. EC annotation of unannotated reactions
  2. Correction of misannotated reactions

Usage:
    conda activate rxnfp
    python -m src.annotation_summary
"""

import os
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.config import (
    ANNOTATION_SUMMARY_PNG,
    EC_PREDICTIONS_CSV,
    MISANNOTATION_CORRECTION_PNG,
    MISANNOTATION_CSV,
    RXN_DATA_CSV,
)
from src.utils import parse_ec_numbers

EC1_NAMES = {
    "1": "Oxidoreductases",
    "2": "Transferases",
    "3": "Hydrolases",
    "4": "Lyases",
    "5": "Isomerases",
    "6": "Ligases",
    "7": "Translocases",
}

COL_V, COL_P = "#4C72B0", "#DD8452"
COL_G = "#55A868"


def _style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=9)


def make_annotation_figure(df_rxn, df_u, output_path):
    """4-panel figure summarizing EC annotation results."""
    ec_lists = [parse_ec_numbers(v) for v in df_rxn["ec_numbers"].values]
    total_rxns = len(df_rxn)
    n_annotated = sum(1 for e in ec_lists if e)
    n_unannotated = total_rxns - n_annotated

    has_pf = df_u[
        df_u["prefilter_predicted_ec"].notna()
        & (df_u["prefilter_predicted_ec"] != "None")
        & (df_u["prefilter_n_annotated_neighbours"] >= 3)
    ].copy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("EC Number Annotation of Unannotated Reactions", fontsize=15, fontweight="bold", y=0.98)

    # Panel 1: Annotation funnel
    ax = axes[0, 0]
    _style_ax(ax)
    n_with_center = len(df_u)
    n_predicted = len(has_pf[has_pf["prefilter_confidence"] >= 0.6])
    n_high_80 = len(has_pf[has_pf["prefilter_confidence"] >= 0.8])
    n_high_90 = len(has_pf[has_pf["prefilter_confidence"] >= 0.9])
    categories = [f"Total\nunannotated", f"With reaction\ncenter", f"Predicted\n(n>=3, agreement>=60%)", f"Very high agreement\n(>=80%)", f"Highest agreement\n(>=90%)"]
    values = [n_unannotated, n_with_center, n_predicted, n_high_80, n_high_90]
    colors = ["#cccccc", "#aaaaaa", COL_G, "#2d8632", "#1a6621"]
    bars = ax.barh(range(len(categories)), values, color=colors, edgecolor="white", linewidth=0.5, height=0.6)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Number of reactions", fontsize=10)
    ax.set_title("Annotation Funnel", fontsize=12, fontweight="bold", pad=10)
    for i, (v, bar) in enumerate(zip(values, bars)):
        pct = 100 * v / n_unannotated
        ax.text(v + total_rxns * 0.01, i, f"{v:,} ({pct:.1f}%)", ha="left", va="center", fontsize=9, color="#333333")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    # Panel 2: Before/after annotation coverage by EC class
    ax = axes[0, 1]
    _style_ax(ax)
    ec1_existing = Counter()
    for ecs in ec_lists:
        for ec in ecs:
            ec1_existing[ec.split(".")[0]] += 1
    high_pf = has_pf[has_pf["prefilter_confidence"] >= 0.6]
    ec1_new = Counter()
    for ec in high_pf["prefilter_predicted_ec"]:
        c = str(ec).split(".")[0]
        if c.isdigit():
            ec1_new[c] += 1
    all_classes = sorted(set(list(ec1_existing.keys()) + list(ec1_new.keys())))
    x = np.arange(len(all_classes))
    w = 0.35
    existing_counts = [ec1_existing.get(c, 0) for c in all_classes]
    new_counts = [ec1_new.get(c, 0) for c in all_classes]
    ax.bar(x - w / 2, existing_counts, w, label="Existing annotations", color=COL_V, edgecolor="white", linewidth=0.5)
    ax.bar(x + w / 2, new_counts, w, label="New predictions (>=60%)", color=COL_G, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"EC {c}\n{EC1_NAMES.get(c, '')}" for c in all_classes], fontsize=8)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Existing Annotations vs New Predictions\nby EC Class", fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=9, frameon=False)
    for i in range(len(all_classes)):
        ax.text(x[i] - w / 2, existing_counts[i] + 30, f"{existing_counts[i]:,}", ha="center", va="bottom", fontsize=7, color="#333333")
        if new_counts[i] > 0:
            ax.text(x[i] + w / 2, new_counts[i] + 30, f"{new_counts[i]:,}", ha="center", va="bottom", fontsize=7, color="#333333")

    # Panel 3: Confidence tier breakdown
    ax = axes[1, 0]
    _style_ax(ax)
    tiers = [("60-65%", (0.6, 0.65)), ("65-70%", (0.65, 0.7)), ("70-75%", (0.7, 0.75)), ("75-80%", (0.75, 0.8)), ("80-85%", (0.8, 0.85)), ("85-90%", (0.85, 0.9)), ("90-95%", (0.9, 0.95)), ("95-100%", (0.95, 1.01))]
    tier_counts = []
    for label, (lo, hi) in tiers:
        cnt = len(has_pf[(has_pf["prefilter_confidence"] >= lo) & (has_pf["prefilter_confidence"] < hi)])
        tier_counts.append(cnt)
    tier_colors = [COL_P, COL_P, COL_P, COL_P, COL_G, COL_G, "#2d8632", "#1a6621"]
    bars = ax.bar([t[0] for t in tiers], tier_counts, color=tier_colors, edgecolor="white", linewidth=0.5, width=0.6)
    ax.set_xlabel("% Neighbor Agreement", fontsize=10)
    ax.set_ylabel("Number of predictions", fontsize=10)
    ax.set_title("Predicted Reactions by Agreement Level\n(prefilter, n>=3, agreement>=60%)", fontsize=12, fontweight="bold", pad=10)
    for bar, cnt in zip(bars, tier_counts):
        pct = 100 * cnt / n_predicted if n_predicted > 0 else 0
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20, f"{cnt:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=8, color="#333333")

    # Panel 4: Overall summary stats
    ax = axes[1, 1]
    ax.axis("off")
    predicted_pf = has_pf[has_pf["prefilter_confidence"] >= 0.6]
    mean_conf = predicted_pf["prefilter_confidence"].mean() * 100 if len(predicted_pf) > 0 else 0
    median_conf = predicted_pf["prefilter_confidence"].median() * 100 if len(predicted_pf) > 0 else 0
    mean_cos = predicted_pf["prefilter_cos_mean"].mean() if len(predicted_pf) > 0 else 0
    mean_nann = predicted_pf["prefilter_n_annotated_neighbours"].mean() if len(predicted_pf) > 0 else 0
    summary_lines = [
        ("Total reactions in database", f"{total_rxns:,}"),
        ("Previously annotated", f"{n_annotated:,} ({100 * n_annotated / total_rxns:.1f}%)"),
        ("Previously unannotated", f"{n_unannotated:,} ({100 * n_unannotated / total_rxns:.1f}%)"),
        ("", ""),
        ("With reaction center extracted", f"{n_with_center:,}"),
        ("Predicted (n>=3, agreement>=60%)", f"{n_predicted:,} ({100 * n_predicted / n_unannotated:.1f}% of unannotated)"),
        ("Very high agreement (>=80%)", f"{n_high_80:,} ({100 * n_high_80 / n_unannotated:.1f}% of unannotated)"),
        ("Highest agreement (>=90%)", f"{n_high_90:,} ({100 * n_high_90 / n_unannotated:.1f}% of unannotated)"),
        ("", ""),
        ("Mean % neighbor agreement", f"{mean_conf:.1f}%"),
        ("Median % neighbor agreement", f"{median_conf:.1f}%"),
        ("Mean cosine similarity", f"{mean_cos:.3f}"),
        ("Mean annotated neighbors", f"{mean_nann:.1f}"),
        ("", ""),
        ("Estimated accuracy (leave-one-out)", "90.2%"),
        ("Estimated F1 (leave-one-out)", "86.5%"),
        ("", ""),
        ("New annotation rate", f"{100 * n_predicted / total_rxns:.1f}% of all reactions gain EC"),
    ]
    y = 0.95
    for label, value in summary_lines:
        if label == "" and value == "":
            y -= 0.03
            continue
        ax.text(0.05, y, label, transform=ax.transAxes, fontsize=10, va="top", fontweight="normal", color="#555555")
        ax.text(0.95, y, value, transform=ax.transAxes, fontsize=10, va="top", ha="right", fontweight="bold", color="#222222")
        y -= 0.065
    ax.set_title("Summary Statistics", fontsize=12, fontweight="bold", pad=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved annotation summary to {output_path}")


def make_misannotation_correction_figure(df_mis, df_rxn, output_path):
    """4-panel figure summarizing misannotation correction results."""
    ec_lists = [parse_ec_numbers(v) for v in df_rxn["ec_numbers"].values]
    n_annotated = sum(1 for e in ec_lists if e)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Misannotation Detection and Correction (n={len(df_mis):,} candidates)", fontsize=15, fontweight="bold", y=0.98)

    actual_ec1, predicted_ec1, same_class = [], [], []
    for _, row in df_mis.iterrows():
        actual_ecs = parse_ec_numbers(row["actual_ec"])
        pred_ec = str(row["top_predicted_ec"])
        a1 = set(ec.split(".")[0] for ec in actual_ecs)
        p1 = pred_ec.split(".")[0]
        actual_ec1.append(list(a1)[0] if a1 else None)
        predicted_ec1.append(p1)
        same_class.append(p1 in a1)

    df_mis = df_mis.copy()
    df_mis["actual_ec1"] = actual_ec1
    df_mis["predicted_ec1"] = predicted_ec1
    df_mis["same_class"] = same_class

    # Panel 1: Misannotation type breakdown
    ax = axes[0, 0]
    _style_ax(ax)
    n_same = sum(same_class)
    n_diff = len(same_class) - n_same
    conf_tiers = {"50-60%": ((0.5, 0.6), []), "60-70%": ((0.6, 0.7), []), "70-80%": ((0.7, 0.8), []), "80-90%": ((0.8, 0.9), []), ">=90%": ((0.9, 1.01), [])}
    for _, row in df_mis.iterrows():
        c = row["confidence"]
        for label, ((lo, hi), lst) in conf_tiers.items():
            if lo <= c < hi:
                lst.append(row)
                break
    tier_labels = list(conf_tiers.keys())
    tier_same = [sum(1 for r in lst for _ in [1] if r["same_class"]) for _, (_, lst) in conf_tiers.items()]
    tier_diff = [sum(1 for r in lst for _ in [1] if not r["same_class"]) for _, (_, lst) in conf_tiers.items()]
    x = np.arange(len(tier_labels))
    w = 0.35
    ax.bar(x - w / 2, tier_same, w, label=f"Within-class error ({n_same})", color=COL_P, edgecolor="white", linewidth=0.5)
    ax.bar(x + w / 2, tier_diff, w, label=f"Cross-class error ({n_diff})", color="#d9534f", edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(tier_labels, fontsize=9)
    ax.set_xlabel("% Neighbor Agreement", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Misannotation Types by Confidence", fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=9, frameon=False)
    for i in range(len(tier_labels)):
        total_i = tier_same[i] + tier_diff[i]
        if total_i > 0:
            ax.text(x[i], max(tier_same[i], tier_diff[i]), f"n={total_i}", ha="center", va="bottom", fontsize=8, color="#333333")

    # Panel 2: EC class flow (actual -> predicted)
    ax = axes[0, 1]
    _style_ax(ax)
    transitions = Counter()
    for a, p in zip(actual_ec1, predicted_ec1):
        if a and p:
            transitions[(a, p)] += 1
    top_trans = transitions.most_common(15)
    trans_labels = [f"EC {a} -> EC {p}" for (a, p), _ in reversed(top_trans)]
    trans_counts = [c for _, c in reversed(top_trans)]
    trans_colors = [COL_G if a == p else "#d9534f" for (a, p), _ in reversed(top_trans)]
    y_pos = np.arange(len(trans_labels))
    ax.barh(y_pos, trans_counts, color=trans_colors, edgecolor="white", linewidth=0.5, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(trans_labels, fontsize=9)
    ax.set_xlabel("Count", fontsize=10)
    ax.set_title("Top EC Class Transitions\n(actual -> predicted)", fontsize=12, fontweight="bold", pad=10)
    for i, c in enumerate(trans_counts):
        ax.text(c, i, str(c), ha="left", va="center", fontsize=8, color="#333333")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    legend_elements = [mpatches.Patch(facecolor=COL_G, label="Within-class"), mpatches.Patch(facecolor="#d9534f", label="Cross-class")]
    ax.legend(handles=legend_elements, fontsize=9, frameon=False, loc="lower right")

    # Panel 3: Impact — misannotation rate by EC class
    ax = axes[1, 0]
    _style_ax(ax)
    ec1_annotated = Counter()
    for ecs in ec_lists:
        for ec in ecs:
            ec1_annotated[ec.split(".")[0]] += 1
    ec1_mis = Counter(df_mis["actual_ec1"].dropna())
    all_classes = sorted(set(list(ec1_annotated.keys()) + list(ec1_mis.keys())))
    mis_rates, mis_counts, ann_counts = [], [], []
    for c in all_classes:
        m = ec1_mis.get(c, 0)
        a = ec1_annotated.get(c, 0)
        mis_counts.append(m)
        ann_counts.append(a)
        mis_rates.append(100 * m / a if a > 0 else 0)
    x = np.arange(len(all_classes))
    ax.bar(x, mis_rates, color=COL_P, edgecolor="white", linewidth=0.5, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"EC {c}\n{EC1_NAMES.get(c, '')}" for c in all_classes], fontsize=8)
    ax.set_ylabel("Misannotation rate (%)", fontsize=10)
    ax.set_title("Estimated Misannotation Rate by EC Class", fontsize=12, fontweight="bold", pad=10)
    for i, (rate, m, a) in enumerate(zip(mis_rates, mis_counts, ann_counts)):
        ax.text(x[i], rate, f"{rate:.1f}%\n({m}/{a})", ha="center", va="bottom", fontsize=8, color="#333333")

    # Panel 4: Overall summary stats
    ax = axes[1, 1]
    ax.axis("off")
    n_same_total = sum(same_class)
    n_diff_total = len(same_class) - n_same_total
    mean_conf = df_mis["confidence"].mean() * 100
    median_conf = df_mis["confidence"].median() * 100
    overall_mis_rate = 100 * len(df_mis) / n_annotated
    summary_lines = [
        ("Total annotated reactions", f"{n_annotated:,}"),
        ("Misannotation candidates", f"{len(df_mis):,} ({overall_mis_rate:.1f}% of annotated)"),
        ("", ""),
        ("Within-class errors (same EC class)", f"{n_same_total:,} ({100 * n_same_total / len(df_mis):.1f}%)"),
        ("Cross-class errors (different EC class)", f"{n_diff_total:,} ({100 * n_diff_total / len(df_mis):.1f}%)"),
        ("", ""),
        ("Mean % neighbor agreement", f"{mean_conf:.1f}%"),
        ("Median % neighbor agreement", f"{median_conf:.1f}%"),
        ("", ""),
        ("Detection criteria:", ""),
        ("  Prefilter search", "reaction-center restricted"),
        ("  % neighbor agreement", ">=50%"),
        ("  Overlap with actual EC", "zero (F1=0)"),
        ("  Min annotated neighbors", ">=5"),
    ]
    y = 0.95
    for label, value in summary_lines:
        if label == "" and value == "":
            y -= 0.03
            continue
        ax.text(0.05, y, label, transform=ax.transAxes, fontsize=10, va="top", fontweight="normal", color="#555555")
        ax.text(0.95, y, value, transform=ax.transAxes, fontsize=10, va="top", ha="right", fontweight="bold", color="#222222")
        y -= 0.065
    ax.set_title("Summary Statistics", fontsize=12, fontweight="bold", pad=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved misannotation correction summary to {output_path}")


def main():
    print("Loading data ...")
    df_rxn = pd.read_csv(RXN_DATA_CSV)
    df_u = pd.read_csv(EC_PREDICTIONS_CSV)
    df_mis = pd.read_csv(MISANNOTATION_CSV)

    print(f"Total reactions: {len(df_rxn):,}")
    print(f"Unannotated predictions: {len(df_u):,}")
    print(f"Misannotation candidates: {len(df_mis):,}")

    print("\nGenerating annotation summary ...")
    make_annotation_figure(df_rxn, df_u, ANNOTATION_SUMMARY_PNG)

    print("Generating misannotation correction summary ...")
    make_misannotation_correction_figure(df_mis, df_rxn, MISANNOTATION_CORRECTION_PNG)

    print("\nDone.")


if __name__ == "__main__":
    main()
