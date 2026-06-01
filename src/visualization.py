#!/usr/bin/env python
"""
Visualize EC predictions for unannotated reactions and misannotation candidates.

Generates two figures:
  1. figures/unannotated_predictions.png
  2. figures/misannotation_analysis.png

Usage:
    conda activate rxnfp
    python -m src.visualization
"""

import os
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.config import (
    EC_PREDICTIONS_CSV,
    EVAL_RESULTS_CSV,
    MISANNOTATION_ANALYSIS_PNG,
    MISANNOTATION_CSV,
    UNANNOTATED_PREDICTIONS_PNG,
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


def _style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=9)


def _safe_ec1(ec_str):
    if not ec_str or ec_str == "None":
        return None
    return str(ec_str).split(".")[0]


def _safe_ec2(ec_str):
    if not ec_str or ec_str == "None":
        return None
    parts = str(ec_str).split(".")
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"
    return None


def make_unannotated_figure(df_u, output_path):
    """Generate 6-panel figure for unannotated reaction predictions."""
    has_pf = df_u[
        df_u["prefilter_predicted_ec"].notna()
        & (df_u["prefilter_predicted_ec"] != "None")
    ].copy()
    has_van = df_u[
        df_u["vanilla_predicted_ec"].notna() & (df_u["vanilla_predicted_ec"] != "None")
    ].copy()

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        "EC Predictions for Unannotated Reactions",
        fontsize=15, fontweight="bold", y=0.98,
    )

    # Panel 1: EC class distribution (prefilter)
    ax = axes[0, 0]
    _style_ax(ax)
    ec1_counts = Counter()
    for ec in has_pf["prefilter_predicted_ec"]:
        c = _safe_ec1(ec)
        if c:
            ec1_counts[c] += 1
    classes = sorted(ec1_counts.keys())
    counts = [ec1_counts[c] for c in classes]
    labels = [f"EC {c}\n{EC1_NAMES.get(c, '')}" for c in classes]
    colors = plt.cm.Set2(np.linspace(0, 0.9, len(classes)))
    bars = ax.bar(labels, counts, color=colors, edgecolor="white", linewidth=0.5)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20, f"{count:,}", ha="center", va="bottom", fontsize=8, color="#333333")
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Predicted EC Class Distribution\n(prefilter)", fontsize=12, fontweight="bold", pad=10)

    # Panel 2: Vanilla vs Prefilter agreement scatter
    ax = axes[0, 1]
    _style_ax(ax)
    both = df_u[
        df_u["vanilla_predicted_ec"].notna() & (df_u["vanilla_predicted_ec"] != "None")
        & df_u["prefilter_predicted_ec"].notna() & (df_u["prefilter_predicted_ec"] != "None")
    ].copy()
    agree = both["vanilla_predicted_ec"] == both["prefilter_predicted_ec"]
    ax.scatter(both.loc[~agree, "vanilla_confidence"] * 100, both.loc[~agree, "prefilter_confidence"] * 100, s=8, alpha=0.3, c="#999999", label=f"Disagree (n={int((~agree).sum()):,})", rasterized=True)
    ax.scatter(both.loc[agree, "vanilla_confidence"] * 100, both.loc[agree, "prefilter_confidence"] * 100, s=8, alpha=0.3, c=COL_P, label=f"Agree (n={int(agree.sum()):,})", rasterized=True)
    ax.plot([0, 100], [0, 100], "k--", alpha=0.3, linewidth=1)
    ax.set_xlabel("Vanilla % Neighbor Agreement", fontsize=10)
    ax.set_ylabel("Prefilter % Neighbor Agreement", fontsize=10)
    ax.set_title("Vanilla vs Prefilter Agreement", fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=9, frameon=False, loc="upper left", markerscale=2)
    ax.set_xlim(0, 102)
    ax.set_ylim(0, 102)

    # Panel 3: % Neighbor agreement by EC class (box plot)
    ax = axes[0, 2]
    _style_ax(ax)
    has_pf["ec1"] = has_pf["prefilter_predicted_ec"].apply(_safe_ec1)
    ec1_order = sorted(has_pf["ec1"].dropna().unique())
    box_data = [has_pf.loc[has_pf["ec1"] == c, "prefilter_confidence"].values * 100 for c in ec1_order]
    bp = ax.boxplot(box_data, labels=[f"EC {c}" for c in ec1_order], patch_artist=True, widths=0.6, medianprops=dict(color="black", linewidth=1.5), flierprops=dict(markersize=2, alpha=0.3))
    box_colors = plt.cm.Set2(np.linspace(0, 0.9, len(ec1_order)))
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("white")
    ax.set_ylabel("% Neighbor Agreement", fontsize=10)
    ax.set_title("% Neighbor Agreement by EC Class\n(prefilter)", fontsize=12, fontweight="bold", pad=10)

    # Panel 4: Number of annotated neighbours
    ax = axes[1, 0]
    _style_ax(ax)
    max_n = max(has_van["vanilla_n_annotated_neighbours"].max(), has_pf["prefilter_n_annotated_neighbours"].max())
    bins = np.arange(0, min(max_n + 2, 32), 1)
    ax.hist(has_van["vanilla_n_annotated_neighbours"], bins=bins, alpha=1.0, label="Without Prefilter", color=COL_V, edgecolor="white", linewidth=0.3)
    ax.hist(has_pf["prefilter_n_annotated_neighbours"], bins=bins, alpha=0.85, label="With Prefilter", color=COL_P, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Number of Annotated Neighbours (in top-30)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Annotated Neighbours per Query", fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=9, frameon=False)

    # Panel 5: Top 20 most predicted ECs (prefilter)
    ax = axes[1, 1]
    _style_ax(ax)
    ec_full_counts = Counter(has_pf["prefilter_predicted_ec"].values)
    top20 = ec_full_counts.most_common(20)
    top20_labels = [ec for ec, _ in reversed(top20)]
    top20_counts = [c for _, c in reversed(top20)]
    y_pos = np.arange(len(top20_labels))
    ax.barh(y_pos, top20_counts, color=COL_P, edgecolor="white", linewidth=0.5, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top20_labels, fontsize=8)
    ax.set_xlabel("Count", fontsize=10)
    ax.set_title("Top 20 Predicted ECs\n(prefilter)", fontsize=12, fontweight="bold", pad=10)
    for i, c in enumerate(top20_counts):
        ax.text(c + 5, i, str(c), ha="left", va="center", fontsize=8, color="#333333")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    # Panel 6: Cosine similarity distribution
    ax = axes[1, 2]
    _style_ax(ax)
    bins_cos = np.linspace(0, 1, 41)
    ax.hist(has_van["vanilla_cos_mean"], bins=bins_cos, alpha=1.0, label="Without Prefilter", color=COL_V, edgecolor="white", linewidth=0.3)
    ax.hist(has_pf["prefilter_cos_mean"], bins=bins_cos, alpha=0.85, label="With Prefilter", color=COL_P, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Mean Cosine Similarity (top-30 neighbours)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Cosine Similarity Distribution", fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=9, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved unannotated predictions figure to {output_path}")


def make_misannotation_figure(df_mis, df_eval, output_path):
    """Generate 6-panel figure for misannotation candidate analysis."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        f"Misannotation Candidate Analysis (n={len(df_mis):,})",
        fontsize=15, fontweight="bold", y=0.98,
    )

    actual_ec1_list = []
    predicted_ec1_list = []
    for _, row in df_mis.iterrows():
        actual_ecs = parse_ec_numbers(row["actual_ec"])
        a1 = _safe_ec1(actual_ecs[0]) if actual_ecs else None
        p1 = _safe_ec1(row["top_predicted_ec"])
        actual_ec1_list.append(a1)
        predicted_ec1_list.append(p1)
    df_mis = df_mis.copy()
    df_mis["actual_ec1"] = actual_ec1_list
    df_mis["predicted_ec1"] = predicted_ec1_list

    # Panel 1: Actual vs Predicted EC class heatmap
    ax = axes[0, 0]
    all_classes = sorted(set(c for c in actual_ec1_list + predicted_ec1_list if c))
    confusion = pd.DataFrame(0, index=all_classes, columns=all_classes)
    for a, p in zip(actual_ec1_list, predicted_ec1_list):
        if a and p:
            confusion.loc[a, p] += 1
    im = ax.imshow(confusion.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(all_classes)))
    ax.set_xticklabels([f"EC {c}" for c in all_classes], fontsize=9, rotation=45, ha="right")
    ax.set_yticks(range(len(all_classes)))
    ax.set_yticklabels([f"EC {c}" for c in all_classes], fontsize=9)
    ax.set_xlabel("Predicted EC Class", fontsize=10)
    ax.set_ylabel("Actual EC Class", fontsize=10)
    ax.set_title("Actual vs Predicted EC Class", fontsize=12, fontweight="bold", pad=10)
    for i in range(len(all_classes)):
        for j in range(len(all_classes)):
            val = confusion.values[i, j]
            if val > 0:
                text_color = "white" if val > confusion.values.max() * 0.6 else "black"
                ax.text(j, i, str(val), ha="center", va="center", fontsize=8, color=text_color)
    fig.colorbar(im, ax=ax, shrink=0.8)

    # Panel 2: % Neighbor agreement distribution
    ax = axes[0, 1]
    _style_ax(ax)
    bins = np.linspace(50, 100, 21)
    ax.hist(df_mis["confidence"] * 100, bins=bins, color=COL_P, edgecolor="white", linewidth=0.5, alpha=1.0)
    ax.set_xlabel("% Neighbor Agreement", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("% Neighbor Agreement Distribution\n(misannotation candidates)", fontsize=12, fontweight="bold", pad=10)
    median_conf = df_mis["confidence"].median() * 100
    ax.axvline(median_conf, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(median_conf + 0.5, ax.get_ylim()[1] * 0.9, f"median={median_conf:.1f}%", fontsize=9, va="top")

    # Panel 3: Top 15 reaction centers
    ax = axes[0, 2]
    _style_ax(ax)
    center_counts = Counter(df_mis["reaction_center"].values)
    top15 = center_counts.most_common(15)
    top15_labels = [c for c, _ in reversed(top15)]
    top15_counts = [n for _, n in reversed(top15)]
    y_pos = np.arange(len(top15_labels))
    ax.barh(y_pos, top15_counts, color=COL_V, edgecolor="white", linewidth=0.5, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top15_labels, fontsize=7)
    ax.set_xlabel("Count", fontsize=10)
    ax.set_title("Top 15 Reaction Centers\n(misannotation candidates)", fontsize=12, fontweight="bold", pad=10)
    for i, c in enumerate(top15_counts):
        ax.text(c + 0.5, i, str(c), ha="left", va="center", fontsize=8, color="#333333")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    # Panel 4: EC subclass breakdown for top 3 classes
    ax = axes[1, 0]
    _style_ax(ax)
    ec1_top3 = Counter(df_mis["actual_ec1"].dropna()).most_common(3)
    ec2_counts_by_class = {}
    for ec_class, _ in ec1_top3:
        subset = df_mis[df_mis["actual_ec1"] == ec_class]
        ec2_counter = Counter()
        for ec_str in subset["actual_ec"]:
            for ec in parse_ec_numbers(ec_str):
                ec2 = _safe_ec2(ec)
                if ec2:
                    ec2_counter[ec2] += 1
        ec2_counts_by_class[ec_class] = ec2_counter.most_common(5)

    all_subclasses, all_counts, all_colors = [], [], []
    class_colors = [COL_V, COL_P, "#55A868"]
    group_positions = []
    pos = 0
    for idx, (ec_class, _) in enumerate(ec1_top3):
        subs = ec2_counts_by_class[ec_class]
        for sub, cnt in subs:
            all_subclasses.append(sub)
            all_counts.append(cnt)
            all_colors.append(class_colors[idx])
            group_positions.append(pos)
            pos += 1
        pos += 0.5

    ax.barh(group_positions, all_counts, color=all_colors, edgecolor="white", linewidth=0.5, height=0.7)
    ax.set_yticks(group_positions)
    ax.set_yticklabels(all_subclasses, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Count", fontsize=10)
    ax.set_title("EC Subclass Breakdown\n(top 3 misannotated classes)", fontsize=12, fontweight="bold", pad=10)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=class_colors[i], label=f"EC {ec1_top3[i][0]} {EC1_NAMES.get(ec1_top3[i][0], '')}")
        for i in range(len(ec1_top3))
    ]
    ax.legend(handles=legend_elements, fontsize=8, frameon=False, loc="lower right")
    for i, c in enumerate(all_counts):
        ax.text(c + 0.5, group_positions[i], str(c), ha="left", va="center", fontsize=8, color="#333333")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    # Panel 5: Actual EC class vs Predicted EC class strip plot
    ax = axes[1, 1]
    _style_ax(ax)
    valid = df_mis.dropna(subset=["actual_ec1", "predicted_ec1"])
    pred_classes = sorted(valid["predicted_ec1"].unique())
    pred_color_map = {}
    cmap_colors = plt.cm.tab10(np.linspace(0, 1, max(len(pred_classes), 1)))
    for i, pc in enumerate(pred_classes):
        pred_color_map[pc] = cmap_colors[i]
    for pc in pred_classes:
        subset = valid[valid["predicted_ec1"] == pc]
        jitter = np.random.normal(0, 0.12, size=len(subset))
        x_vals = subset["actual_ec1"].astype(int).values + jitter
        ax.scatter(x_vals, subset["confidence"] * 100, s=12, alpha=0.5, color=pred_color_map[pc], label=f"Pred EC {pc}", rasterized=True)
    ax.set_xlabel("Actual EC Class", fontsize=10)
    ax.set_ylabel("% Neighbor Agreement", fontsize=10)
    ax.set_title("Actual EC vs Predicted EC\n(colored by prediction)", fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=7, frameon=False, loc="upper right", ncol=2, markerscale=2)

    # Panel 6: Confidence vs cosine similarity
    ax = axes[1, 2]
    _style_ax(ax)
    mis_ids = set(df_mis["query_id"])
    eval_mis = df_eval[df_eval["query_id"].isin(mis_ids)].copy()
    if len(eval_mis) > 0:
        ax.scatter(eval_mis["prefilter_cos_mean"], eval_mis["prefilter_majority_confidence"] * 100, s=12, alpha=0.4, color=COL_P, rasterized=True)
        ax.set_xlabel("Mean Cosine Similarity (prefilter)", fontsize=10)
        ax.set_ylabel("% Neighbor Agreement", fontsize=10)
        ax.set_title("Cosine Similarity vs % Neighbor Agreement\n(misannotation candidates)", fontsize=12, fontweight="bold", pad=10)
        corr = np.corrcoef(eval_mis["prefilter_cos_mean"], eval_mis["prefilter_majority_confidence"])[0, 1]
        ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes, fontsize=10, va="top", bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=1.0, edgecolor="#cccccc"))
    else:
        ax.text(0.5, 0.5, "No matching data", transform=ax.transAxes, ha="center", va="center", fontsize=14)
        ax.set_title("Cosine Similarity vs % Neighbor Agreement", fontsize=12, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved misannotation analysis figure to {output_path}")


def main():
    print("Loading data ...")
    df_u = pd.read_csv(EC_PREDICTIONS_CSV)
    df_mis = pd.read_csv(MISANNOTATION_CSV)
    df_eval = pd.read_csv(EVAL_RESULTS_CSV)

    print(f"Unannotated predictions: {len(df_u):,}")
    print(f"Misannotation candidates: {len(df_mis):,}")

    print("\nGenerating unannotated predictions figure ...")
    make_unannotated_figure(df_u, UNANNOTATED_PREDICTIONS_PNG)

    print("Generating misannotation analysis figure ...")
    make_misannotation_figure(df_mis, df_eval, MISANNOTATION_ANALYSIS_PNG)

    print("\nDone.")


if __name__ == "__main__":
    main()
