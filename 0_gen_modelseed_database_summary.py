#!/usr/bin/env python
"""
Generate dataset summary figures and LaTeX table.

Outputs:
  - dataset_pie_charts.png (300 DPI)
  - dataset_summary_table.tex

Usage:
    conda activate rxnfp
    python 0_generate_dataset_figures.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from config import BASE_DIR

OUTPUT_PNG = os.path.join(BASE_DIR, "dataset_pie_charts.png")
OUTPUT_TEX = os.path.join(BASE_DIR, "dataset_summary_table.tex")


def generate_latex_table():
    """Generate LaTeX table for dataset summary statistics."""
    tex = r"""\begin{table}[h]
\centering
\caption{Summary of the ModelSEED reaction dataset.}
\label{tab:dataset_summary}
\begin{tabular}{lr}
\toprule
\textbf{Statistic} & \textbf{Count} \\
\midrule
Total reactions       & 36,646 \\
With EC number        & 16,591 \\
Without EC number     & 20,055 \\
\quad Single EC       & 15,170 \\
\quad Multiple ECs    & 1,421  \\
Unique EC numbers     & 6,803  \\
\bottomrule
\end{tabular}
\end{table}
"""
    with open(OUTPUT_TEX, "w") as f:
        f.write(tex)
    print(f"Saved LaTeX table to {OUTPUT_TEX}")


def generate_pie_charts():
    """Generate two pie charts: EC count distribution and EC main classes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("white")

    # ── Pie 1: EC count distribution ─────────────────────────────────────
    ec_dist = {
        1: 15170,
        2: 1177,
        3: 156,
        4: 43,
        5: 19,
        6: 5,
        7: 2,
        8: 5,
        9: 1,
        11: 1,
        13: 3,
        14: 4,
        15: 1,
        16: 2,
        81: 1,
        165: 1,
    }

    labels_1 = [
        "Single EC number",
        "Two EC numbers",
        "Three EC numbers",
        "More than four EC numbers",
    ]
    sizes_1 = [
        ec_dist[1],
        ec_dist[2],
        ec_dist[3],
        sum(v for k, v in ec_dist.items() if k >= 4),
    ]

    colors_1 = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]
    explode_1 = (0.03, 0.03, 0.03, 0.06)

    total_1 = sum(sizes_1)
    wedges1, _ = ax1.pie(
        sizes_1,
        labels=None,
        startangle=90,
        colors=colors_1,
        explode=explode_1,
        wedgeprops=dict(linewidth=1.5, edgecolor="white"),
    )

    legend_labels_1 = [
        f"{l}:  {s:,}  ({100 * s / total_1:.1f}%)" for l, s in zip(labels_1, sizes_1)
    ]
    ax1.legend(
        wedges1,
        legend_labels_1,
        loc="lower left",
        fontsize=10,
        frameon=True,
        fancybox=True,
        shadow=True,
    )

    ax1.set_title(
        "EC Count Distribution per Reaction\n(16,591 annotated reactions)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )

    # ── Pie 2: EC main classes ───────────────────────────────────────────
    ec_classes = {
        "1": 2150,
        "2": 1947,
        "3": 1303,
        "4": 753,
        "5": 332,
        "6": 237,
        "7": 81,
    }
    class_names = {
        "1": "Oxidoreductases",
        "2": "Transferases",
        "3": "Hydrolases",
        "4": "Lyases",
        "5": "Isomerases",
        "6": "Ligases",
        "7": "Translocases",
    }

    sizes_2 = list(ec_classes.values())
    colors_2 = [
        "#E24A33",
        "#348ABD",
        "#988ED5",
        "#FBC15E",
        "#8EBA42",
        "#FFB5B8",
        "#777777",
    ]
    explode_2 = (0.03,) * 7

    total_2 = sum(sizes_2)
    wedges2, _ = ax2.pie(
        sizes_2,
        labels=None,
        startangle=90,
        colors=colors_2,
        explode=explode_2,
        wedgeprops=dict(linewidth=1.5, edgecolor="white"),
    )

    legend_labels_2 = [
        f"EC {k} {class_names[k]}:  {v:,}  ({100 * v / total_2:.1f}%)"
        for k, v in ec_classes.items()
    ]
    ax2.legend(
        wedges2,
        legend_labels_2,
        loc="lower left",
        fontsize=10,
        frameon=True,
        fancybox=True,
        shadow=True,
    )

    ax2.set_title(
        "Distribution of Unique EC Numbers\nby Enzyme Main Class (6,803 total)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )

    plt.tight_layout(pad=2)
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved pie charts to {OUTPUT_PNG}")


if __name__ == "__main__":
    generate_latex_table()
    generate_pie_charts()
    print("Done.")
