#!/usr/bin/env python
"""
Print comprehensive dataset statistics and generate summary figures.

Outputs:
  - Dataset statistics to stdout
  - figures/dataset_pie_charts.png
  - data/dataset_summary_table.tex

Usage:
    conda activate rxnfp
    python -m src.dataset_stats
"""

import ast
import csv
import os
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from src.config import (
    DATASET_PIE_PNG,
    DATASET_TABLE_TEX,
    MAPPED_RXNS_CSV,
    MAPPED_RXNS_WITH_RXN_CENTERS_CSV,
    RXN_DATA_CSV,
    RXN_FINGERPRINTS_NPY,
)
from src.utils import parse_ec_numbers


def print_stats():
    """Print comprehensive dataset statistics."""
    with open(RXN_DATA_CSV) as f:
        rxn_data = list(csv.DictReader(f))

    total = len(rxn_data)
    has_ec = sum(
        1 for r in rxn_data
        if r.get("ec_numbers", "").strip() and r.get("ec_numbers", "").strip() != "[]"
    )
    no_ec = total - has_ec

    single_ec, multi_ec = 0, 0
    ec_counts = Counter()
    all_ecs = set()
    for r in rxn_data:
        ecs = parse_ec_numbers(r.get("ec_numbers", ""))
        if ecs:
            if len(ecs) == 1:
                single_ec += 1
            else:
                multi_ec += 1
            ec_counts[len(ecs)] += 1
            all_ecs.update(ecs)

    ec1_counts = Counter()
    for ec in all_ecs:
        ec1_counts[ec.split(".")[0]] += 1

    print("=" * 60)
    print("ORIGINAL DATASET (rxn_data.csv)")
    print("=" * 60)
    print(f"  Total reactions:       {total}")
    print(f"  With EC number:        {has_ec}")
    print(f"  Without EC number:     {no_ec}")
    print(f"  Single EC:             {single_ec}")
    print(f"  Multiple ECs:          {multi_ec}")
    print(f"  Unique EC numbers:     {len(all_ecs)}")
    print(f"  EC count distribution: {dict(sorted(ec_counts.items()))}")
    print(f"  EC main classes:       {dict(sorted(ec1_counts.items()))}")
    print()

    # Mass balance issues
    hp_prod_only = hp_react_only = water_imbalanced = 0
    for r in rxn_data:
        smi = r.get("rxn_smiles", "")
        if ">>" not in smi:
            continue
        react, prod = smi.split(">>")
        rp, pp = react.split("."), prod.split(".")
        r_hp = sum(1 for x in rp if x.strip() == "[H+]")
        p_hp = sum(1 for x in pp if x.strip() == "[H+]")
        r_w = sum(1 for x in rp if x.strip() == "O")
        p_w = sum(1 for x in pp if x.strip() == "O")
        if p_hp > r_hp:
            hp_prod_only += 1
        if r_hp > p_hp:
            hp_react_only += 1
        if r_w != p_w:
            water_imbalanced += 1

    print("=" * 60)
    print("MASS BALANCE ISSUES")
    print("=" * 60)
    print(f"  [H+] only in products:  {hp_prod_only}")
    print(f"  [H+] only in reactants: {hp_react_only}")
    print(f"  Water imbalanced:       {water_imbalanced}")
    print()

    # Atom mapping
    with open(MAPPED_RXNS_CSV) as f:
        mapped = list(csv.DictReader(f))
    mapped_success = sum(1 for r in mapped if r.get("mapped_rxn", "").strip())
    mapped_fail = total - mapped_success

    print("=" * 60)
    print("ATOM MAPPING (mapped_rxns.csv)")
    print("=" * 60)
    print(f"  Successfully mapped:   {mapped_success}")
    print(f"  Failed to map:         {mapped_fail}")
    print()

    # Reaction centers
    with open(MAPPED_RXNS_WITH_RXN_CENTERS_CSV) as f:
        centers = list(csv.DictReader(f))
    has_center = sum(1 for r in centers if r.get("reaction_center", "[]") != "[]")
    no_center = total - has_center

    sizes, center_strs = [], []
    for r in centers:
        try:
            labels = ast.literal_eval(r.get("reaction_center", "[]"))
            if labels:
                sizes.append(len(labels))
                center_strs.append(r["reaction_center"])
        except (ValueError, SyntaxError):
            pass

    unique_centers = len(set(center_strs))
    center_freq = Counter(center_strs)
    singletons = sum(1 for n in center_freq.values() if n == 1)

    print("=" * 60)
    print("REACTION CENTERS")
    print("=" * 60)
    print(f"  With reaction center:  {has_center}")
    print(f"  Without reaction center: {no_center}")
    print(f"    - Failed mapping:      {mapped_fail}")
    print(f"    - Identity (R==P):     {no_center - mapped_fail}")
    print(f"  Center size: mean={np.mean(sizes):.1f}, median={np.median(sizes):.0f}, min={min(sizes)}, max={max(sizes)}")
    print(f"  Unique centers:        {unique_centers}")
    print(f"  Singleton centers:     {singletons}")
    print()

    # RXNFP fingerprints
    fps = np.load(RXN_FINGERPRINTS_NPY)
    print("=" * 60)
    print("RXNFP FINGERPRINTS")
    print("=" * 60)
    print(f"  Shape: {fps.shape} ({fps.shape[1]}-dimensional)")
    print(f"  Dtype: {fps.dtype}")
    print()

    # Evaluation subsets
    ec_by_id = {}
    for r in rxn_data:
        ecs = parse_ec_numbers(r.get("ec_numbers", ""))
        ec_by_id[r["id"]] = ecs

    annotated_with_center = unannotated_with_center = 0
    for r in centers:
        if r.get("reaction_center", "[]") == "[]":
            continue
        ecs = ec_by_id.get(r["id"], [])
        if ecs:
            annotated_with_center += 1
        else:
            unannotated_with_center += 1

    print("=" * 60)
    print("EVALUATION SUBSETS")
    print("=" * 60)
    print(f"  Annotated + center (leave-one-out): {annotated_with_center}")
    print(f"  Unannotated + center (prediction):  {unannotated_with_center}")
    print(f"  Annotated without center:           {has_ec - annotated_with_center}")
    print()


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
    with open(DATASET_TABLE_TEX, "w") as f:
        f.write(tex)
    print(f"Saved LaTeX table to {DATASET_TABLE_TEX}")


def generate_pie_charts():
    """Generate two pie charts: EC count distribution and EC main classes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("white")

    ec_dist = {1: 15170, 2: 1177, 3: 156, 4: 43, 5: 19, 6: 5, 7: 2, 8: 5, 9: 1, 11: 1, 13: 3, 14: 4, 15: 1, 16: 2, 81: 1, 165: 1}
    labels_1 = ["Single EC number", "Two EC numbers", "Three EC numbers", "More than four EC numbers"]
    sizes_1 = [ec_dist[1], ec_dist[2], ec_dist[3], sum(v for k, v in ec_dist.items() if k >= 4)]
    colors_1 = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]
    explode_1 = (0.03, 0.03, 0.03, 0.06)
    total_1 = sum(sizes_1)
    wedges1, _ = ax1.pie(sizes_1, labels=None, startangle=90, colors=colors_1, explode=explode_1, wedgeprops=dict(linewidth=1.5, edgecolor="white"))
    legend_labels_1 = [f"{l}:  {s:,}  ({100 * s / total_1:.1f}%)" for l, s in zip(labels_1, sizes_1)]
    ax1.legend(wedges1, legend_labels_1, loc="lower left", fontsize=10, frameon=True, fancybox=True, shadow=True)
    ax1.set_title("EC Count Distribution per Reaction\n(16,591 annotated reactions)", fontsize=13, fontweight="bold", pad=15)

    ec_classes = {"1": 2150, "2": 1947, "3": 1303, "4": 753, "5": 332, "6": 237, "7": 81}
    class_names = {"1": "Oxidoreductases", "2": "Transferases", "3": "Hydrolases", "4": "Lyases", "5": "Isomerases", "6": "Ligases", "7": "Translocases"}
    sizes_2 = list(ec_classes.values())
    colors_2 = ["#E24A33", "#348ABD", "#988ED5", "#FBC15E", "#8EBA42", "#FFB5B8", "#777777"]
    explode_2 = (0.03,) * 7
    total_2 = sum(sizes_2)
    wedges2, _ = ax2.pie(sizes_2, labels=None, startangle=90, colors=colors_2, explode=explode_2, wedgeprops=dict(linewidth=1.5, edgecolor="white"))
    legend_labels_2 = [f"EC {k} {class_names[k]}:  {v:,}  ({100 * v / total_2:.1f}%)" for k, v in ec_classes.items()]
    ax2.legend(wedges2, legend_labels_2, loc="lower left", fontsize=10, frameon=True, fancybox=True, shadow=True)
    ax2.set_title("Distribution of Unique EC Numbers\nby Enzyme Main Class (6,803 total)", fontsize=13, fontweight="bold", pad=15)

    plt.tight_layout(pad=2)
    plt.savefig(DATASET_PIE_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved pie charts to {DATASET_PIE_PNG}")


if __name__ == "__main__":
    print_stats()
    generate_latex_table()
    generate_pie_charts()
    print("Done.")
