#!/usr/bin/env python
"""
Batch evaluation comparing vanilla FAISS search vs.
reaction-center-prefiltered search.

Evaluations:
  1. Leave-one-out EC prediction on annotated reactions
  2. EC prediction coverage for unannotated reactions
  3. Misannotation detection for annotated reactions
  4. Retrieval quality (center purity, Tanimoto, cosine sim)

Parallelised across 70 CPU cores.

Usage:
    conda activate rxnfp
    python -m src.evaluation              # full evaluation + plot
    python -m src.evaluation --plot-only  # regenerate plot from saved CSVs
"""

import argparse
import ast
import csv
import math
import os
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool

import matplotlib

matplotlib.use("Agg")
import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from src.config import (
    EC_PREDICTIONS_CSV,
    EVAL_PLOT_PNG,
    EVAL_RESULTS_CSV,
    FAISS_INDEX_FILE,
    MAPPED_RXNS_WITH_RXN_CENTERS_CSV,
    MISANNOTATION_CSV,
    RXN_DATA_CSV,
    RXN_FINGERPRINTS_NPY,
)
from src.utils import l2_normalize_vectors, load_faiss_index, parse_ec_numbers

RDLogger.logger().setLevel(RDLogger.ERROR)

TOP_K = 30
MIN_ANNOTATED = 1
MIN_CONFIDENCE = 0.0
N_WORKERS = 70

# Module-level globals (set before fork, inherited by workers)
_IDS = None
_EC_LISTS = None
_SMILES = None
_CENTERS = None
_FPS_NORMED = None
_CENTER_TO_INDICES = None
_FAISS_INDEX = None


def _reactant_fps(rxn_smiles):
    """Return a list of Morgan fingerprints for the reactant molecules."""
    if not rxn_smiles or ">>" not in str(rxn_smiles):
        return []
    fps = []
    for smi in str(rxn_smiles).split(">>")[0].split("."):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
    return fps


def max_tanimoto(fps_a, fps_b):
    """Max pairwise Tanimoto between two lists of fingerprints."""
    if not fps_a or not fps_b:
        return 0.0
    return max(DataStructs.TanimotoSimilarity(fa, fb) for fa in fps_a for fb in fps_b)


def _top_k_search(query_idx, candidate_indices=None, exclude_self=True):
    """Return (indices, similarities) for top-k neighbours."""
    query_vec = _FPS_NORMED[query_idx].reshape(1, -1)

    if candidate_indices is None:
        k = TOP_K + 1 if exclude_self else TOP_K
        distances, indices = _FAISS_INDEX.search(query_vec, k)
        distances, indices = distances.flatten(), indices.flatten()
        if exclude_self:
            mask = indices != query_idx
            indices, distances = indices[mask][:TOP_K], distances[mask][:TOP_K]
        return indices, distances
    else:
        cand_arr = np.array(candidate_indices, dtype=np.int64)
        sims = (_FPS_NORMED[cand_arr] @ query_vec.T).flatten()
        if exclude_self:
            sims[cand_arr == query_idx] = -1.0
        top_local = np.argsort(-sims)[: min(TOP_K, len(sims))]
        return cand_arr[top_local], sims[top_local]


def _majority_vote_ec(indices, sims):
    """Return (predicted_ec, confidence, n_annotated) from majority vote."""
    ec_counter = Counter()
    ec_best_sim = {}
    n_annotated = 0
    for idx, sim in zip(indices, sims):
        ecs = _EC_LISTS[idx]
        if ecs:
            n_annotated += 1
            for ec in ecs:
                ec_counter[ec] += 1
                if sim > ec_best_sim.get(ec, -np.inf):
                    ec_best_sim[ec] = sim
    if n_annotated < MIN_ANNOTATED:
        return None, 0.0, n_annotated
    predicted_ec = max(ec_counter, key=lambda ec: (ec_counter[ec], ec_best_sim[ec]))
    confidence = ec_counter[predicted_ec] / n_annotated
    if confidence < MIN_CONFIDENCE:
        return None, confidence, n_annotated
    return predicted_ec, confidence, n_annotated


def _predict_ecs(indices, threshold=0.2):
    """Predict multiple ECs from neighbours using frequency threshold."""
    ec_counter = Counter()
    n_annotated = 0
    for idx in indices:
        ecs = _EC_LISTS[idx]
        if ecs:
            n_annotated += 1
            ec_counter.update(ecs)
    if n_annotated < MIN_ANNOTATED:
        return [], {}, n_annotated
    ec_confidences = {ec: count / n_annotated for ec, count in ec_counter.items()}
    predicted_ecs = sorted(
        (ec for ec, conf in ec_confidences.items() if conf >= threshold),
        key=lambda ec: -ec_confidences[ec],
    )
    return predicted_ecs, ec_confidences, n_annotated


def _has_prediction(val):
    """Return True if val is a non-null, non-None prediction value."""
    if val is None:
        return False
    try:
        if math.isnan(val):
            return False
    except TypeError:
        pass
    return str(val) not in ("None", "nan", "")


def eval_annotated(query_idx):
    """Leave-one-out evaluation for an annotated reaction."""
    query_id = _IDS[query_idx]
    query_ecs = _EC_LISTS[query_idx]
    query_center = _CENTERS[query_idx]

    if not query_ecs or query_center == "[]":
        return None

    query_fps = _reactant_fps(_SMILES[query_idx])
    query_ec_set = set(query_ecs)

    result = {
        "query_id": query_id,
        "query_ec": str(query_ecs),
        "query_center": query_center,
    }

    for method, cand_idx in [
        ("vanilla", None),
        ("prefilter", _CENTER_TO_INDICES.get(query_center)),
    ]:
        if method == "prefilter" and not cand_idx:
            for key, val in [
                ("top1_correct", False),
                ("majority_ec", None),
                ("majority_confidence", 0.0),
                ("n_annotated_neighbours", 0),
                ("majority_correct", False),
                ("predicted_ecs", "[]"),
                ("n_predicted_ecs", 0),
                ("precision", 0.0),
                ("recall", 0.0),
                ("f1", 0.0),
                ("topk_hit", False),
                ("center_purity", 0.0),
                ("mean_tanimoto", 0.0),
                ("cos_mean", 0.0),
            ]:
                result[f"{method}_{key}"] = val
            continue

        indices, sims = _top_k_search(query_idx, cand_idx, exclude_self=True)

        top1_ecs = set(_EC_LISTS[indices[0]]) if len(indices) > 0 else set()
        result[f"{method}_top1_correct"] = bool(query_ec_set & top1_ecs)

        pred_ec, confidence, n_ann = _majority_vote_ec(indices, sims)
        result[f"{method}_majority_ec"] = pred_ec
        result[f"{method}_majority_confidence"] = confidence
        result[f"{method}_n_annotated_neighbours"] = n_ann
        result[f"{method}_majority_correct"] = (
            pred_ec in query_ec_set if pred_ec else False
        )

        pred_ecs, _, _ = _predict_ecs(indices, threshold=0.2)
        pred_ec_set = set(pred_ecs)
        tp = len(query_ec_set & pred_ec_set)
        precision = tp / len(pred_ec_set) if pred_ec_set else 0.0
        recall = tp / len(query_ec_set) if query_ec_set else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        result[f"{method}_predicted_ecs"] = str(pred_ecs)
        result[f"{method}_n_predicted_ecs"] = len(pred_ecs)
        result[f"{method}_precision"] = precision
        result[f"{method}_recall"] = recall
        result[f"{method}_f1"] = f1

        result[f"{method}_topk_hit"] = any(
            query_ec_set & set(_EC_LISTS[i]) for i in indices
        )

        same_center = sum(1 for i in indices if _CENTERS[i] == query_center)
        result[f"{method}_center_purity"] = (
            same_center / len(indices) if len(indices) > 0 else 0.0
        )

        tanimotos = [
            max_tanimoto(query_fps, _reactant_fps(_SMILES[i])) for i in indices
        ]
        result[f"{method}_mean_tanimoto"] = (
            float(np.mean(tanimotos)) if tanimotos else 0.0
        )
        result[f"{method}_cos_mean"] = float(np.mean(sims)) if len(sims) > 0 else 0.0

    return result


def eval_unannotated(query_idx):
    """Predict EC for an unannotated reaction."""
    query_id = _IDS[query_idx]
    query_center = _CENTERS[query_idx]

    if query_center == "[]":
        return None

    result = {"query_id": query_id, "query_center": query_center}

    for method, cand_idx in [
        ("vanilla", None),
        ("prefilter", _CENTER_TO_INDICES.get(query_center)),
    ]:
        if method == "prefilter" and not cand_idx:
            result[f"{method}_predicted_ec"] = None
            result[f"{method}_predicted_ecs"] = "[]"
            result[f"{method}_confidence"] = 0.0
            result[f"{method}_n_annotated_neighbours"] = 0
            result[f"{method}_cos_mean"] = 0.0
            continue

        indices, sims = _top_k_search(query_idx, cand_idx, exclude_self=True)
        pred_ec, confidence, n_annotated = _majority_vote_ec(indices, sims)
        pred_ecs, _, _ = _predict_ecs(indices, threshold=0.2)

        result[f"{method}_predicted_ec"] = pred_ec
        result[f"{method}_predicted_ecs"] = str(pred_ecs)
        result[f"{method}_confidence"] = confidence
        result[f"{method}_n_annotated_neighbours"] = n_annotated
        result[f"{method}_cos_mean"] = float(np.mean(sims)) if len(sims) > 0 else 0.0

    return result


# -- Visualization -----------------------------------------------------------


def _style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=9)


def _grouped_bar(ax, labels, v_vals, p_vals, ylabel, title, ylim_top=1.12):
    """Draw a grouped bar chart with value labels."""
    COL_V, COL_P = "#4C72B0", "#DD8452"
    w, x = 0.32, np.arange(len(labels))
    ax.bar(
        x - w / 2,
        v_vals,
        w,
        label="Without Prefilter",
        color=COL_V,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.bar(
        x + w / 2,
        p_vals,
        w,
        label="With Prefilter",
        color=COL_P,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_ylim(0, ylim_top)
    ax.legend(fontsize=9, frameon=False)
    _style_ax(ax)
    for i in range(len(labels)):
        ax.text(
            x[i] - w / 2,
            v_vals[i] + 0.003,
            f"{v_vals[i]:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#333333",
        )
        ax.text(
            x[i] + w / 2,
            p_vals[i] + 0.003,
            f"{p_vals[i]:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#333333",
        )


def make_plots(annotated_results, unannotated_results, misannotations, output_path):
    """Generate 6-panel comparison figure."""
    COL_V, COL_P = "#4C72B0", "#DD8452"

    fig, axes = plt.subplots(2, 3, figsize=(18, 13))
    fig.suptitle(
        "RXNFP Similarity Search: Without vs With Reaction-Center Prefilter",
        fontsize=15,
        fontweight="bold",
        y=1.0,
    )

    # Panel 1: Single-EC Prediction Accuracy
    metrics_1 = ["top1_correct", "majority_correct", "topk_hit"]
    labels_1 = ["Top-1\nAccuracy", "Majority-Vote\nAccuracy", "Top-k\nHit Rate"]
    v1 = [np.mean([r[f"vanilla_{m}"] for r in annotated_results]) for m in metrics_1]
    p1 = [np.mean([r[f"prefilter_{m}"] for r in annotated_results]) for m in metrics_1]
    _grouped_bar(
        axes[0, 0],
        labels_1,
        v1,
        p1,
        ylabel="Accuracy",
        title="Single-EC Prediction\n(leave-one-out)",
    )

    # Panel 2: Multi-EC Precision / Recall / F1
    metrics_2 = ["precision", "recall", "f1"]
    labels_2 = ["Precision", "Recall", "F1 Score"]
    v2 = [np.mean([r[f"vanilla_{m}"] for r in annotated_results]) for m in metrics_2]
    p2 = [np.mean([r[f"prefilter_{m}"] for r in annotated_results]) for m in metrics_2]
    _grouped_bar(
        axes[0, 1],
        labels_2,
        v2,
        p2,
        ylabel="Score",
        title="Multi-EC Prediction\n(leave-one-out, threshold=0.2)",
    )

    # Panel 3: Retrieval Quality (dual y-axis)
    ax3 = axes[0, 2]
    _style_ax(ax3)
    v_tan = np.mean([r["vanilla_mean_tanimoto"] for r in annotated_results])
    p_tan = np.mean([r["prefilter_mean_tanimoto"] for r in annotated_results])
    v_con = np.mean([r["vanilla_majority_confidence"] for r in annotated_results]) * 100
    p_con = (
        np.mean([r["prefilter_majority_confidence"] for r in annotated_results]) * 100
    )
    w3, x3 = 0.32, np.arange(2)
    ax3.bar(
        x3[0] - w3 / 2, v_tan, w3, color=COL_V, edgecolor="white", linewidth=0.5,
        label="Without Prefilter",
    )
    ax3.bar(
        x3[0] + w3 / 2, p_tan, w3, color=COL_P, edgecolor="white", linewidth=0.5,
        label="With Prefilter",
    )
    ax3.set_ylabel("Substrate Tanimoto", fontsize=10)
    ax3.set_ylim(0, 1.2)
    ax3.text(x3[0] - w3 / 2, v_tan + 0.005, f"{v_tan:.3f}", ha="center", va="bottom", fontsize=8, color="#333333")
    ax3.text(x3[0] + w3 / 2, p_tan + 0.005, f"{p_tan:.3f}", ha="center", va="bottom", fontsize=8, color="#333333")
    ax3b = ax3.twinx()
    ax3b.bar(x3[1] - w3 / 2, v_con, w3, color=COL_V, edgecolor="white", linewidth=0.5)
    ax3b.bar(x3[1] + w3 / 2, p_con, w3, color=COL_P, edgecolor="white", linewidth=0.5)
    ax3b.set_ylabel("% Neighbor Agreement", fontsize=10)
    ax3b.set_ylim(0, 120)
    ax3b.spines["top"].set_visible(False)
    ax3b.text(x3[1] - w3 / 2, v_con + 0.3, f"{v_con:.1f}%", ha="center", va="bottom", fontsize=8, color="#333333")
    ax3b.text(x3[1] + w3 / 2, p_con + 0.3, f"{p_con:.1f}%", ha="center", va="bottom", fontsize=8, color="#333333")
    ax3.set_xticks(x3)
    ax3.set_xticklabels(["Substrate\nTanimoto", "% Neighbor\nAgreement"], fontsize=9)
    ax3.set_title("Retrieval Quality", fontsize=12, fontweight="bold", pad=10)
    ax3.legend(fontsize=9, frameon=False)

    # Panel 4: Coverage & High-Confidence (unannotated, dual y-axis)
    ax = axes[1, 0]
    _style_ax(ax)
    total_u = len(unannotated_results)
    v_cov = sum(1 for r in unannotated_results if _has_prediction(r["vanilla_predicted_ec"]))
    p_cov = sum(1 for r in unannotated_results if _has_prediction(r["prefilter_predicted_ec"]))
    v_conf_list = [r["vanilla_confidence"] for r in unannotated_results if _has_prediction(r["vanilla_predicted_ec"])]
    p_conf_list = [r["prefilter_confidence"] for r in unannotated_results if _has_prediction(r["prefilter_predicted_ec"])]
    v_high = sum(1 for c in v_conf_list if c >= 0.5)
    p_high = sum(1 for c in p_conf_list if c >= 0.5)
    v_cov_pct, p_cov_pct = 100 * v_cov / total_u, 100 * p_cov / total_u
    w, x = 0.32, np.arange(2)
    ax.bar(x[0] - w / 2, v_cov_pct, w, label="Without Prefilter", color=COL_V, edgecolor="white", linewidth=0.5)
    ax.bar(x[0] + w / 2, p_cov_pct, w, label="With Prefilter", color=COL_P, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Coverage (%)", fontsize=10, color=COL_V)
    ax.tick_params(axis="y", labelcolor=COL_V)
    ax.set_ylim(0, 130)
    ax.text(x[0] - w / 2, v_cov_pct + 0.4, f"{v_cov_pct:.1f}%", ha="center", va="bottom", fontsize=8, color="#333333")
    ax.text(x[0] + w / 2, p_cov_pct + 0.4, f"{p_cov_pct:.1f}%", ha="center", va="bottom", fontsize=8, color="#333333")
    ax2 = ax.twinx()
    ax2.bar(x[1] - w / 2, v_high, w, color=COL_V, edgecolor="white", linewidth=0.5)
    ax2.bar(x[1] + w / 2, p_high, w, color=COL_P, edgecolor="white", linewidth=0.5)
    ax2.set_ylabel("High-Agreement Count", fontsize=10, color=COL_P)
    ax2.tick_params(axis="y", labelcolor=COL_P)
    ax2.spines["top"].set_visible(False)
    ax2.set_ylim(0, max(v_high, p_high) * 1.45)
    ax2.text(x[1] - w / 2, v_high + max(v_high, p_high) * 0.01, f"{v_high:,}", ha="center", va="bottom", fontsize=8, color="#333333")
    ax2.text(x[1] + w / 2, p_high + max(v_high, p_high) * 0.01, f"{p_high:,}", ha="center", va="bottom", fontsize=8, color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels(["Coverage\n(%)", "High-Agreement\nPredictions (>=50%)"], fontsize=9)
    ax.set_title(f"Unannotated Reactions (n={total_u:,})", fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=9, frameon=False, loc="upper left")

    # Panel 5: % Neighbor Agreement Distribution (unannotated)
    ax = axes[1, 1]
    _style_ax(ax)
    bins = np.linspace(0, 100, 21)
    ax.hist([c * 100 for c in v_conf_list], bins=bins, alpha=1.0, label=f"Without Prefilter (n={len(v_conf_list):,})", color=COL_V, edgecolor="white", linewidth=0.3)
    ax.hist([c * 100 for c in p_conf_list], bins=bins, alpha=0.85, label=f"With Prefilter (n={len(p_conf_list):,})", color=COL_P, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("% Neighbor Agreement", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("% Neighbor Agreement Distribution\n(unannotated reactions)", fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=9, frameon=False, loc="upper right")

    # Panel 6: Misannotation Candidates by EC Class
    ax = axes[1, 2]
    _style_ax(ax)
    ec1_names = {
        "1": "Oxidoreductases", "2": "Transferases", "3": "Hydrolases",
        "4": "Lyases", "5": "Isomerases", "6": "Ligases", "7": "Translocases",
    }
    if misannotations:
        ec1_counts = Counter()
        for m in misannotations:
            for ec in parse_ec_numbers(m["actual_ec"]):
                ec1_counts[ec.split(".")[0]] += 1
        sorted_classes = sorted(ec1_counts, key=lambda c: -ec1_counts[c])
        counts = [ec1_counts[c] for c in sorted_classes]
        y_pos = np.arange(len(sorted_classes))
        ax.barh(y_pos, counts, color=plt.cm.Set2(np.linspace(0, 0.9, len(sorted_classes))), edgecolor="white", linewidth=0.5, height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"EC {c} {ec1_names.get(c, '')}" for c in sorted_classes], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Count", fontsize=10)
        ax.set_title(f"Misannotation Candidates by EC Class\n(n={len(misannotations):,}, zero overlap with predicted ECs)", fontsize=12, fontweight="bold", pad=10)
        for i, c in enumerate(counts):
            ax.text(c + max(counts) * 0.005, i, str(c), ha="left", va="center", fontsize=9, color="#333333")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    else:
        ax.text(0.5, 0.5, "No misannotation\ncandidates found", transform=ax.transAxes, ha="center", va="center", fontsize=14)
        ax.set_title("Misannotation Candidates", fontsize=12, fontweight="bold")

    fig.subplots_adjust(top=0.93, bottom=0.08, left=0.07, right=0.97, hspace=0.45, wspace=0.35)
    fig.savefig(output_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved comparison figure to {output_path}")


# -- Main --------------------------------------------------------------------


def _run_evaluation():
    global _IDS, _EC_LISTS, _SMILES, _CENTERS, _FPS_NORMED, _CENTER_TO_INDICES, _FAISS_INDEX

    t0 = time.time()

    print("Loading data ...")
    df = pd.read_csv(RXN_DATA_CSV)
    df_centers = pd.read_csv(MAPPED_RXNS_WITH_RXN_CENTERS_CSV)
    fps = np.load(RXN_FINGERPRINTS_NPY).astype(np.float32)
    fps_normed = fps.copy()
    l2_normalize_vectors(fps_normed)

    ids = df["id"].values
    ec_lists = [parse_ec_numbers(v) for v in df["ec_numbers"].values]
    smiles = df["rxn_smiles"].values
    centers = df_centers["reaction_center"].values

    center_to_indices = defaultdict(list)
    for i, c in enumerate(centers):
        if c != "[]":
            center_to_indices[c].append(i)

    print("Loading FAISS index ...")
    faiss_index = load_faiss_index(FAISS_INDEX_FILE)
    if faiss_index is None:
        print("ERROR: Could not load FAISS index. Exiting.")
        sys.exit(1)
    faiss.omp_set_num_threads(1)

    _IDS, _EC_LISTS, _SMILES, _CENTERS = ids, ec_lists, smiles, centers
    _FPS_NORMED, _CENTER_TO_INDICES, _FAISS_INDEX = (
        fps_normed, dict(center_to_indices), faiss_index,
    )

    annotated_idx = [i for i in range(len(ids)) if ec_lists[i] and centers[i] != "[]"]
    unannotated_idx = [i for i in range(len(ids)) if not ec_lists[i] and centers[i] != "[]"]
    print(f"Annotated reactions (with EC + center):    {len(annotated_idx):,}")
    print(f"Unannotated reactions (no EC, with center): {len(unannotated_idx):,}")
    print(f"Using {N_WORKERS} workers\n")

    def _run_pool(worker, indices, label):
        results, t = [], time.time()
        with Pool(N_WORKERS) as pool:
            for count, r in enumerate(pool.imap_unordered(worker, indices, chunksize=8), start=1):
                if r is not None:
                    results.append(r)
                if count % 2000 == 0:
                    elapsed = time.time() - t
                    rate = count / elapsed
                    eta = (len(indices) - count) / rate if rate > 0 else 0
                    print(f"  {label}: {count:>6d} / {len(indices)} ({count / len(indices) * 100:.1f}%)  [{rate:.0f} rxn/s, ETA {eta:.0f}s]")
                    sys.stdout.flush()
        print(f"  {label}: {len(indices)} / {len(indices)} done ({time.time() - t:.1f}s)\n")
        return results

    print("=== Evaluation 1: Leave-One-Out EC Prediction ===")
    annotated_results = _run_pool(eval_annotated, annotated_idx, "Annotated")

    print("=== Evaluation 2: EC Prediction for Unannotated Reactions ===")
    unannotated_results = _run_pool(eval_unannotated, unannotated_idx, "Unannotated")

    # Print summaries
    print("=" * 72)
    print("LEAVE-ONE-OUT EC PREDICTION (annotated reactions)")
    print("=" * 72)
    for metric, label in [
        ("top1_correct", "Top-1 Accuracy"),
        ("majority_correct", "Majority-Vote Accuracy"),
        ("topk_hit", "Top-k Hit Rate"),
        ("precision", "Multi-EC Precision"),
        ("recall", "Multi-EC Recall"),
        ("f1", "Multi-EC F1"),
        ("majority_confidence", "Mean % Neighbor Agreement"),
        ("center_purity", "Center Purity"),
        ("mean_tanimoto", "Mean Substrate Tanimoto"),
        ("cos_mean", "Mean Cosine Similarity"),
    ]:
        v = np.mean([r[f"vanilla_{metric}"] for r in annotated_results])
        p = np.mean([r[f"prefilter_{metric}"] for r in annotated_results])
        delta = p - v
        print(f"  {label:<30s}  Vanilla: {v:.4f}  Prefilter: {p:.4f}  ({'+' if delta >= 0 else ''}{delta:.4f})")
    print()

    print("=" * 72)
    print("EC PREDICTION COVERAGE (unannotated reactions)")
    print("=" * 72)
    v_covered = sum(1 for r in unannotated_results if _has_prediction(r["vanilla_predicted_ec"]))
    p_covered = sum(1 for r in unannotated_results if _has_prediction(r["prefilter_predicted_ec"]))
    total_u = len(unannotated_results)
    v_conf = [r["vanilla_confidence"] for r in unannotated_results if _has_prediction(r["vanilla_predicted_ec"])]
    p_conf = [r["prefilter_confidence"] for r in unannotated_results if _has_prediction(r["prefilter_predicted_ec"])]
    print(f"  Reactions with prediction:  Vanilla: {v_covered}/{total_u} ({v_covered / total_u * 100:.1f}%)  Prefilter: {p_covered}/{total_u} ({p_covered / total_u * 100:.1f}%)")
    print(f"  Mean % neighbor agreement:  Vanilla: {np.mean(v_conf):.4f}  Prefilter: {np.mean(p_conf) if p_conf else 0:.4f}")
    print(f"  High-agreement (>=0.5):     Vanilla: {sum(1 for c in v_conf if c >= 0.5)}  Prefilter: {sum(1 for c in p_conf if c >= 0.5)}")
    print()

    # Misannotation detection
    print("=" * 72)
    print("MISANNOTATION CANDIDATES")
    print("=" * 72)
    id_to_idx = {_IDS[i]: i for i in range(len(_IDS))}
    misannotations = []
    for r in annotated_results:
        conf = r["prefilter_majority_confidence"]
        if r["prefilter_f1"] == 0.0 and r["prefilter_n_predicted_ecs"] > 0 and conf >= 0.5:
            query_idx = id_to_idx[r["query_id"]]
            query_ec_set = set(ast.literal_eval(r["query_ec"]))
            pred_ecs = ast.literal_eval(r["prefilter_predicted_ecs"])
            cand_indices = _CENTER_TO_INDICES.get(r["query_center"], [])
            top_pred_ec = r["prefilter_majority_ec"]

            if ".-" in str(top_pred_ec):
                continue

            actual_ec2s = set(".".join(ec.split(".")[:2]) for ec in query_ec_set)
            pred_ec2s = set(".".join(ec.split(".")[:2]) for ec in pred_ecs)
            if actual_ec2s & pred_ec2s:
                continue

            if len(cand_indices) > 200:
                continue

            actual_ec1s = set(ec.split(".")[0] for ec in query_ec_set if "-" not in ec)
            pred_ec1s = set(ec.split(".")[0] for ec in pred_ecs)
            same_ec1_neighbors = [
                idx for idx in cand_indices
                if idx != query_idx and any(ec.split(".")[0] in actual_ec1s for ec in _EC_LISTS[idx])
            ]
            if same_ec1_neighbors and not any(
                ec.split(".")[0] in pred_ec1s for idx in same_ec1_neighbors for ec in _EC_LISTS[idx]
            ):
                continue

            actual_ec_support = sum(
                1 for idx in cand_indices
                if idx != query_idx and any(ec in _EC_LISTS[idx] for ec in query_ec_set)
            )
            if actual_ec_support == 0:
                continue

            support_by_ec = {ec: [] for ec in pred_ecs}
            for idx in cand_indices:
                if idx != query_idx and _EC_LISTS[idx]:
                    for ec in pred_ecs:
                        if ec in _EC_LISTS[idx]:
                            support_by_ec[ec].append(str(_IDS[idx]))
            misannotations.append({
                "query_id": r["query_id"],
                "actual_ec": r["query_ec"],
                "predicted_ecs": r["prefilter_predicted_ecs"],
                "top_predicted_ec": r["prefilter_majority_ec"],
                "confidence": conf,
                "n_supporting": str([len(support_by_ec[ec]) for ec in pred_ecs]),
                "supporting_reactions": str([support_by_ec[ec] for ec in pred_ecs]),
                "reaction_center": r["query_center"],
            })
    print(f"  Misannotation candidates (prefilter, confidence >= 0.5): {len(misannotations)}")
    for m in misannotations[:5]:
        print(f"    {m['query_id']}: actual={m['actual_ec']}  predicted={m['predicted_ecs']}  conf={m['confidence']:.3f}")
    print()

    # Save CSVs
    for rows, path in [
        (annotated_results, EVAL_RESULTS_CSV),
        (unannotated_results, EC_PREDICTIONS_CSV),
        (misannotations, MISANNOTATION_CSV),
    ]:
        if rows:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(f"Saved {path}")

    print(f"\nTotal time: {time.time() - t0:.1f}s")
    return annotated_results, unannotated_results, misannotations


def _plot_from_csvs():
    print("Loading CSVs ...")
    annotated_results = pd.read_csv(EVAL_RESULTS_CSV).to_dict(orient="records")
    unannotated_results = pd.read_csv(EC_PREDICTIONS_CSV).to_dict(orient="records")
    misannotations = pd.read_csv(MISANNOTATION_CSV).to_dict(orient="records")
    print(f"  Annotated:    {len(annotated_results):,}")
    print(f"  Unannotated:  {len(unannotated_results):,}")
    print(f"  Misannotated: {len(misannotations):,}")
    return annotated_results, unannotated_results, misannotations


def main():
    parser = argparse.ArgumentParser(description="Batch evaluation and/or plotting.")
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Regenerate plot from saved CSVs without re-running evaluation.",
    )
    args = parser.parse_args()

    if args.plot_only:
        annotated_results, unannotated_results, misannotations = _plot_from_csvs()
    else:
        annotated_results, unannotated_results, misannotations = _run_evaluation()

    print("\nGenerating comparison plots ...")
    make_plots(annotated_results, unannotated_results, misannotations, EVAL_PLOT_PNG)


if __name__ == "__main__":
    main()
