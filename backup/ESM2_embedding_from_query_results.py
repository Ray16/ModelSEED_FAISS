#!/usr/bin/env python3
"""
Unified Pipeline: Reaction Similarity → ESM-2 Embedding Validation
====================================================================
1. Query FAISS index for top-K similar reactions
2. Sample negative-control reactions from 30 diverse EC classes
3. Fetch Swiss-Prot sequences from UniProt for all ECs
4. Compute ESM-2 (esm2_t48_15B_UR50D) mean-pool embeddings
5. Visualize: PCA, t-SNE, UMAP scatter + hierarchical clustering dendrogram
   Similar reactions in dark orange, controls in gray

Usage:
    python similarity_esm_pipeline.py --rxn_name rxn00646 --top_k 10
    python similarity_esm_pipeline.py --rxn_name rxn00646 --top_k 10 --max_seqs 3
    python similarity_esm_pipeline.py --rxn_name rxn00646 --top_k 10 --esm_model esm2_t33_650M_UR50D
"""

import argparse
import os
import sys
import time
import subprocess
from io import StringIO
from collections import defaultdict

import numpy as np
import pandas as pd

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform

import torch

# ── Import your existing project modules ────────────────────────────────────
from config import (
    FAISS_INDEX_FILE,
    QUERY_RESULT_CSV,
    RXN_DATA_CSV,
    RXN_FINGERPRINTS_NPY,
)
from utils import l2_normalize_vectors, load_faiss_index, parse_ec_numbers

# ── Config ──────────────────────────────────────────────────────────────────
OUT_DIR         = os.path.dirname(QUERY_RESULT_CSV)
SEQS_FILE       = os.path.join(OUT_DIR, "sequences.fasta")
EMBEDDINGS_NPY  = os.path.join(OUT_DIR, "esm_embeddings.npy")
METADATA_CSV    = os.path.join(OUT_DIR, "esm_metadata.csv")
SCATTER_PNG     = os.path.join(OUT_DIR, "esm_scatter.png")
SCATTER_PDF     = os.path.join(OUT_DIR, "esm_scatter.pdf")
DENDRO_PNG      = os.path.join(OUT_DIR, "esm_dendrogram.png")
DENDRO_PDF      = os.path.join(OUT_DIR, "esm_dendrogram.pdf")
STATS_TXT       = os.path.join(OUT_DIR, "esm_cluster_stats.txt")

MAX_SEQS_PER_EC = 3       # keep manageable for ESM inference
MAX_SEQ_LEN     = 1022    # ESM-2 max input length
REQUEST_TIMEOUT = 60
ESM_BATCH_SIZE  = 4       # batch size for ESM inference (adjust for GPU memory)

# Default ESM model — use 3B
DEFAULT_ESM_MODEL = "esm2_t48_3B_UR50D"

# ── 30 diverse negative-control ECs across all 7 major EC classes ───────────
DEFAULT_CONTROL_ECS = [
    # EC 1 — Oxidoreductases
    "1.1.1.1",     # alcohol dehydrogenase
    "1.2.1.12",    # glyceraldehyde-3-phosphate dehydrogenase
    "1.4.3.4",     # monoamine oxidase
    "1.11.1.6",    # catalase
    "1.14.13.39",  # nitric-oxide synthase
    "1.15.1.1",    # superoxide dismutase
    # EC 2 — Transferases
    "2.1.1.37",    # DNA (cytosine-5-)-methyltransferase
    "2.3.1.9",     # acetyl-CoA C-acetyltransferase
    "2.4.2.1",     # purine-nucleoside phosphorylase
    "2.6.1.1",     # aspartate transaminase
    "2.7.1.1",     # hexokinase
    "2.7.7.6",     # DNA-directed RNA polymerase
    # EC 3 — Hydrolases
    "3.1.1.3",     # triacylglycerol lipase
    "3.1.3.1",     # alkaline phosphatase
    "3.2.1.1",     # alpha-amylase
    "3.4.21.4",    # trypsin
    "3.5.1.5",     # urease
    "3.5.4.4",     # adenosine deaminase
    # EC 4 — Lyases
    "4.1.1.1",     # pyruvate decarboxylase
    "4.2.1.1",     # carbonic anhydrase
    "4.2.1.11",    # phosphopyruvate hydratase (enolase)
    "4.6.1.1",     # adenylate cyclase
    # EC 5 — Isomerases
    "5.1.3.1",     # ribulose-phosphate 3-epimerase
    "5.2.1.1",     # maleate isomerase
    "5.3.1.1",     # triose-phosphate isomerase
    "5.4.2.2",     # phosphoglucomutase
    # EC 6 — Ligases (non-overlapping with query)
    "6.1.1.1",     # tyrosine--tRNA ligase
    "6.4.1.2",     # acetyl-CoA carboxylase
    # EC 7 — Translocases
    "7.1.1.1",     # NADH:ubiquinone reductase
    "7.2.2.3",     # H+/K+-exchanging ATPase
]

# ── Colors ──────────────────────────────────────────────────────────────────
SIMILAR_COLOR = "#D2691E"
CONTROL_COLOR = "#999999"
# ────────────────────────────────────────────────────────────────────────────

# ── HTTP session ────────────────────────────────────────────────────────────
def get_session():
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=2,
                  status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))
    return session

SESSION = get_session()


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 1: FAISS Similarity Search
# ═══════════════════════════════════════════════════════════════════════════

def run_similarity_search(rxn_name: str, top_k: int = 10) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print(f"FAISS Similarity Search: {rxn_name} (top {top_k})")
    print(f"{'='*60}")

    index = load_faiss_index(FAISS_INDEX_FILE)
    df = pd.read_csv(RXN_DATA_CSV)

    matches = df[df.id == rxn_name].index.to_numpy()
    if len(matches) == 0:
        print(f"ERROR: Reaction '{rxn_name}' not found in {RXN_DATA_CSV}")
        sys.exit(1)
    query_idx = matches[0]
    query_ec = parse_ec_numbers(df.iloc[query_idx, 3])

    print(f"  Query reaction: {df.iloc[query_idx, 0]}")
    print(f"  Query EC:       {query_ec}")

    query_vector = (
        np.load(RXN_FINGERPRINTS_NPY)[query_idx]
        .astype(np.float32).reshape(1, -1)
    )
    query_vector = l2_normalize_vectors(query_vector)
    D, I = index.search(query_vector, top_k)

    result_df = pd.DataFrame({
        "similarity_ranking": list(range(1, len(D[0]) + 1)),
        "rxn_name": [df.iloc[idx, 0] for idx in I[0]],
        "ec_number": [parse_ec_numbers(df.iloc[idx, 3]) for idx in I[0]],
        "distance": D[0],
        "rxn_smiles": [df.iloc[idx]["rxn_smiles"] for idx in I[0]],
    })
    result_df.to_csv(QUERY_RESULT_CSV, index=False)

    print(f"\n  Top {len(result_df)} similar reactions:")
    for _, row in result_df.iterrows():
        ec_display = str(row['ec_number']).strip("[]'\" ")
        print(f"    #{int(row['similarity_ranking']):2d}  {str(row['rxn_name']):12s}  "
              f"EC {ec_display}  dist={float(row['distance']):.4f}")
    print(f"  Saved to: {QUERY_RESULT_CSV}")
    return result_df


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 2: Collect EC groups
# ═══════════════════════════════════════════════════════════════════════════

def collect_ec_groups(result_df: pd.DataFrame, control_ecs: list):
    ec_info = {}
    for _, row in result_df.iterrows():
        ec_raw = str(row["ec_number"]).strip("[]'\" ").split("'")[0].strip()
        if not ec_raw or ec_raw == "nan":
            continue
        if ec_raw not in ec_info:
            ec_info[ec_raw] = {"group": "similar", "rxns": []}
        ec_info[ec_raw]["rxns"].append(str(row["rxn_name"]))

    for ec in control_ecs:
        ec = ec.strip()
        if ec not in ec_info:
            ec_info[ec] = {"group": "control", "rxns": ["control"]}

    n_sim = sum(1 for v in ec_info.values() if v["group"] == "similar")
    n_ctrl = sum(1 for v in ec_info.values() if v["group"] == "control")
    print(f"\n  EC groups: {n_sim} similar, {n_ctrl} control")
    for ec, info in sorted(ec_info.items()):
        tag = "SIMILAR" if info["group"] == "similar" else "CONTROL"
        print(f"    [{tag:7s}] EC {ec:14s}  ({', '.join(info['rxns'][:3])})")
    return ec_info


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 3: Fetch Sequences
# ═══════════════════════════════════════════════════════════════════════════

def fetch_uniprot_by_ec(ec, max_seqs=MAX_SEQS_PER_EC):
    url = (
        f"https://rest.uniprot.org/uniprotkb/search"
        f"?query=ec:{ec}+AND+reviewed:true"
        f"&format=fasta&size={max_seqs}"
    )
    print(f"  [UniProt] EC {ec} — fetching up to {max_seqs} seqs ...")
    try:
        resp = SESSION.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        records = list(SeqIO.parse(StringIO(resp.text), "fasta"))
        print(f"  [UniProt] EC {ec} — got {len(records)} sequences")
        return records
    except Exception as e:
        print(f"  WARNING: UniProt query failed for EC {ec}: {e}")
        return []


def build_sequences(ec_info):
    """Fetch sequences, return metadata DataFrame and list of sequences."""
    print(f"\n{'='*60}")
    print("Fetching sequences from UniProt")
    print(f"{'='*60}")

    rows = []
    records = []

    for ec, info in ec_info.items():
        group = info["group"]
        fasta_records = fetch_uniprot_by_ec(ec)
        time.sleep(0.3)

        for rec in fasta_records:
            parts = rec.id.split("|")
            acc = parts[1] if len(parts) > 1 else rec.id

            org = "unknown"
            if "OS=" in rec.description:
                org = (rec.description.split("OS=")[1]
                       .split("OX=")[0].strip()
                       .replace(" ", "_")[:40])

            seq = str(rec.seq)
            if len(seq) < 50:
                continue
            # Truncate for ESM-2 max length
            if len(seq) > MAX_SEQ_LEN:
                seq = seq[:MAX_SEQ_LEN]

            header = f"{acc}__EC_{ec}__{org}"
            rows.append({
                "header": header,
                "accession": acc,
                "ec": ec,
                "ec_class": ec.split(".")[0],
                "organism": org.replace("_", " "),
                "group": group,
                "seq_len": len(seq),
            })
            records.append(SeqRecord(Seq(seq), id=header, description=""))

    meta_df = pd.DataFrame(rows)
    print(f"\n  Total sequences: {len(meta_df)}")
    print(f"    Similar:  {(meta_df['group'] == 'similar').sum()}")
    print(f"    Control:  {(meta_df['group'] == 'control').sum()}")

    if records:
        SeqIO.write(records, SEQS_FILE, "fasta")
        print(f"  Written: {SEQS_FILE}")

    meta_df.to_csv(METADATA_CSV, index=False)
    return meta_df, [str(r.seq) for r in records]


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 4: ESM-2 Embeddings
# ═══════════════════════════════════════════════════════════════════════════

def compute_esm_embeddings(sequences, model_name=DEFAULT_ESM_MODEL,
                           batch_size=ESM_BATCH_SIZE):
    """
    Compute mean-pool ESM-2 embeddings for a list of protein sequences.
    Returns numpy array of shape (n_seqs, embed_dim).
    """
    print(f"\n{'='*60}")
    print(f"Computing ESM-2 Embeddings ({model_name})")
    print(f"{'='*60}")

    import esm

    # Load model
    print(f"  Loading model {model_name} ...")
    if model_name == "esm2_t48_15B_UR50D":
        model, alphabet = esm.pretrained.esm2_t48_15B_UR50D()
    elif model_name == "esm2_t36_3B_UR50D":
        model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    elif model_name == "esm2_t33_650M_UR50D":
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    elif model_name == "esm2_t30_150M_UR50D":
        model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    elif model_name == "esm2_t12_35M_UR50D":
        model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    elif model_name == "esm2_t6_8M_UR50D":
        model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    else:
        raise ValueError(f"Unknown ESM model: {model_name}")

    batch_converter = alphabet.get_batch_converter()
    n_layers = model.num_layers

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # For very large models, use half precision on GPU
    if "15B" in model_name and device.type == "cuda":
        model = model.half()
        print(f"  Using float16 for 15B model")

    model = model.to(device)
    model.eval()

    all_embeddings = []

    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i + batch_size]
        batch_data = [(f"seq_{i+j}", seq) for j, seq in enumerate(batch_seqs)]

        print(f"  Batch {i // batch_size + 1}/{(len(sequences) + batch_size - 1) // batch_size} "
              f"({len(batch_seqs)} seqs, max_len={max(len(s) for s in batch_seqs)}) ...")

        batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[n_layers],
                            return_contacts=False)

        # Mean-pool over sequence length (excluding BOS/EOS tokens)
        token_reps = results["representations"][n_layers]  # (B, L, D)

        for j in range(len(batch_seqs)):
            seq_len = len(batch_seqs[j])
            # tokens: [BOS, aa1, aa2, ..., EOS, PAD, ...]
            # take positions 1 to seq_len+1
            embedding = token_reps[j, 1:seq_len + 1, :].float().mean(dim=0)
            all_embeddings.append(embedding.cpu().numpy())

        # Free GPU memory
        del batch_tokens, results, token_reps
        if device.type == "cuda":
            torch.cuda.empty_cache()

    embeddings = np.stack(all_embeddings)
    print(f"\n  Embedding shape: {embeddings.shape}")
    np.save(EMBEDDINGS_NPY, embeddings)
    print(f"  Saved: {EMBEDDINGS_NPY}")

    return embeddings


# ═══════════════════════════════════════════════════════════════════════════
#  STEP 5: Visualization & Statistics
# ═══════════════════════════════════════════════════════════════════════════

def compute_cluster_statistics(embeddings, meta_df):
    """
    Compute inter- and intra-group distances, silhouette score, etc.
    """
    print(f"\n{'='*60}")
    print("Cluster Statistics")
    print(f"{'='*60}")

    groups = meta_df["group"].values
    ecs = meta_df["ec"].values

    # Pairwise cosine distances
    from scipy.spatial.distance import cosine
    n = len(embeddings)
    dist_matrix = squareform(pdist(embeddings, metric="cosine"))

    # Intra-similar distances
    sim_mask = groups == "similar"
    ctrl_mask = groups == "control"

    sim_indices = np.where(sim_mask)[0]
    ctrl_indices = np.where(ctrl_mask)[0]

    intra_sim = []
    for i in range(len(sim_indices)):
        for j in range(i + 1, len(sim_indices)):
            intra_sim.append(dist_matrix[sim_indices[i], sim_indices[j]])

    intra_ctrl = []
    for i in range(len(ctrl_indices)):
        for j in range(i + 1, len(ctrl_indices)):
            intra_ctrl.append(dist_matrix[ctrl_indices[i], ctrl_indices[j]])

    inter = []
    for i in sim_indices:
        for j in ctrl_indices:
            inter.append(dist_matrix[i, j])

    # Intra-EC distances for similar group
    unique_sim_ecs = meta_df[sim_mask]["ec"].unique()
    intra_ec = {}
    for ec in unique_sim_ecs:
        ec_idx = np.where((ecs == ec) & sim_mask)[0]
        if len(ec_idx) > 1:
            dists = []
            for i in range(len(ec_idx)):
                for j in range(i + 1, len(ec_idx)):
                    dists.append(dist_matrix[ec_idx[i], ec_idx[j]])
            intra_ec[ec] = np.mean(dists)

    # Silhouette score (similar vs control)
    labels = np.where(sim_mask, 1, 0)
    sil = silhouette_score(embeddings, labels, metric="cosine")

    stats_lines = [
        "ESM-2 Embedding Cluster Statistics",
        "=" * 50,
        f"Total sequences:          {n}",
        f"  Similar:                {sim_mask.sum()}",
        f"  Control:                {ctrl_mask.sum()}",
        "",
        "Cosine Distance Summary (lower = more similar):",
        f"  Intra-similar (mean):   {np.mean(intra_sim):.4f} ± {np.std(intra_sim):.4f}",
        f"  Intra-control (mean):   {np.mean(intra_ctrl):.4f} ± {np.std(intra_ctrl):.4f}",
        f"  Inter-group (mean):     {np.mean(inter):.4f} ± {np.std(inter):.4f}",
        "",
        "Intra-EC distances (similar group):",
    ]
    for ec, d in sorted(intra_ec.items()):
        stats_lines.append(f"  EC {ec:14s}:  {d:.4f}")

    stats_lines.extend([
        "",
        f"Silhouette Score (similar vs control): {sil:.4f}",
        "  (1.0 = perfect separation, 0.0 = overlapping, <0 = wrong clustering)",
        "",
    ])

    # Separation ratio
    if np.mean(inter) > 0:
        ratio = np.mean(intra_sim) / np.mean(inter)
        stats_lines.append(f"Separation ratio (intra-sim / inter): {ratio:.4f}")
        stats_lines.append(f"  (<1.0 means similar-group is tighter than inter-group → good)")

    stats_text = "\n".join(stats_lines)
    print(stats_text)

    with open(STATS_TXT, "w") as f:
        f.write(stats_text)
    print(f"\n  Stats saved: {STATS_TXT}")

    return dist_matrix


def draw_scatter(embeddings, meta_df, out_png, out_pdf=None):
    """
    2D scatter: PCA + t-SNE side by side.
    Similar = dark orange, control = gray.
    """
    print(f"\nRendering scatter plots ...")

    groups = meta_df["group"].values
    ecs = meta_df["ec"].values

    # PCA
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(embeddings)

    # t-SNE
    perp = min(30, max(5, len(embeddings) // 4))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42,
                metric="cosine", init="pca", n_iter=2000)
    tsne_coords = tsne.fit_transform(embeddings)

    # UMAP (optional)
    try:
        import umap
        umap_reducer = umap.UMAP(n_components=2, metric="cosine",
                                  n_neighbors=min(15, len(embeddings) - 1),
                                  random_state=42)
        umap_coords = umap_reducer.fit_transform(embeddings)
        has_umap = True
    except ImportError:
        has_umap = False
        print("  UMAP not available (pip install umap-learn). Skipping.")

    n_plots = 3 if has_umap else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 6))
    bg = "#fafafa"
    fig.patch.set_facecolor(bg)

    def _plot(ax, coords, title):
        ax.set_facecolor(bg)

        # Plot controls first (background)
        ctrl = groups == "control"
        ax.scatter(coords[ctrl, 0], coords[ctrl, 1],
                   c=CONTROL_COLOR, s=30, alpha=0.5, edgecolors="white",
                   linewidths=0.3, zorder=2, label="Control")

        # Plot similar on top (foreground)
        sim = groups == "similar"
        ax.scatter(coords[sim, 0], coords[sim, 1],
                   c=SIMILAR_COLOR, s=70, alpha=0.85, edgecolors="white",
                   linewidths=0.5, zorder=3, label="Similar")

        # Label similar points with EC
        for idx in np.where(sim)[0]:
            ec = ecs[idx]
            ax.annotate(f"EC {ec}", (coords[idx, 0], coords[idx, 1]),
                        fontsize=5.5, color=SIMILAR_COLOR, fontweight="bold",
                        xytext=(4, 4), textcoords="offset points",
                        path_effects=[pe.withStroke(linewidth=2, foreground="white")])

        ax.set_title(title, fontsize=12, fontweight="bold", color="#222")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#ccc")
        ax.spines["bottom"].set_color("#ccc")
        ax.tick_params(colors="#888", labelsize=7)

    _plot(axes[0], pca_coords,
          f"PCA (var explained: {pca.explained_variance_ratio_.sum():.1%})")
    axes[0].set_xlabel("PC1", fontsize=9, color="#666")
    axes[0].set_ylabel("PC2", fontsize=9, color="#666")

    _plot(axes[1], tsne_coords, f"t-SNE (perplexity={perp})")
    axes[1].set_xlabel("t-SNE 1", fontsize=9, color="#666")
    axes[1].set_ylabel("t-SNE 2", fontsize=9, color="#666")

    if has_umap:
        _plot(axes[2], umap_coords, "UMAP")
        axes[2].set_xlabel("UMAP 1", fontsize=9, color="#666")
        axes[2].set_ylabel("UMAP 2", fontsize=9, color="#666")

    # Shared legend
    handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=SIMILAR_COLOR,
               markersize=10, label='Similar reactions (FAISS)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=CONTROL_COLOR,
               markersize=8, label='Negative controls (30 ECs)'),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               fontsize=10, framealpha=0.9, edgecolor="#ddd",
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("ESM-2 Embedding Space — Reaction Similarity Validation",
                 fontsize=14, fontweight="bold", color="#222", y=1.02)

    plt.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  Scatter PNG: {out_png}")

    if out_pdf:
        fig.savefig(out_pdf, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Scatter PDF: {out_pdf}")
    plt.close(fig)


def draw_dendrogram(embeddings, meta_df, out_png, out_pdf=None):
    """
    Hierarchical clustering dendrogram (Ward linkage on cosine distances).
    Leaf labels colored by group.
    """
    print(f"\nRendering dendrogram ...")

    groups = meta_df["group"].values
    ecs = meta_df["ec"].values
    accessions = meta_df["accession"].values

    # Ward linkage on euclidean (of L2-normalized embeddings ≈ cosine)
    normed = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
    Z = linkage(normed, method="ward", metric="euclidean")

    # Build labels
    labels = [f"{acc} (EC {ec})" for acc, ec in zip(accessions, ecs)]
    label_colors = [SIMILAR_COLOR if g == "similar" else CONTROL_COLOR
                    for g in groups]

    n = len(labels)
    fig_height = max(8, n * 0.22)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    bg = "#fafafa"
    ax.set_facecolor(bg)
    fig.patch.set_facecolor(bg)

    # Draw dendrogram
    ddata = dendrogram(
        Z,
        orientation="right",
        labels=labels,
        leaf_font_size=6,
        ax=ax,
        color_threshold=0,     # all branches same color initially
        above_threshold_color=CONTROL_COLOR,
    )

    # Color leaf labels
    ylabels = ax.get_yticklabels()
    for yl in ylabels:
        txt = yl.get_text()
        # Find matching index
        for idx, lab in enumerate(labels):
            if txt == lab:
                yl.set_color(label_colors[idx])
                if groups[idx] == "similar":
                    yl.set_fontweight("bold")
                    yl.set_fontsize(7)
                else:
                    yl.set_fontsize(5.5)
                break

    ax.set_title("Hierarchical Clustering of ESM-2 Embeddings",
                 fontsize=13, fontweight="bold", color="#222", pad=12)
    ax.set_xlabel("Ward linkage distance", fontsize=10, color="#444")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#ccc")
    ax.tick_params(axis="x", colors="#888", labelsize=8)

    # Legend
    patches = [
        mpatches.Patch(facecolor=SIMILAR_COLOR, edgecolor="white",
                       label="Similar reactions (FAISS)"),
        mpatches.Patch(facecolor=CONTROL_COLOR, edgecolor="white",
                       label="Negative controls"),
    ]
    ax.legend(handles=patches, loc="lower right", framealpha=0.9,
              facecolor="white", edgecolor="#ddd", fontsize=9)

    plt.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  Dendrogram PNG: {out_png}")

    if out_pdf:
        fig.savefig(out_pdf, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Dendrogram PDF: {out_pdf}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    global MAX_SEQS_PER_EC

    parser = argparse.ArgumentParser(
        description="Reaction Similarity → ESM-2 Embedding Validation"
    )
    parser.add_argument("--rxn_name", type=str, required=True,
                        help="Query reaction ID (e.g. rxn00646)")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of similar reactions (default: 10)")
    parser.add_argument("--control_ecs", nargs="*", default=None,
                        help="EC numbers for negative controls")
    parser.add_argument("--max_seqs", type=int, default=MAX_SEQS_PER_EC,
                        help=f"Max sequences per EC (default: {MAX_SEQS_PER_EC})")
    parser.add_argument("--esm_model", type=str, default=DEFAULT_ESM_MODEL,
                        choices=["esm2_t48_15B_UR50D", "esm2_t36_3B_UR50D",
                                 "esm2_t33_650M_UR50D", "esm2_t30_150M_UR50D",
                                 "esm2_t12_35M_UR50D", "esm2_t6_8M_UR50D"],
                        help=f"ESM-2 model (default: {DEFAULT_ESM_MODEL})")
    parser.add_argument("--batch_size", type=int, default=ESM_BATCH_SIZE,
                        help=f"ESM batch size (default: {ESM_BATCH_SIZE})")
    parser.add_argument("--skip_esm", action="store_true",
                        help="Skip ESM inference, load existing embeddings")
    args = parser.parse_args()

    control_ecs = args.control_ecs if args.control_ecs else DEFAULT_CONTROL_ECS
    MAX_SEQS_PER_EC = args.max_seqs

    print("=" * 60)
    print("Reaction Similarity → ESM-2 Embedding Validation")
    print("=" * 60)

    # Step 1: FAISS search
    result_df = run_similarity_search(args.rxn_name, args.top_k)

    # Step 2: Collect EC groups
    print(f"\n{'='*60}")
    print("Collecting EC groups (similar + controls)")
    print(f"{'='*60}")
    ec_info = collect_ec_groups(result_df, control_ecs)

    # Step 3: Fetch sequences
    meta_df, sequences = build_sequences(ec_info)

    if len(sequences) < 4:
        print("\nERROR: Too few sequences fetched.")
        sys.exit(1)

    # Step 4: ESM-2 embeddings
    if args.skip_esm and os.path.exists(EMBEDDINGS_NPY):
        print(f"\n  Loading existing embeddings: {EMBEDDINGS_NPY}")
        embeddings = np.load(EMBEDDINGS_NPY)
        meta_df = pd.read_csv(METADATA_CSV)
    else:
        embeddings = compute_esm_embeddings(
            sequences,
            model_name=args.esm_model,
            batch_size=args.batch_size,
        )

    # Step 5: Statistics
    dist_matrix = compute_cluster_statistics(embeddings, meta_df)

    # Step 6: Visualize
    print(f"\n{'='*60}")
    print("Visualization")
    print(f"{'='*60}")
    draw_scatter(embeddings, meta_df, SCATTER_PNG, SCATTER_PDF)
    draw_dendrogram(embeddings, meta_df, DENDRO_PNG, DENDRO_PDF)

    # Summary
    print(f"\n{'='*60}")
    print("Done! Output files:")
    for f in [QUERY_RESULT_CSV, SEQS_FILE, METADATA_CSV, EMBEDDINGS_NPY,
              SCATTER_PNG, SCATTER_PDF, DENDRO_PNG, DENDRO_PDF, STATS_TXT]:
        sz = os.path.getsize(f) if os.path.exists(f) else 0
        print(f"  {f}  ({sz:,} bytes)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()