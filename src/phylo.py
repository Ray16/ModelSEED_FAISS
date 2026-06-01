#!/usr/bin/env python
"""
Phylogenetic tree pipeline for similar reactions.

Steps: FAISS search -> fetch UniProt sequences -> CD-HIT dedup ->
       MAFFT alignment -> FastTree -> visualization.

Resumable: skips steps whose output files already exist (use --force to overwrite).

Usage:
    conda activate rxnfp
    python -m src.phylo --rxn_name rxn00044
    python -m src.phylo --rxn_name rxn00044 --top_k 100 --force
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from io import StringIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from Bio import AlignIO, Phylo, SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import (
    FAISS_INDEX_FILE,
    PHYLO_ALIGNED,
    PHYLO_METADATA_CSV,
    PHYLO_SEQS_CDHIT,
    PHYLO_SEQS_RAW,
    PHYLO_STATS_TXT,
    PHYLO_TREE_NWK,
    PHYLO_TREE_PNG,
    QUERY_RESULT_CSV,
    RXN_DATA_CSV,
    RXN_FINGERPRINTS_NPY,
)
from src.utils import l2_normalize_vectors, load_faiss_index, parse_ec_numbers

MAX_SEQS_PER_EC = 8
MAFFT_BIN = "mafft"
MAFFT_THREADS = 4
REQUEST_TIMEOUT = 60
CDHIT_IDENTITY = 0.9


def get_session():
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


SESSION = get_session()


def run_similarity_search(rxn_name, top_k, force=False):
    if os.path.exists(QUERY_RESULT_CSV) and not force:
        print(f"  [Resume] Loading existing FAISS results: {QUERY_RESULT_CSV}")
        return pd.read_csv(QUERY_RESULT_CSV)

    print(f"\nFAISS Similarity Search: {rxn_name} (top {top_k})")
    index = load_faiss_index(FAISS_INDEX_FILE)
    df = pd.read_csv(RXN_DATA_CSV)

    matches = df[df.id == rxn_name].index.to_numpy()
    if len(matches) == 0:
        print(f"ERROR: Reaction '{rxn_name}' not found")
        sys.exit(1)

    query_idx = matches[0]
    qvec = np.load(RXN_FINGERPRINTS_NPY)[query_idx].astype(np.float32).reshape(1, -1)
    qvec = l2_normalize_vectors(qvec)
    D, I = index.search(qvec, top_k)

    result_df = pd.DataFrame({
        "similarity_ranking": list(range(1, len(D[0]) + 1)),
        "rxn_name": [df.iloc[idx, 0] for idx in I[0]],
        "ec_number": [parse_ec_numbers(df.iloc[idx, 3]) for idx in I[0]],
        "distance": D[0],
    })
    result_df.to_csv(QUERY_RESULT_CSV, index=False)
    return result_df


def build_sequences(ec_info, force=False):
    if os.path.exists(PHYLO_SEQS_RAW) and os.path.exists(PHYLO_METADATA_CSV) and not force:
        print(f"  [Resume] Loading existing sequences and metadata.")
        return pd.read_csv(PHYLO_METADATA_CSV)

    print(f"\nFetching sequences for {len(ec_info)} unique EC numbers...")
    records, metadata = [], []

    for ec, info in ec_info.items():
        url = (
            f"https://rest.uniprot.org/uniprotkb/search"
            f"?query=ec:{ec}+AND+reviewed:true&format=fasta&size={MAX_SEQS_PER_EC}"
        )
        try:
            resp = SESSION.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            fasta_records = list(SeqIO.parse(StringIO(resp.text), "fasta"))
            time.sleep(0.3)

            for rec in fasta_records:
                acc = rec.id.split("|")[1] if "|" in rec.id else rec.id
                org = (
                    rec.description.split("OS=")[1].split("OX=")[0].strip().replace(" ", "_")[:40]
                    if "OS=" in rec.description else "unknown"
                )
                header = f"{acc}__EC_{ec}__{org}"
                records.append(SeqRecord(rec.seq, id=header, description=""))
                metadata.append({"header": header, "ec": ec, "distance": info["avg_dist"], "organism": org.replace("_", " ")})
        except Exception as e:
            print(f"  WARNING: EC {ec} failed: {e}")

    meta_df = pd.DataFrame(metadata)
    SeqIO.write(records, PHYLO_SEQS_RAW, "fasta")
    meta_df.to_csv(PHYLO_METADATA_CSV, index=False)
    return meta_df


def run_cdhit(in_fasta, out_fasta, force=False):
    if os.path.exists(out_fasta) and not force:
        print(f"  [Resume] Using existing CD-HIT file: {out_fasta}")
        return

    print("\nRunning CD-HIT Deduplication...")
    cdhit_bin = shutil.which("cd-hit")
    cmd = [cdhit_bin, "-i", in_fasta, "-o", out_fasta, "-c", str(CDHIT_IDENTITY), "-n", "5", "-d", "0"]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)


def run_mafft(in_fasta, out_fasta, force=False):
    if os.path.exists(out_fasta) and not force:
        print(f"  [Resume] Using existing MAFFT alignment: {out_fasta}")
        return

    print("\nRunning MAFFT alignment...")
    cmd = [MAFFT_BIN, "--auto", "--thread", str(MAFFT_THREADS), in_fasta]
    with open(out_fasta, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"MAFFT failed: {result.stderr.decode()}")


def build_tree(aligned_fasta, tree_nwk, force=False):
    if os.path.exists(tree_nwk) and not force:
        print(f"  [Resume] Loading existing tree: {tree_nwk}")
        return Phylo.read(tree_nwk, "newick")

    print("\nBuilding tree with FastTree...")
    fasttree_bin = shutil.which("FastTree") or shutil.which("fasttree")
    cmd = [fasttree_bin, "-wag", "-gamma", aligned_fasta]
    with open(tree_nwk, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE)

    tree = Phylo.read(tree_nwk, "newick")
    tree.root_at_midpoint()
    Phylo.write(tree, tree_nwk, "newick")
    return tree


def get_lookup(name, mapping):
    if not name:
        return None
    if name in mapping:
        return mapping[name]
    for k, v in mapping.items():
        if k in name or name in k:
            return v
    return None


def assign_clade_dist(clade, dist_map):
    clade_dist = {}

    def _recurse(c):
        if c.is_terminal():
            d = get_lookup(c.name, dist_map)
            clade_dist[id(c)] = d
            return [d] if d is not None else []
        child_ds = []
        for child in c.clades:
            child_ds.extend(_recurse(child))
        valid = [d for d in child_ds if d is not None]
        clade_dist[id(c)] = np.mean(valid) if valid else None
        return valid

    _recurse(clade)
    return clade_dist


def draw_rectangular_tree(tree, meta_df, out_png, rxn_name):
    print("Rendering tree...")

    dist_map = dict(zip(meta_df["header"], meta_df["distance"]))
    clade_dist = assign_clade_dist(tree.root, dist_map)

    original_cmap = cm.get_cmap("inferno")
    colors = original_cmap(np.linspace(0, 0.85, 256))
    truncated_cmap = mcolors.ListedColormap(colors)

    norm = mcolors.Normalize(vmin=meta_df["distance"].min(), vmax=meta_df["distance"].max())

    fig, ax = plt.subplots(figsize=(16, len(tree.get_terminals()) * 0.25 + 3))
    bg = "#ffffff"
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    y_coords = {id(leaf): i for i, leaf in enumerate(tree.get_terminals())}
    x_coords = {}

    def get_x(clade, x=0):
        x += clade.branch_length or 0
        x_coords[id(clade)] = x
        if not clade.is_terminal():
            child_ys = [get_x(child, x) for child in clade.clades]
            y_coords[id(clade)] = np.mean(child_ys)
        return y_coords[id(clade)]

    get_x(tree.root)

    for clade in tree.find_clades(order="level"):
        if clade == tree.root:
            continue
        parent = tree.get_path(clade)[-2] if len(tree.get_path(clade)) > 1 else tree.root

        dist_val = clade_dist.get(id(clade))
        color = mcolors.to_hex(truncated_cmap(norm(dist_val))) if dist_val is not None else "#333333"

        ax.plot([x_coords[id(parent)], x_coords[id(clade)]], [y_coords[id(clade)], y_coords[id(clade)]], color=color, lw=3.0, solid_capstyle="round", zorder=2)
        ax.plot([x_coords[id(parent)], x_coords[id(parent)]], [y_coords[id(parent)], y_coords[id(clade)]], color=color, lw=3.0, solid_capstyle="round", zorder=2)

        if clade.is_terminal():
            display_name = clade.name.replace("__", " ").replace("_", " ") if clade.name else ""
            dynamic_offset = 0.05
            ax.text(x_coords[id(clade)] + dynamic_offset, y_coords[id(clade)], display_name, va="center", fontsize=8.5, fontweight="bold", color=color, fontfamily="monospace")

    max_x = max(x_coords.values())
    ax.plot([0, 0.1], [-1.5, -1.5], color="black", lw=2)
    ax.text(0.05, -2.5, "0.1 substitutions/site", ha="center", fontsize=9, fontweight="bold")

    sm = plt.cm.ScalarMappable(cmap=truncated_cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.4, pad=0.15, aspect=25)
    cbar.set_label("Cosine Similarity", fontweight="bold", labelpad=10)

    ax.set_ylim(-3, len(tree.get_terminals()) + 1)
    ax.set_xlim(0, max_x * 1.6)
    ax.axis("off")

    plt.savefig(out_png, bbox_inches="tight", dpi=300, facecolor=bg)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Resumable Phylogenetic Pipeline")
    parser.add_argument("--rxn_name", required=True)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--force", action="store_true", help="Overwrite existing progress")
    args = parser.parse_args()

    if args.force:
        print("!! Force flag detected: Overwriting all existing results.")

    result_df = run_similarity_search(args.rxn_name, args.top_k, force=args.force)

    ec_distances = defaultdict(list)
    for _, row in result_df.iterrows():
        ec_raw = str(row["ec_number"]).strip("[]'\" ").split("'")[0].strip()
        if not ec_raw or ec_raw == "nan":
            continue
        ec_distances[ec_raw].append(float(row["distance"]))
    ec_info = {ec: {"avg_dist": float(np.mean(dists))} for ec, dists in ec_distances.items()}

    meta_df = build_sequences(ec_info, force=args.force)

    run_cdhit(PHYLO_SEQS_RAW, PHYLO_SEQS_CDHIT, force=args.force)

    survivors = {rec.id for rec in SeqIO.parse(PHYLO_SEQS_CDHIT, "fasta")}
    meta_df = meta_df[meta_df["header"].isin(survivors)].reset_index(drop=True)

    run_mafft(PHYLO_SEQS_CDHIT, PHYLO_ALIGNED, force=args.force)

    tree = build_tree(PHYLO_ALIGNED, PHYLO_TREE_NWK, force=args.force)

    draw_rectangular_tree(tree, meta_df, PHYLO_TREE_PNG, args.rxn_name)
    print(f"\nDone! Results saved in data/ and figures/")


if __name__ == "__main__":
    main()
