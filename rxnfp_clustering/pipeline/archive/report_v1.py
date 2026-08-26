#!/usr/bin/env python
"""Summarize results.csv into ranked tables + figures."""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
DATA = os.path.join(HERE, "data"); RESULTS = os.path.join(HERE, "results"); CACHE = os.path.join(HERE, "cache")
df = pd.read_csv(os.path.join(RESULTS, "results.csv"))
FIG = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)

# Pretty names / order
EMB_ORDER = ["ecfp4_diff", "ecfp4_concat", "ecfp4_drfp", "drfp", "atompair_diff",
             "rdk_diff", "maccs_diff", "rxnfp"]
present = [e for e in EMB_ORDER if e in df.embedding.unique()]

# ---- Best operating point per (embedding, algo) by EC3 AMI ----
def best_by(metric):
    idx = df.groupby(["embedding", "algo"])[metric].idxmax()
    return df.loc[idx].sort_values(metric, ascending=False)

print("\n=== TOP 20 configs by EC3 AMI (primary) ===")
top = df.sort_values("ec3_ami", ascending=False).head(20)
cols = ["embedding", "algo", "params", "n_clusters", "noise_frac",
        "ec3_ami", "ec3_homogeneity", "ec3_vmeasure", "ec3_purity",
        "ec_ami", "ec_homogeneity"]
print(top[cols].to_string(index=False))

print("\n=== Best config per embedding (by EC3 AMI) ===")
bestemb = df.loc[df.groupby("embedding")["ec3_ami"].idxmax()].sort_values("ec3_ami", ascending=False)
print(bestemb[cols].to_string(index=False))

print("\n=== Best config per algorithm (by EC3 AMI) ===")
bestalgo = df.loc[df.groupby("algo")["ec3_ami"].idxmax()].sort_values("ec3_ami", ascending=False)
print(bestalgo[cols].to_string(index=False))

top.to_csv(os.path.join(RESULTS, "summary_top20.csv"), index=False)
bestemb.to_csv(os.path.join(RESULTS, "summary_best_per_embedding.csv"), index=False)

# ---- Fig 1: AMI vs #clusters, KMeans, per embedding (fair curve) ----
plt.figure(figsize=(8, 5.5))
sub = df[df.algo == "minibatch_kmeans"]
for e in present:
    s = sub[sub.embedding == e].sort_values("n_clusters")
    if len(s): plt.plot(s.n_clusters, s.ec3_ami, marker="o", label=e)
plt.xscale("log"); plt.xlabel("# clusters (KMeans)"); plt.ylabel("EC3 AMI")
plt.title("Embedding comparison — KMeans, EC subclass (EC3) AMI")
plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(FIG, "kmeans_ami_vs_k.png"), dpi=140); plt.close()

# ---- Fig 2: homogeneity vs completeness tradeoff (all configs) ----
plt.figure(figsize=(7.5, 6))
markers = {"minibatch_kmeans": "o", "agglo_avg_cosine": "s", "agglo_ward": "^",
           "birch": "D", "hdbscan": "*", "butina": "P"}
cmap = plt.get_cmap("tab10")
ecolor = {e: cmap(i % 10) for i, e in enumerate(present)}
for _, r in df.iterrows():
    plt.scatter(r.ec3_completeness, r.ec3_homogeneity,
                marker=markers.get(r.algo, "x"), color=ecolor.get(r.embedding, "gray"),
                s=40, alpha=0.7)
for e in present:
    plt.scatter([], [], color=ecolor[e], label=e, s=40)
for a, m in markers.items():
    plt.scatter([], [], marker=m, color="gray", label=a, s=40)
plt.xlabel("EC3 completeness"); plt.ylabel("EC3 homogeneity")
plt.title("Homogeneity vs completeness (all configs)")
plt.legend(fontsize=7, ncol=2); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(FIG, "homogeneity_completeness.png"), dpi=140); plt.close()

# ---- Fig 3: best-per-embedding bar chart across metrics ----
metrics = ["ec3_ami", "ec3_vmeasure", "ec3_homogeneity", "ec3_purity", "ec_ami"]
be = bestemb.set_index("embedding").reindex(present)
x = np.arange(len(present)); w = 0.16
plt.figure(figsize=(10, 5.5))
for j, m in enumerate(metrics):
    plt.bar(x + (j - 2) * w, be[m].values, w, label=m)
plt.xticks(x, present, rotation=20, ha="right")
plt.ylabel("score"); plt.title("Best config per embedding (selected by EC3 AMI)")
plt.legend(fontsize=8); plt.grid(alpha=0.3, axis="y"); plt.tight_layout()
plt.savefig(os.path.join(FIG, "best_per_embedding_bars.png"), dpi=140); plt.close()

print(f"\nwrote figures to {FIG}")
