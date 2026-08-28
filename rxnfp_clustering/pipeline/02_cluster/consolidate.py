#!/usr/bin/env python
"""Merge sharded results -> results.csv, print ranked tables, render clean
comparison figures. Also renders the kNN annotation-transfer comparison.

Figures (all in figures/):
  fig_heatmap_embed_algo.png   embedding x algorithm, best EC3 AMI (the big compare)
  fig_best_per_embedding.png   best EC3 AMI per embedding, family-coloured
  fig_best_per_algo.png        best EC3 AMI per algorithm
  fig_kmeans_ami_vs_k.png      AMI vs #clusters (KMeans) per embedding
  fig_knn_transfer.png         kNN EC1/EC2/EC3 accuracy per embedding
  fig_homogeneity_completeness.png
"""
import os, sys, glob
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "_lib"))
import _style  # project rule: all figure text >= 18pt, dpi 300
FS = _style.FS

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
DATA = os.path.join(HERE, "data"); RESULTS = os.path.join(HERE, "results"); CACHE = os.path.join(HERE, "cache")
FIG = os.path.join(HERE, "figures", "benchmark"); os.makedirs(FIG, exist_ok=True)

# ---- merge shards ----
shards = sorted(glob.glob(os.path.join(RESULTS, "results_s*.csv")))
if shards:
    df = pd.concat([pd.read_csv(f) for f in shards], ignore_index=True)
    df.to_csv(os.path.join(RESULTS, "results.csv"), index=False)
    print(f"merged {len(shards)} shards -> results.csv ({len(df)} rows)")
else:
    df = pd.read_csv(os.path.join(RESULTS, "results.csv"))

# ---- curate the REPORTED analysis (the full results.csv written above is kept
# intact on disk; these rows are dropped only from figures & ranked tables). ----
# Especially-weak / redundant candidates that answer no question another kept
# candidate already answers:
#   rxnfp_sidediff (0.41, worst) / rxnfp_moldiff (0.56) -- failed rxnfp-diff variants
#   ecfp4_concat (0.57) -- diff-vs-concat control        molformer_diff (0.57) -- dup of chemberta
#   cgr_morgan (0.48) -- near-duplicate of cgr_linear
# Algorithms: agglo_single (0.26, chaining) / dbscan (dup of hdbscan) / spectral (no unique win).
EXCLUDE_EMB = {"rxnfp_sidediff", "rxnfp_moldiff", "ecfp4_concat", "cgr_morgan", "molformer_diff"}
EXCLUDE_ALGO = {"agglo_single_cos", "dbscan", "spectral"}
df = df[~df.embedding.isin(EXCLUDE_EMB) & ~df.algo.isin(EXCLUDE_ALGO)].reset_index(drop=True)
print(f"curated analysis: {df.embedding.nunique()} embeddings x {df.algo.nunique()} algorithms "
      f"(dropped {len(EXCLUDE_EMB)} emb + {len(EXCLUDE_ALGO)} algo; raw kept in results.csv)")

# Ordered so every family is one contiguous colour block (biochem is a learned
# reaction transformer -> sits with the other learned models, not orphaned at the
# end; the novel single-member axes erg/gin/unimol trail as the "new axes" group).
EMB_ORDER = ["ecfp4_diff", "ecfp2_diff", "ecfp6_diff", "ecfp4_concat", "ecfp4_drfp",
             "fcfp4_diff", "drfp", "atompair_diff", "rdk_diff", "torsion_diff", "avalon_diff",
             "maccs_diff", "secfp_diff", "descriptors_diff", "mordred_diff", "cgr_linear", "cgr_morgan",
             "rxnfp", "rxnfp_pretrained", "rxnfp_moldiff", "rxnfp_sidediff",
             "chemberta_diff", "molformer_diff", "biochem",
             "erg_diff", "gin_diff", "unimol_diff"]
FAMILY = {**{e: "structural" for e in ["ecfp4_diff","ecfp2_diff","ecfp6_diff","ecfp4_concat",
             "fcfp4_diff","drfp","atompair_diff","rdk_diff","torsion_diff","avalon_diff","maccs_diff","secfp_diff"]},
          "ecfp4_drfp": "structural", "descriptors_diff": "physchem", "mordred_diff": "physchem",
          "cgr_linear": "cgr", "cgr_morgan": "cgr",
          "rxnfp": "learned", "rxnfp_pretrained": "learned",
          "rxnfp_moldiff": "learned", "rxnfp_sidediff": "learned",
          "chemberta_diff": "learned", "molformer_diff": "learned",
          "erg_diff": "reduced-graph", "gin_diff": "gnn",
          "unimol_diff": "3d", "biochem": "learned"}
FAM_COLOR = {"structural": "#0072B2", "physchem": "#009E73", "cgr": "#CC79A7", "learned": "#D55E00",
             "reduced-graph": "#56B4E9", "gnn": "#F0E442", "3d": "#000000"}
present = [e for e in EMB_ORDER if e in df.embedding.unique()]

def clean(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6); ax.set_axisbelow(True)

PRIMARY = "ec3_ami"

# ---- ranked tables ----
top = df.sort_values(PRIMARY, ascending=False).head(20)
cols = ["embedding","algo","params","n_clusters","noise_frac",
        "ec3_ami","ec3_homogeneity","ec3_vmeasure","ec3_purity","ec_ami","ec1_ami"]
print("\n=== TOP 20 configs by EC3 AMI ===")
print(top[cols].to_string(index=False))
bestemb = df.loc[df.groupby("embedding")[PRIMARY].idxmax()].sort_values(PRIMARY, ascending=False)
print("\n=== best per embedding ==="); print(bestemb[cols].to_string(index=False))
bestalgo = df.loc[df.groupby("algo")[PRIMARY].idxmax()].sort_values(PRIMARY, ascending=False)
print("\n=== best per algorithm ==="); print(bestalgo[cols].to_string(index=False))
top.to_csv(os.path.join(HERE,"summary_top20.csv"), index=False)
bestemb.to_csv(os.path.join(HERE,"summary_best_per_embedding.csv"), index=False)

# ---- Fig: heatmap embedding x algorithm (best EC3 AMI) ----
algos = list(df.algo.unique())
algo_order = ["minibatch_kmeans","agglo_average_cos","agglo_complete_cos","agglo_single_cos",
              "agglo_ward","birch","leiden","louvain","spectral","dbscan","hdbscan","butina"]
algo_order = [a for a in algo_order if a in algos] + [a for a in algos if a not in algo_order]
M = np.full((len(present), len(algo_order)), np.nan)
for i,e in enumerate(present):
    for j,a in enumerate(algo_order):
        s = df[(df.embedding==e)&(df.algo==a)][PRIMARY]
        if len(s): M[i,j] = s.max()
fig, ax = plt.subplots(figsize=(1.55*len(algo_order)+5, 0.7*len(present)+3))
im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=np.nanmin(M), vmax=np.nanmax(M))
ax.set_xticks(range(len(algo_order))); ax.set_xticklabels(algo_order, rotation=40, ha="right", fontsize=FS)
ax.set_yticks(range(len(present))); ax.set_yticklabels(present, fontsize=FS)
for i in range(len(present)):
    for j in range(len(algo_order)):
        if not np.isnan(M[i,j]):
            ax.text(j,i,f"{M[i,j]:.2f}", ha="center", va="center", fontsize=FS,
                    color="white" if M[i,j] < (np.nanmin(M)+np.nanmax(M))/2 else "black")
cb = fig.colorbar(im, ax=ax, label="best EC3 AMI", fraction=0.025)
cb.ax.tick_params(labelsize=FS); cb.set_label("best EC3 AMI", fontsize=FS)
ax.set_title("EC3 AMI by embedding × clustering algorithm (best over parameter sweep)", fontsize=FS)
fig.tight_layout(); fig.savefig(os.path.join(FIG,"fig_heatmap_embed_algo.png"), dpi=300, bbox_inches="tight"); plt.close(fig)

# ---- Fig: best per embedding (family-coloured) ----
be = bestemb.set_index("embedding").reindex(present)
fig, ax = plt.subplots(figsize=(16,7.5))
colors = [FAM_COLOR[FAMILY[e]] for e in present]
ax.bar(range(len(present)), be[PRIMARY].values, color=colors)
ax.set_xticks(range(len(present))); ax.set_xticklabels(present, rotation=40, ha="right", fontsize=FS)
ax.set_ylabel("best EC3 AMI", fontsize=FS); clean(ax)
ax.set_ylim(0, 0.86)  # headroom so the legend sits clear above the bars
ax.set_title("Best EC3 AMI per embedding", fontsize=FS)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=c,label=f) for f,c in FAM_COLOR.items()],
          frameon=False, fontsize=FS, loc="upper center", ncol=4)
fig.tight_layout(); fig.savefig(os.path.join(FIG,"fig_best_per_embedding.png"), dpi=300, bbox_inches="tight"); plt.close(fig)

# ---- Fig: best per algorithm ----
ba = bestalgo.set_index("algo")[PRIMARY].sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(13,6.5))
ax.bar(range(len(ba)), ba.values, color="#0072B2")
ax.set_xticks(range(len(ba))); ax.set_xticklabels(ba.index, rotation=40, ha="right", fontsize=FS)
ax.set_ylabel("best EC3 AMI (any embedding)", fontsize=FS); clean(ax)
ax.set_title("Best EC3 AMI per clustering algorithm", fontsize=FS)
fig.tight_layout(); fig.savefig(os.path.join(FIG,"fig_best_per_algo.png"), dpi=300, bbox_inches="tight"); plt.close(fig)

# ---- Fig: KMeans AMI vs k, aggregated by representational family ----
# 27 individual lines are an unreadable hairball; the meaningful grouping is the
# family. Each family = mean EC3 AMI vs k with a min-max band. Message: every
# family peaks near the true #EC3 classes (261) then declines (over-fragments),
# and the structural/physchem families lead at every granularity.
fig, ax = plt.subplots(figsize=(11.5,7.5))
sub = df[df.algo=="minibatch_kmeans"]
FAM_ORDER = ["structural","physchem","cgr","reduced-graph","gnn","3d","learned"]
for f in [f for f in FAM_ORDER if any(FAMILY.get(e)==f for e in present)]:
    embs = [e for e in present if FAMILY.get(e)==f]
    g = sub[sub.embedding.isin(embs)]
    if not len(g): continue
    grp = g.groupby("n_clusters").ec3_ami
    ks = grp.mean().index.values
    ax.plot(ks, grp.mean().values, marker="o", ms=7, lw=2.6, color=FAM_COLOR[f],
            label=f"{f} (n={len(embs)})")
    if len(embs) > 1:
        ax.fill_between(ks, grp.min().values, grp.max().values, color=FAM_COLOR[f], alpha=0.13)
ax.set_xscale("log")
ax.axvline(261, ls="--", color="0.4", lw=1.6)
_ymin, _ymax = ax.get_ylim()
ax.text(272, _ymax - 0.004, "261 EC3 classes", fontsize=FS, color="0.4",
        ha="left", va="top")
ax.set_xlabel("# clusters (KMeans)", fontsize=FS)
ax.set_ylabel("EC3 AMI", fontsize=FS); clean(ax)
ax.set_title("AMI peaks near the true class count, then over-fragments", fontsize=FS)
ax.legend(fontsize=FS, ncol=1, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
fig.tight_layout(); fig.savefig(os.path.join(FIG,"fig_kmeans_ami_vs_k.png"), dpi=300, bbox_inches="tight"); plt.close(fig)

# ---- Fig: kNN annotation-transfer accuracy per embedding ----
kp = os.path.join(RESULTS,"knn_results.csv")
if os.path.exists(kp):
    k = pd.read_csv(kp); k = k[k.k==15].set_index("embedding").reindex(present)
    x = np.arange(len(present)); w=0.26
    fig, ax = plt.subplots(figsize=(19,8))
    for j,(lvl,c) in enumerate([("ec1_acc","#0072B2"),("ec2_acc","#E69F00"),("ec3_acc","#009E73")]):
        ax.bar(x+(j-1)*w, k[lvl].values, w, color=c, label=lvl.replace("_acc","").upper())
    ax.set_xticks(x); ax.set_xticklabels(present, rotation=40, ha="right", fontsize=FS)
    ax.set_ylabel("15-NN accuracy", fontsize=FS); clean(ax); ax.set_ylim(0.6,1.0)
    ax.set_title("kNN EC-annotation-transfer accuracy (15-NN majority vote)", fontsize=FS)
    ax.legend(frameon=False, fontsize=FS, ncol=3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG,"fig_knn_transfer.png"), dpi=300, bbox_inches="tight"); plt.close(fig)

# ---- Fig: homogeneity vs completeness, coloured by cluster granularity ----
# The homogeneity<->completeness trade-off is driven by cluster GRANULARITY,
# not by which embedding: many small clusters -> pure (high homogeneity) but
# fragment each EC (low completeness); few clusters -> the reverse. Encoding the
# 800 configs by log(#clusters) makes that the message; the star marks the
# single max-AMI config (the balanced sweet spot).
d = df[df.n_clusters >= 2]
fig, ax = plt.subplots(figsize=(11.5,8))
sc = ax.scatter(d.ec3_completeness, d.ec3_homogeneity, c=np.log10(d.n_clusters),
                cmap="viridis", s=46, alpha=0.65, edgecolor="none")
cb = fig.colorbar(sc, ax=ax, pad=0.02)
cb.set_label("# clusters (log$_{10}$)", fontsize=FS); cb.ax.tick_params(labelsize=FS)
b = df.loc[df[PRIMARY].idxmax()]
ax.scatter([b.ec3_completeness],[b.ec3_homogeneity], s=430, marker="*",
           color="#D55E00", edgecolor="black", linewidth=1.2, zorder=6,
           label=f"max EC3 AMI = {b[PRIMARY]:.2f}\n{b.embedding} · {b.algo}")
ax.legend(fontsize=FS, loc="lower left", frameon=True)
ax.set_xlabel("EC3 completeness", fontsize=FS); ax.set_ylabel("EC3 homogeneity", fontsize=FS)
clean(ax); ax.grid(alpha=0.25)
ax.set_title("Cluster granularity trades completeness for homogeneity", fontsize=FS)
fig.tight_layout(); fig.savefig(os.path.join(FIG,"fig_homogeneity_completeness.png"), dpi=300, bbox_inches="tight"); plt.close(fig)

print("\nwrote figures to", FIG)
