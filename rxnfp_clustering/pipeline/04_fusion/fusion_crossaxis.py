#!/usr/bin/env python
"""Cross-axis fusion: fuse ONE representative per *orthogonal representational
axis* -- {descriptors_diff (physchem), cgr_morgan (atom-mapped bond-change),
gin_diff (pretrained 2D-graph GNN), unimol_diff (3D geometry),
erg_diff (pharmacophore reduced-graph)} -- and compare against the
previously-best SAME-FAMILY-style fusion (PhysChem+CGR+ChemBERTa, both
"global descriptor + exact bond-change + a SMILES-string transformer") from
Part II of RESULTS.md.

Both fusions are re-scored HERE, on the SAME fair common subset (the
intersection of every mask involved: CGR composes for ~75%, GIN featurizes
for ~76%, Uni-Mol/ErG are ~100%), so the comparison is apples-to-apples --
Part II's 0.696-0.718 numbers were computed on the (larger) CGR-only subset
and are not directly comparable to a 5-view intersection.

Same recipe as fusion_figure.py / triple_combos.py: per-view
normalize -> concat -> TruncatedSVD-256 -> L2-normalize (cosine) ->
agglomerative average-linkage, k=800. Primary metric EC3 AMI; also EC3
15-NN accuracy and full-EC 1-NN accuracy.
-> results/crossaxis_fusion.csv, figures/fig_crossaxis_fusion.png
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_mutual_info_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from collections import Counter
import sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "_lib"))
import _style  # project rule: all figure text >= 18pt, dpi 300
FS = _style.FS

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(HERE, "data"); EMB = os.path.join(HERE, "embeddings")
RESULTS = os.path.join(HERE, "results"); FIG = os.path.join(HERE, "figures")
df = pd.read_parquet(os.path.join(DATA, "dataset.parquet"))
N = len(df)

# ---- fair common subset: intersection of every mask involved ----
def load_mask(path, default_true=True):
    if os.path.exists(path):
        return np.load(path)
    return np.ones(N, bool) if default_true else np.zeros(N, bool)

cgr_mask = load_mask(os.path.join(DATA, "cgr_mask.npy"))
erg_mask = load_mask(os.path.join(EMB, "erg_diff_mask.npy"))
gin_mask = load_mask(os.path.join(EMB, "gin_diff_mask.npy"))
unimol_mask = load_mask(os.path.join(EMB, "unimol_diff_mask.npy"))
common = cgr_mask & erg_mask & gin_mask & unimol_mask
idx = np.where(common)[0]
print(f"masks: cgr={cgr_mask.sum()} erg={erg_mask.sum()} gin={gin_mask.sum()} "
      f"unimol={unimol_mask.sum()} -> common fair subset n={len(idx)} ({100*len(idx)/N:.1f}%)")

CONT = {"descriptors_diff", "chemberta_diff", "gin_diff", "unimol_diff"}
_c = {}
def view(n):
    if n in _c: return _c[n]
    X = np.load(os.path.join(EMB, n + ".npy"))[idx]
    X = StandardScaler().fit_transform(X) if n in CONT else X.astype(np.float32)
    X[~np.isfinite(X)] = 0.0
    _c[n] = normalize(X).astype(np.float32)
    return _c[n]

def evalu(views):
    X = np.hstack([view(v) for v in views])
    Z = TruncatedSVD(n_components=min(256, X.shape[1]-1), random_state=0).fit_transform(X)
    zz = np.linalg.norm(Z, axis=1) < 1e-9
    if zz.any(): Z[zz] = 0.0; Z[zz, 0] = 1.0
    Z = normalize(Z)
    d = pdist(Z, "cosine")
    pred = fcluster(linkage(d, "average"), t=800, criterion="maxclust")
    ami = adjusted_mutual_info_score(df.ec3.to_numpy()[idx], pred)
    nn = NearestNeighbors(n_neighbors=16, metric="cosine", n_jobs=6).fit(Z)
    _, nb = nn.kneighbors(Z); nb = nb[:, 1:]
    def knn(k, lvl):
        lt = df[lvl].to_numpy()[idx]
        pr = np.array([Counter(lt[nb[i, :k]]).most_common(1)[0][0] for i in range(len(idx))])
        return (pr == lt).mean()
    return ami, knn(15, "ec3"), knn(1, "ec")

CONFIGS = [
    ("PhysChem", ["descriptors_diff"], "single"),
    ("CGR", ["cgr_morgan"], "single"),
    ("GIN", ["gin_diff"], "single"),
    ("Uni-Mol", ["unimol_diff"], "single"),
    ("ErG", ["erg_diff"], "single"),
    ("Same-family: PhysChem+CGR+ChemBERTa", ["descriptors_diff", "cgr_morgan", "chemberta_diff"], "samefam"),
    ("Cross-axis: PhysChem+CGR+GIN+UniMol+ErG", ["descriptors_diff", "cgr_morgan", "gin_diff", "unimol_diff", "erg_diff"], "crossaxis"),
    # Deployment question: does the BEST novel axis (or all of them) add anything ON TOP OF
    # the incumbent same-family winner? (Uni-Mol is the strongest single novel view here.)
    ("Incumbent + Uni-Mol", ["descriptors_diff", "cgr_morgan", "chemberta_diff", "unimol_diff"], "augment"),
    ("Incumbent + all novel", ["descriptors_diff", "cgr_morgan", "chemberta_diff", "gin_diff", "unimol_diff", "erg_diff"], "augment"),
]
rows = []
for name, vs, kind in CONFIGS:
    ami, k15, kfull = evalu(vs)
    rows.append((name, kind, ami, k15, kfull))
    print(f"{name:42s} EC3_AMI={ami:.4f} EC3_15NN={k15:.4f} fullEC_1NN={kfull:.4f}")

out = pd.DataFrame(rows, columns=["config", "kind", "ec3_ami", "ec3_15nn", "full_ec_1nn"])
out.to_csv(os.path.join(RESULTS, "crossaxis_fusion.csv"), index=False)

samefam_ami = out[out.kind == "samefam"].ec3_ami.iloc[0]
crossaxis_ami = out[out.kind == "crossaxis"].ec3_ami.iloc[0]
verdict = ("BEATS" if crossaxis_ami > samefam_ami else
          "TIES" if abs(crossaxis_ami - samefam_ami) < 0.003 else "LOSES TO")
print(f"\nVERDICT: cross-axis fusion ({crossaxis_ami:.4f}) {verdict} "
      f"same-family fusion ({samefam_ami:.4f}) on the n={len(idx)} common subset "
      f"(delta={crossaxis_ami-samefam_ami:+.4f})")

# Deployment verdict: does augmenting the incumbent winner with novel axes help?
for label in ("Incumbent + Uni-Mol", "Incumbent + all novel"):
    a = out[out.config == label].ec3_ami.iloc[0]
    d = a - samefam_ami
    v = "HELPS" if d >= 0.003 else "HURTS" if d <= -0.003 else "NEUTRAL"
    print(f"AUGMENT: {label} ({a:.4f}) {v} vs incumbent ({samefam_ami:.4f}) delta={d:+.4f}")

# ---- figure ----
labels = [r[0] for r in rows]
metrics = [("EC3 AMI (cluster)", 2), ("EC3 acc (15-NN)", 3), ("full-EC acc (1-NN)", 4)]
x = np.arange(len(labels)); w = 0.26
mcolors = ["#8aa0b6", "#E69F00", "#009E73"]
fig, ax = plt.subplots(figsize=(20, 9.5))
for j, (mlabel, mi) in enumerate(metrics):
    vals = [r[mi] for r in rows]
    bars = ax.bar(x + (j-1)*w, vals, w, label=mlabel, color=mcolors[j])
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.010, f"{v:.2f}", ha="center", fontsize=FS)
for i, r in enumerate(rows):
    if r[1] in ("samefam", "crossaxis", "augment"):
        c = {"crossaxis": "#009E73", "samefam": "#CC79A7", "augment": "#E69F00"}[r[1]]
        ax.axvspan(i-0.45, i+0.45, color=c, alpha=0.08, zorder=0)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=FS)
ax.set_ylim(0, 1.08); ax.set_ylabel("score", fontsize=FS)
ax.tick_params(axis="y", labelsize=FS)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.grid(axis="y", alpha=0.25); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=FS, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
ax.set_title(f"Cross-axis fusion vs. same-family fusion  (fair 5-way subset, n={len(idx)})",
            fontsize=FS, pad=34)
fig.tight_layout()
out_fig = os.path.join(FIG, "fig_crossaxis_fusion.png")
fig.savefig(out_fig, dpi=300, bbox_inches="tight")
print("wrote", out_fig)
