#!/usr/bin/env python
"""'Our method' figure: single views vs multi-view FUSION on the fair CGR subset.
Clean grouped bars across EC3 AMI, EC3 15-NN, full-EC 1-NN. -> figures/fig_fusion.png
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
DATA = os.path.join(HERE, "data"); EMB = os.path.join(HERE, "embeddings"); FIG = os.path.join(HERE, "figures", "benchmark")
df = pd.read_parquet(os.path.join(DATA, "dataset.parquet"))
mask = np.load(os.path.join(DATA, "cgr_mask.npy")); idx = np.where(mask)[0]
CONT = {"descriptors_diff"}
_c = {}
def view(n):
    if n in _c: return _c[n]
    X = np.load(os.path.join(EMB, n+".npy"))[idx]
    X = StandardScaler().fit_transform(X) if n in CONT else X.astype(np.float32)
    X[~np.isfinite(X)] = 0.0; _c[n] = normalize(X).astype(np.float32); return _c[n]
def evalu(views):
    X = np.hstack([view(v) for v in views])
    Z = TruncatedSVD(n_components=min(256, X.shape[1]-1), random_state=0).fit_transform(X)
    zz = np.linalg.norm(Z, axis=1) < 1e-9
    if zz.any(): Z[zz] = 0.0; Z[zz, 0] = 1.0
    Z = normalize(Z)
    d = pdist(Z, "cosine"); pred = fcluster(linkage(d, "average"), t=800, criterion="maxclust")
    ami = adjusted_mutual_info_score(df.ec3.to_numpy()[idx], pred)
    nn = NearestNeighbors(n_neighbors=16, metric="cosine", n_jobs=6).fit(Z)
    _, nb = nn.kneighbors(Z); nb = nb[:, 1:]
    def knn(k, lvl):
        lt = df[lvl].to_numpy()[idx]
        pr = np.array([Counter(lt[nb[i, :k]]).most_common(1)[0][0] for i in range(len(idx))])
        return (pr == lt).mean()
    return ami, knn(15, "ec3"), knn(1, "ec")

CONFIGS = [
    ("ECFP4-diff", ["ecfp4_diff"], "single"),
    ("PhysChem", ["descriptors_diff"], "single"),
    ("CGR", ["cgr_morgan"], "single"),
    ("ECFP4+DRFP", ["ecfp4_diff", "drfp"], "single"),
    ("Fuse: ECFP+PhysChem", ["ecfp4_diff", "descriptors_diff"], "fusion"),
    ("Fuse: ECFP+PhysChem+CGR", ["ecfp4_diff", "descriptors_diff", "cgr_morgan"], "fusion"),
]
rows = []
for name, vs, kind in CONFIGS:
    ami, k15, kfull = evalu(vs); rows.append((name, kind, ami, k15, kfull))
    print(f"{name:24s} EC3_AMI={ami:.4f} EC3_15NN={k15:.4f} fullEC_1NN={kfull:.4f}")

labels = [r[0] for r in rows]
metrics = [("EC3 AMI (cluster)", 2), ("EC3 acc (15-NN)", 3), ("full-EC acc (1-NN)", 4)]
x = np.arange(len(labels)); w = 0.26
COL = {"single": "#8aa0b6", "fusion": "#0072B2"}
fig, ax = plt.subplots(figsize=(17, 9))
mcolors = ["#8aa0b6", "#E69F00", "#009E73"]
for j, (mlabel, mi) in enumerate(metrics):
    vals = [r[mi] for r in rows]
    bars = ax.bar(x + (j-1)*w, vals, w, label=mlabel, color=mcolors[j])
    for b, v in zip(bars, vals): ax.text(b.get_x()+b.get_width()/2, v+0.010, f"{v:.2f}", ha="center", fontsize=FS)
# mark fusion group
for i, r in enumerate(rows):
    if r[1] == "fusion": ax.axvspan(i-0.45, i+0.45, color="#0072B2", alpha=0.06, zorder=0)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=FS)
ax.set_ylim(0, 1.08); ax.set_ylabel("score", fontsize=FS)
ax.tick_params(axis="y", labelsize=FS)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.grid(axis="y", alpha=0.25); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=FS, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.09))
ax.set_title(f"Multi-view fusion beats every single view  (fair CGR subset, n={len(idx)})", fontsize=FS, pad=28)
fig.tight_layout()
out = os.path.join(FIG, "fig_fusion.png"); fig.savefig(out, dpi=300, bbox_inches="tight")
print("wrote", out)
