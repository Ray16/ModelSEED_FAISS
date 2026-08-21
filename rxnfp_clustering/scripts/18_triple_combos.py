#!/usr/bin/env python
"""Compare ALL 3-view fusion combos from a curated pool.
Objective: EC3 AMI (agglo-average k=800) + full-EC 1-NN, on the CGR-defined subset.
-> results/triple_combos.csv and figures/fig_triple_combos.png
"""
import os, warnings, itertools
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _style  # project rule: all figure text >= 18pt, dpi 300
FS = _style.FS

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data"); EMB = os.path.join(HERE, "embeddings")
RESULTS = os.path.join(HERE, "results"); FIG = os.path.join(HERE, "figures")
df = pd.read_parquet(os.path.join(DATA, "dataset.parquet"))
mask = np.load(os.path.join(DATA, "cgr_mask.npy")); idx = np.where(mask)[0]
CONT = {"descriptors_diff", "chemberta_diff", "rxnfp"}

POOL = ["descriptors_diff", "ecfp4_diff", "drfp", "secfp_diff",
        "cgr_morgan", "atompair_diff", "chemberta_diff"]
SHORT = {"descriptors_diff": "PhysChem", "ecfp4_diff": "ECFP4", "drfp": "DRFP",
         "secfp_diff": "MHFP", "cgr_morgan": "CGR", "atompair_diff": "AtomPair",
         "chemberta_diff": "ChemBERTa"}

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
    nn = NearestNeighbors(n_neighbors=2, metric="cosine", n_jobs=6).fit(Z)
    _, nb = nn.kneighbors(Z); nb = nb[:, 1:]
    lt = df.ec.to_numpy()[idx]
    full1 = (lt[nb[:, 0]] == lt).mean()
    return ami, full1

rows = []
for combo in itertools.combinations(POOL, 3):
    ami, f1 = evalu(list(combo))
    rows.append({"combo": "+".join(SHORT[c] for c in combo), "ec3_ami": round(ami, 4),
                 "full_ec_1nn": round(f1, 4)})
    print(f"{'+'.join(SHORT[c] for c in combo):28s} EC3_AMI={ami:.4f} fullEC_1NN={f1:.4f}")
res = pd.DataFrame(rows).sort_values("ec3_ami", ascending=False)
res.to_csv(os.path.join(RESULTS, "triple_combos.csv"), index=False)

# plot top 15 by EC3 AMI, horizontal grouped bars
top = res.head(15).iloc[::-1]
y = np.arange(len(top)); h = 0.40
fig, ax = plt.subplots(figsize=(15, 12))
ax.barh(y+h/2, top.ec3_ami, h, color="#0072B2", label="EC3 AMI (cluster)")
ax.barh(y-h/2, top.full_ec_1nn, h, color="#E69F00", label="full-EC 1-NN (retrieval)")
for yi, (a, f) in enumerate(zip(top.ec3_ami, top.full_ec_1nn)):
    ax.text(a+0.005, yi+h/2, f"{a:.3f}", va="center", fontsize=FS)
    ax.text(f+0.005, yi-h/2, f"{f:.3f}", va="center", fontsize=FS)
ax.set_yticks(y); ax.set_yticklabels(top.combo, fontsize=FS)
ax.tick_params(axis="x", labelsize=FS)
ax.set_xlim(0, max(top.ec3_ami.max(), top.full_ec_1nn.max())+0.12)
ax.set_xlabel("score", fontsize=FS); ax.set_title(f"3-view fusion combos ranked by EC3 AMI (CGR subset, n={len(idx)})", fontsize=FS)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.grid(axis="x", alpha=0.25); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=FS, loc="upper center", bbox_to_anchor=(0.5,-0.06), ncol=2)
fig.tight_layout(); out = os.path.join(FIG, "fig_triple_combos.png")
fig.savefig(out, dpi=300, bbox_inches="tight"); print("\nwrote", out)
print("\nBEST triple:", res.iloc[0].to_dict())
