#!/usr/bin/env python
"""Rung 1: unsupervised MULTI-VIEW FUSION vs the best single views.
Views are complementary (CGR wins full-EC retrieval; descriptors win EC3/full-EC
clustering; ECFP is the all-rounder), so fusing them should beat each alone.

Fusion = per-view standardize/L2-normalize -> concat -> TruncatedSVD 256 -> L2.
Evaluated on the CGR-defined subset (fair) at EC3 and full-EC, clustering + kNN.
CGR channel is zeroed where unavailable so fusion also works on the full set.
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_mutual_info_score, homogeneity_completeness_v_measure
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from collections import Counter

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
DATA = os.path.join(HERE, "data"); RESULTS = os.path.join(HERE, "results"); CACHE = os.path.join(HERE, "cache"); EMB = os.path.join(HERE, "embeddings")
df = pd.read_parquet(os.path.join(DATA, "dataset.parquet"))
mask = np.load(os.path.join(DATA, "cgr_mask.npy"))
CONT = {"descriptors_diff"}

def view(name, idx):
    X = np.load(os.path.join(EMB, name + ".npy"))[idx]
    X = StandardScaler().fit_transform(X) if name in CONT else X.astype(np.float32)
    X[~np.isfinite(X)] = 0.0
    return normalize(X).astype(np.float32)

def to_space(X):
    Z = TruncatedSVD(n_components=min(256, X.shape[1]-1), random_state=0).fit_transform(X)
    z = np.linalg.norm(Z, axis=1) < 1e-9
    if z.any(): Z[z] = 0.0; Z[z, 0] = 1.0
    return normalize(Z).astype(np.float32)

def evalu(Z, idx):
    d = pdist(Z, "cosine"); L = linkage(d, "average")
    pred = fcluster(L, t=800, criterion="maxclust")
    out = {}
    for lvl in ("ec3", "ec"):
        lt = df[lvl].to_numpy()[idx]
        out[f"{lvl}_ami"] = round(adjusted_mutual_info_score(lt, pred), 4)
        out[f"{lvl}_homog"] = round(homogeneity_completeness_v_measure(lt, pred)[0], 4)
    nn = NearestNeighbors(n_neighbors=16, metric="cosine", n_jobs=6).fit(Z)
    _, nb = nn.kneighbors(Z); nb = nb[:, 1:]
    for k in (1, 15):
        for lvl in ("ec3", "ec"):
            lt = df[lvl].to_numpy()[idx]
            pr = np.array([Counter(lt[nb[i, :k]]).most_common(1)[0][0] for i in range(len(idx))])
            out[f"{lvl}_knn{k}"] = round((pr == lt).mean(), 4)
    return out

idx = np.where(mask)[0]
configs = {
    "ecfp4_diff (single)":       ["ecfp4_diff"],
    "descriptors (single)":      ["descriptors_diff"],
    "cgr_morgan (single)":       ["cgr_morgan"],
    "FUSE ecfp+desc":            ["ecfp4_diff", "descriptors_diff"],
    "FUSE ecfp+desc+cgr":        ["ecfp4_diff", "descriptors_diff", "cgr_morgan"],
    "FUSE all4 +drfp":           ["ecfp4_diff", "descriptors_diff", "cgr_morgan", "drfp"],
}
print(f"subset n={len(idx)} (CGR-defined)\n")
print(f"{'config':24s} {'EC3 AMI':>8} {'EC3 homog':>10} {'EC3 15NN':>9} "
      f"{'fullEC AMI':>11} {'fullEC 1NN':>11} {'fullEC 15NN':>12}")
for cname, views in configs.items():
    X = np.hstack([view(v, idx) for v in views])
    r = evalu(to_space(X), idx)
    print(f"{cname:24s} {r['ec3_ami']:>8} {r['ec3_homog']:>10} {r['ec3_knn15']:>9} "
          f"{r['ec_ami']:>11} {r['ec_knn1']:>11} {r['ec_knn15']:>12}")
