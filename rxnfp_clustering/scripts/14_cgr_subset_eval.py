#!/usr/bin/env python
"""Fair CGR comparison: score CGR vs incumbents on the SAME reaction subsets
(where CGR is defined / high-confidence), since CGR only covers 74.8% and atom-
mapping confidence is low on enzymatic reactions. Isolates 'is CGR better where
it applies?' from 'CGR coverage is limited'. Focus: EC3 and full-EC.
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

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
DATA = os.path.join(HERE, "data"); RESULTS = os.path.join(HERE, "results"); CACHE = os.path.join(HERE, "cache"); EMB = os.path.join(HERE, "embeddings")
df = pd.read_parquet(os.path.join(DATA, "dataset.parquet"))
mask = np.load(os.path.join(DATA, "cgr_mask.npy"))
conf = np.load(os.path.join(DATA, "cgr_confidence.npy"))
CONT = {"descriptors_diff"}

def space(name, idx):
    X = np.load(os.path.join(EMB, name + ".npy"))[idx]
    X = StandardScaler().fit_transform(X) if name in CONT else X.astype(np.float32)
    X[~np.isfinite(X)] = 0.0; X = normalize(X)
    Z = TruncatedSVD(n_components=min(256, X.shape[1]-1), random_state=0).fit_transform(X)
    z = np.linalg.norm(Z, axis=1) < 1e-9
    if z.any(): Z[z] = 0.0; Z[z, 0] = 1.0
    return normalize(Z).astype(np.float32)

def evalu(name, idx, labels):
    Z = space(name, idx)
    # clustering: agglo average cosine k=800
    d = pdist(Z, "cosine"); L = linkage(d, "average")
    pred = fcluster(L, t=800, criterion="maxclust")
    out = {}
    for lvl in ("ec3", "ec"):
        lt = labels[lvl].to_numpy()[idx]
        out[f"{lvl}_ami"] = round(adjusted_mutual_info_score(lt, pred), 4)
        out[f"{lvl}_homog"] = round(homogeneity_completeness_v_measure(lt, pred)[0], 4)
    # kNN 1-NN and 15-NN
    nn = NearestNeighbors(n_neighbors=16, metric="cosine", n_jobs=6).fit(Z)
    _, nb = nn.kneighbors(Z); nb = nb[:, 1:]
    for k in (1, 15):
        for lvl in ("ec3", "ec"):
            lt = labels[lvl].to_numpy()[idx]
            pr = np.array([Counter(lt[nb[i, :k]]).most_common(1)[0][0] for i in range(len(idx))])
            out[f"{lvl}_knn{k}"] = round((pr == lt).mean(), 4)
    return out

EMB_LIST = ["cgr_morgan", "cgr_linear", "ecfp4_diff", "ecfp4_drfp", "descriptors_diff"]
subsets = {
    "mapped (n=%d)" % mask.sum(): np.where(mask)[0],
    "high-conf>=0.5 (n=%d)" % (mask & (conf >= 0.5)).sum(): np.where(mask & (conf >= 0.5))[0],
}
for sname, idx in subsets.items():
    print(f"\n=== subset: {sname} ===")
    print(f"{'embedding':16s} {'EC3 AMI':>8} {'EC3 homog':>10} {'EC3 1NN':>8} {'EC3 15NN':>9} "
          f"{'fullEC AMI':>11} {'fullEC 1NN':>11} {'fullEC 15NN':>12}")
    for e in EMB_LIST:
        r = evalu(e, idx, df)
        print(f"{e:16s} {r['ec3_ami']:>8} {r['ec3_homog']:>10} {r['ec3_knn1']:>8} {r['ec3_knn15']:>9} "
              f"{r['ec_ami']:>11} {r['ec_knn1']:>11} {r['ec_knn15']:>12}")
