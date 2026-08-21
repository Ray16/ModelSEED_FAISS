#!/usr/bin/env python
"""Greedy forward-selection over reaction-embedding VIEWS to find the best fusion
combo (is ecfp+desc+cgr actually optimal?). Start empty; at each step add the view
that most improves EC3 AMI (agglo-average k=800) on the CGR-defined subset; stop
when no view helps by >0.002. Reports the marginal gain of each added view.
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

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
DATA = os.path.join(HERE, "data"); RESULTS = os.path.join(HERE, "results")
EMB = os.path.join(HERE, "embeddings")
df = pd.read_parquet(os.path.join(DATA, "dataset.parquet"))
mask = np.load(os.path.join(DATA, "cgr_mask.npy")); idx = np.where(mask)[0]
CONT = {"descriptors_diff", "rxnfp", "chemberta_diff", "molformer_diff"}

POOL = ["ecfp4_diff", "ecfp6_diff", "atompair_diff", "maccs_diff", "secfp_diff",
        "descriptors_diff", "drfp", "cgr_morgan", "cgr_linear", "rxnfp", "chemberta_diff"]

_cache = {}
def view(name):
    if name in _cache: return _cache[name]
    X = np.load(os.path.join(EMB, name + ".npy"))[idx]
    X = StandardScaler().fit_transform(X) if name in CONT else X.astype(np.float32)
    X[~np.isfinite(X)] = 0.0; X = normalize(X).astype(np.float32)
    _cache[name] = X; return X

def ec3_ami(views):
    X = np.hstack([view(v) for v in views])
    Z = TruncatedSVD(n_components=min(256, X.shape[1]-1), random_state=0).fit_transform(X)
    zz = np.linalg.norm(Z, axis=1) < 1e-9
    if zz.any(): Z[zz] = 0.0; Z[zz, 0] = 1.0
    Z = normalize(Z)
    d = pdist(Z, "cosine"); L = linkage(d, "average")
    pred = fcluster(L, t=800, criterion="maxclust")
    return adjusted_mutual_info_score(df.ec3.to_numpy()[idx], pred), Z

def full_metrics(Z):
    out = {}
    for lvl in ("ec3", "ec"):
        lt = df[lvl].to_numpy()[idx]
        d = pdist(Z, "cosine"); pred = fcluster(linkage(d, "average"), t=800, criterion="maxclust")
        out[lvl+"_ami"] = round(adjusted_mutual_info_score(lt, pred), 4)
    nn = NearestNeighbors(n_neighbors=16, metric="cosine", n_jobs=6).fit(Z)
    _, nb = nn.kneighbors(Z); nb = nb[:, 1:]
    for k in (1, 15):
        for lvl in ("ec3", "ec"):
            lt = df[lvl].to_numpy()[idx]
            pr = np.array([Counter(lt[nb[i, :k]]).most_common(1)[0][0] for i in range(len(idx))])
            out[f"{lvl}_knn{k}"] = round((pr == lt).mean(), 4)
    return out

selected, cur = [], 0.0
print(f"greedy forward selection on CGR subset (n={len(idx)}), objective=EC3 AMI\n")
while True:
    best_gain, best_v = 0.0, None
    for v in POOL:
        if v in selected: continue
        ami, _ = ec3_ami(selected + [v])
        gain = ami - cur
        if gain > best_gain: best_gain, best_v, best_ami = gain, v, ami
    if best_v is None or best_gain < 0.002:
        print(f"\nstop: no view improves EC3 AMI by >0.002")
        break
    selected.append(best_v); cur = best_ami
    print(f"+ {best_v:18s} -> EC3 AMI {cur:.4f}  (gain +{best_gain:.4f})")

print(f"\nSELECTED COMBO: {' + '.join(selected)}")
_, Zf = ec3_ami(selected)
m = full_metrics(Zf)
print("final metrics:", m)
