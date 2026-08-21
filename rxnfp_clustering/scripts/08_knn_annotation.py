#!/usr/bin/env python
"""kNN EC-annotation-transfer evaluation (retrieval counterpart to clustering).

Question: if a reaction's EC were unknown, how well could we recover it from its
nearest neighbours in each embedding space? This is what a FAISS reaction index
would actually do. Leave-one-out: for every reaction, look at its k nearest
neighbours (excluding itself), majority-vote their EC at each level, and measure
accuracy. Also report accuracy-vs-coverage when we only trust confident calls
(fraction of neighbours agreeing >= threshold).

Same geometry as the clustering benchmark: L2-normalize -> TruncatedSVD 256 ->
L2-normalize (cosine). Writes knn_results.csv.
"""
import os, sys, time, warnings, argparse
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from collections import Counter
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
DATA = os.path.join(HERE, "data"); RESULTS = os.path.join(HERE, "results"); CACHE = os.path.join(HERE, "cache")
EMB = os.path.join(HERE, "embeddings")
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, file=sys.stderr, flush=True)

df = pd.read_parquet(os.path.join(DATA, "dataset.parquet"))
LEVELS = ("ec", "ec3", "ec2", "ec1")
LAB = {lvl: df[lvl].to_numpy() for lvl in LEVELS}
N = len(df)
CONTINUOUS = {"rxnfp", "rxnfp_pretrained", "chemberta_diff", "descriptors_diff",
              "molformer_diff", "rxnfp_moldiff", "rxnfp_sidediff"}

def make_space(name, Xraw):
    X = StandardScaler().fit_transform(Xraw) if name in CONTINUOUS else Xraw.astype(np.float32)
    X[~np.isfinite(X)] = 0.0
    X = normalize(X)
    d = min(256, X.shape[1] - 1)
    Z = TruncatedSVD(n_components=d, random_state=0).fit_transform(X)
    z = np.linalg.norm(Z, axis=1) < 1e-9
    if z.any(): Z[z] = 0.0; Z[z, 0] = 1.0
    return normalize(Z).astype(np.float32)

def evaluate(name):
    p = os.path.join(EMB, name + ".npy")
    if not os.path.exists(p): return []
    Z = make_space(name, np.load(p))
    nn = NearestNeighbors(n_neighbors=16, metric="cosine", n_jobs=6).fit(Z)
    _, idx = nn.kneighbors(Z)
    idx = idx[:, 1:]  # drop self
    rows = []
    for k in (1, 5, 15):
        neigh = idx[:, :k]
        out = {"embedding": name, "k": k}
        for lvl in LEVELS:
            lab = LAB[lvl]
            preds, conf = [], []
            for i in range(N):
                votes = Counter(lab[neigh[i]])
                best, c = votes.most_common(1)[0]
                preds.append(best); conf.append(c / k)
            preds = np.array(preds); conf = np.array(conf)
            correct = preds == lab
            out[f"{lvl}_acc"] = round(correct.mean(), 4)
            # accuracy among confident calls (>=0.6 of neighbours agree) + their coverage
            if k > 1:
                m = conf >= 0.6
                out[f"{lvl}_acc_conf"] = round(correct[m].mean(), 4) if m.any() else 0.0
                out[f"{lvl}_cov_conf"] = round(m.mean(), 4)
        rows.append(out)
    log(f"{name}: 1-NN EC3 acc={rows[0]['ec3_acc']}  15-NN EC3 acc={rows[-1]['ec3_acc']}")
    return rows

ALL_EMB = ["ecfp4_diff", "ecfp2_diff", "ecfp6_diff", "ecfp4_concat", "ecfp4_drfp",
           "drfp", "atompair_diff", "rdk_diff", "torsion_diff", "avalon_diff",
           "maccs_diff", "secfp_diff", "descriptors_diff",
           "rxnfp", "rxnfp_pretrained", "chemberta_diff", "molformer_diff", "rxnfp_moldiff", "rxnfp_sidediff", "cgr_linear", "cgr_morgan"]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    out = os.path.join(RESULTS, "knn_results.csv")
    rows = []
    for e in (a.only or ALL_EMB):
        rows += evaluate(e)
    new = pd.DataFrame(rows)
    # merge-safe: with --only, replace those embeddings' rows in the existing file
    if a.only and os.path.exists(out):
        old = pd.read_csv(out)
        old = old[~old.embedding.isin(a.only)]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(out, index=False)
    log(f"wrote knn_results.csv ({len(new)} rows total)")

if __name__ == "__main__":
    main()
