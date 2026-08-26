#!/usr/bin/env python
"""Expanded benchmark: many embeddings x many similarity/clustering methods.

Adds over the original cluster_eval (removed; see git history):
  - EC1 scoring + configurable primary level
  - graph community detection (Leiden multi-resolution, Louvain) on a mutual-kNN
    cosine similarity graph  -- the "build a similarity graph, then cluster it" paradigm
  - Spectral clustering (nearest-neighbour affinity)
  - DBSCAN (eps sweep)
  - agglomerative linkage sweep: average / complete / single (cosine) + ward (euclidean)
  - continuous embeddings (rxnfp*, chemberta, descriptors, molformer) standardized

Geometry: every embedding L2-normalized -> TruncatedSVD to common 256-d -> L2-norm
(cosine). Butina additionally runs on raw binarized bits. Writes results.csv.
"""
import os, sys, time, warnings, argparse
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import (MiniBatchKMeans, Birch, DBSCAN, SpectralClustering)
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (homogeneity_completeness_v_measure,
                             adjusted_mutual_info_score, adjusted_rand_score)
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
DATA = os.path.join(HERE, "data"); RESULTS = os.path.join(HERE, "results"); CACHE = os.path.join(HERE, "cache")
EMB = os.path.join(HERE, "embeddings")
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, file=sys.stderr, flush=True)

df = pd.read_parquet(os.path.join(DATA, "dataset.parquet"))
LEVELS = ("ec", "ec3", "ec2", "ec1")
LABELS = {lvl: df[lvl].to_numpy() for lvl in LEVELS}
N = len(df)
SVD_DIM = 256
CONTINUOUS = {"rxnfp", "rxnfp_pretrained", "chemberta_diff", "descriptors_diff",
              "molformer_diff", "rxnfp_moldiff", "rxnfp_sidediff",
              "gin_diff", "unimol_diff", "biochem", "mordred_diff"}

def purity(pred, lt):
    mask = pred >= 0
    if mask.sum() == 0: return 0.0
    lp, l = pred[mask], lt[mask]
    return sum(np.unique(l[lp == c], return_counts=True)[1].max() for c in np.unique(lp)) / mask.sum()

def score(emb, algo, params, pred, n_clusters):
    row = dict(embedding=emb, algo=algo, params=params, n_clusters=int(n_clusters),
               noise_frac=round(float((pred < 0).mean()), 4))
    for lvl in LEVELS:
        lt = LABELS[lvl]
        h, c, v = homogeneity_completeness_v_measure(lt, pred)
        row.update({f"{lvl}_homogeneity": round(h, 4), f"{lvl}_completeness": round(c, 4),
                    f"{lvl}_vmeasure": round(v, 4),
                    f"{lvl}_ami": round(adjusted_mutual_info_score(lt, pred), 4),
                    f"{lvl}_ari": round(adjusted_rand_score(lt, pred), 4),
                    f"{lvl}_purity": round(purity(pred, lt), 4)})
    return row

def make_space(name, Xraw):
    X = StandardScaler().fit_transform(Xraw) if name in CONTINUOUS else Xraw.astype(np.float32)
    X[~np.isfinite(X)] = 0.0
    X = normalize(X)
    d = min(SVD_DIM, X.shape[1] - 1)
    Z = TruncatedSVD(n_components=d, random_state=0).fit_transform(X)
    z = np.linalg.norm(Z, axis=1) < 1e-9
    if z.any(): Z[z] = 0.0; Z[z, 0] = 1.0
    return normalize(Z).astype(np.float32)

def knn_graph(Z, k=30):
    """Mutual-kNN cosine similarity graph as igraph."""
    import igraph as ig
    nn = NearestNeighbors(n_neighbors=k+1, metric="cosine").fit(Z)
    dist, idx = nn.kneighbors(Z)
    edges, weights = [], []
    for i in range(Z.shape[0]):
        for j, d in zip(idx[i, 1:], dist[i, 1:]):
            a, b = (i, int(j)) if i < j else (int(j), i)
            edges.append((a, b)); weights.append(1.0 - d)  # cosine similarity
    g = ig.Graph(n=Z.shape[0], edges=edges)
    g.es["weight"] = weights
    g.simplify(combine_edges="max")
    return g

def run_embedding(name):
    path = os.path.join(EMB, name + ".npy")
    if not os.path.exists(path):
        log(f"skip {name}: no npy"); return []
    Xraw = np.load(path)
    Z = make_space(name, Xraw)
    log(f"{name}: raw={Xraw.shape} -> Z={Z.shape}")
    rows = []
    # ---- MiniBatchKMeans grid ----
    for k in (50, 100, 200, 400, 800):
        pred = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=3,
                               batch_size=4096, max_iter=100).fit_predict(Z)
        rows.append(score(name, "minibatch_kmeans", f"k={k}", pred, k))
    log("  kmeans done")
    # ---- Agglomerative linkage sweep (pdist once per metric) ----
    t = time.time()
    dcos = pdist(Z, "cosine")
    for meth in ("average", "complete", "single"):
        Lk = linkage(dcos, meth)
        for k in (100, 400, 800):
            pred = fcluster(Lk, t=k, criterion="maxclust")
            rows.append(score(name, f"agglo_{meth}_cos", f"k={k}", pred, len(np.unique(pred))))
    del dcos
    deuc = pdist(Z, "euclidean"); Lk = linkage(deuc, "ward"); del deuc
    for k in (100, 400, 800):
        pred = fcluster(Lk, t=k, criterion="maxclust")
        rows.append(score(name, "agglo_ward", f"k={k}", pred, len(np.unique(pred))))
    log(f"  agglo sweep ({time.time()-t:.0f}s)")
    # ---- Birch ----
    for k in (200, 800):
        pred = Birch(n_clusters=k, threshold=0.3).fit_predict(Z)
        rows.append(score(name, "birch", f"k={k}", pred, len(np.unique(pred))))
    # ---- Graph community detection: Leiden + Louvain ----
    try:
        import leidenalg as la
        t = time.time(); g = knn_graph(Z, k=30)
        for res in (1.0, 2.0, 5.0, 10.0, 20.0, 40.0):
            part = la.find_partition(g, la.RBConfigurationVertexPartition,
                                     weights="weight", resolution_parameter=res, seed=0)
            pred = np.array(part.membership)
            rows.append(score(name, "leiden", f"res={res}", pred, len(set(pred))))
        lou = g.community_multilevel(weights="weight")
        pred = np.array(lou.membership)
        rows.append(score(name, "louvain", "-", pred, len(set(pred))))
        log(f"  leiden/louvain ({time.time()-t:.0f}s)")
    except Exception as ex:
        log(f"  graph methods failed: {ex}")
    # ---- Spectral (nearest-neighbour affinity) ----
    try:
        t = time.time()
        for k in (200,):
            pred = SpectralClustering(n_clusters=k, affinity="nearest_neighbors",
                                      n_neighbors=15, assign_labels="cluster_qr",
                                      random_state=0, n_jobs=8).fit_predict(Z)
            rows.append(score(name, "spectral", f"k={k}", pred, len(np.unique(pred))))
        log(f"  spectral ({time.time()-t:.0f}s)")
    except Exception as ex:
        log(f"  spectral failed: {ex}")
    # ---- DBSCAN (eps sweep on cosine geometry) ----
    for eps in (0.15, 0.25, 0.35):
        pred = DBSCAN(eps=eps, min_samples=5, metric="euclidean", n_jobs=8).fit_predict(Z)
        nk = len(set(pred)) - (1 if -1 in pred else 0)
        rows.append(score(name, "dbscan", f"eps={eps}", pred, nk))
    log("  dbscan done")
    # ---- HDBSCAN ----
    try:
        import hdbscan
        for mcs in (5, 15):
            pred = hdbscan.HDBSCAN(min_cluster_size=mcs, core_dist_n_jobs=8).fit_predict(Z)
            nk = len(set(pred)) - (1 if -1 in pred else 0)
            rows.append(score(name, "hdbscan", f"mcs={mcs}", pred, nk))
        log("  hdbscan done")
    except ImportError:
        pass
    # ---- Butina on raw binarized bits ----
    if name in ("drfp", "ecfp4_diff", "atompair_diff"):
        try:
            from rdkit import DataStructs
            from rdkit.ML.Cluster import Butina
            bits = (np.abs(Xraw) > 0).astype(np.uint8)
            bvs = [DataStructs.CreateFromBitString("".join(map(str, r))) for r in bits]
            dists = []
            for i in range(1, len(bvs)):
                dists.extend(1.0 - s for s in DataStructs.BulkTanimotoSimilarity(bvs[i], bvs[:i]))
            for cut in (0.4, 0.6):
                cs = Butina.ClusterData(dists, len(bvs), cut, isDistData=True, reordering=True)
                pred = np.empty(len(bvs), int)
                for ci, mem in enumerate(cs):
                    for m in mem: pred[m] = ci
                rows.append(score(name, "butina", f"cut={cut}", pred, len(cs)))
            log("  butina done")
        except Exception as ex:
            log(f"  butina failed: {ex}")
    return rows

ALL_EMB = ["ecfp4_diff", "ecfp2_diff", "ecfp6_diff", "ecfp4_concat", "ecfp4_drfp", "fcfp4_diff",
           "drfp", "atompair_diff", "rdk_diff", "torsion_diff", "avalon_diff",
           "maccs_diff", "secfp_diff", "descriptors_diff", "mordred_diff",
           "rxnfp", "rxnfp_pretrained", "chemberta_diff", "molformer_diff", "rxnfp_moldiff", "rxnfp_sidediff", "cgr_linear", "cgr_morgan",
           "erg_diff", "gin_diff", "unimol_diff", "biochem"]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out", default="results.csv")
    args = ap.parse_args()
    embs = args.only or ALL_EMB
    out = os.path.join(RESULTS, args.out); all_rows = []
    for e in embs:
        all_rows += run_embedding(e)
        pd.DataFrame(all_rows).to_csv(out, index=False)
    log(f"wrote {out} ({len(all_rows)} rows)")

if __name__ == "__main__":
    main()
