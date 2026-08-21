#!/usr/bin/env python
"""Benchmark: (embedding x clustering algorithm) -> EC-purity metrics.

Target metric = how cleanly clusters separate EC numbers. Reported at EC
granularity {full, ec3, ec2}:
  homogeneity  each cluster contains a single EC class  (the "clean split")
  completeness, v_measure
  ami          adjusted mutual information (chance/cardinality corrected ->
               fair single number to rank methods with different #clusters)
  ari
  purity       weighted mean over clusters of majority-EC fraction

Geometry & fairness: every embedding is L2-normalized, then reduced with
TruncatedSVD to a COMMON 256-dim space (cosine geometry preserved). This (a)
puts all methods on equal footing for the distance-based clusterers and (b)
makes agglomerative/kmeans tractable at n=13.7k. Butina additionally runs on
the raw binarized bits (native Tanimoto), independent of the SVD space.

Partition methods (KMeans, Agglomerative, Birch) run at fixed k so homogeneity
is comparable; hierarchical linkage is computed ONCE and cut at each k. Density
methods (HDBSCAN, Butina) discover their own k; homogeneity/purity score only
non-noise points, AMI/V-measure score all points (noise kept as its own label).
"""
import os, sys, time, warnings, argparse
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans, Birch
from sklearn.metrics import (homogeneity_completeness_v_measure,
                             adjusted_mutual_info_score, adjusted_rand_score)
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
DATA = os.path.join(HERE, "data"); RESULTS = os.path.join(HERE, "results"); CACHE = os.path.join(HERE, "cache")
EMB = os.path.join(HERE, "embeddings")
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, file=sys.stderr, flush=True)

df = pd.read_parquet(os.path.join(DATA, "dataset.parquet"))
LABELS = {lvl: df[lvl].to_numpy() for lvl in ("ec", "ec3", "ec2")}
N = len(df)
SVD_DIM = 256
K_GRID = [50, 100, 200, 400, 800, 1600]
K_AGG = [100, 400, 800, 1600]

def purity(pred, lt):
    mask = pred >= 0
    if mask.sum() == 0: return 0.0, 0.0
    lp, l = pred[mask], lt[mask]
    tot = sum(np.unique(l[lp == c], return_counts=True)[1].max() for c in np.unique(lp))
    return tot / mask.sum(), mask.mean()

def score(emb, algo, params, pred, n_clusters):
    row = dict(embedding=emb, algo=algo, params=params, n_clusters=int(n_clusters),
               noise_frac=float((pred < 0).mean()))
    for lvl in ("ec", "ec3", "ec2"):
        lt = LABELS[lvl]
        h, c, v = homogeneity_completeness_v_measure(lt, pred)
        row.update({f"{lvl}_homogeneity": round(h, 4), f"{lvl}_completeness": round(c, 4),
                    f"{lvl}_vmeasure": round(v, 4),
                    f"{lvl}_ami": round(adjusted_mutual_info_score(lt, pred), 4),
                    f"{lvl}_ari": round(adjusted_rand_score(lt, pred), 4),
                    f"{lvl}_purity": round(purity(pred, lt)[0], 4)})
    return row

def make_space(name, Xraw):
    X = StandardScaler().fit_transform(Xraw) if name == "rxnfp" else Xraw.astype(np.float32)
    X = normalize(X)
    d = min(SVD_DIM, X.shape[1] - 1)
    Z = TruncatedSVD(n_components=d, random_state=0).fit_transform(X)
    # zero-norm rows (e.g. reactions whose difference-FP is exactly 0) -> park on a
    # fixed basis vector so they normalize cleanly and cluster together, instead of
    # producing NaN cosine distances.
    norms = np.linalg.norm(Z, axis=1)
    zero = norms < 1e-9
    if zero.any():
        Z[zero] = 0.0; Z[zero, 0] = 1.0
        log(f"  {int(zero.sum())} zero-vector rows parked")
    return normalize(Z).astype(np.float32)  # cosine geometry in reduced space

def run_embedding(name):
    path = os.path.join(EMB, name + ".npy")
    if not os.path.exists(path):
        log(f"skip {name}: no npy"); return []
    Xraw = np.load(path)
    Z = make_space(name, Xraw)
    log(f"{name}: raw={Xraw.shape} -> Z={Z.shape}")
    rows = []
    # ---- MiniBatchKMeans grid ----
    for k in K_GRID:
        t = time.time()
        pred = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=3,
                               batch_size=4096, max_iter=100).fit_predict(Z)
        rows.append(score(name, "minibatch_kmeans", f"k={k}", pred, k))
        log(f"  kmeans k={k} ({time.time()-t:.0f}s)")
    # ---- Agglomerative: one linkage, many cuts ----
    for method, metric, tag in [("ward", "euclidean", "agglo_ward"),
                                ("average", "cosine", "agglo_avg_cosine")]:
        t = time.time()
        d = pdist(Z, metric=metric)
        Lk = linkage(d, method=method)
        for k in K_AGG:
            pred = fcluster(Lk, t=k, criterion="maxclust")
            rows.append(score(name, tag, f"k={k}", pred, len(np.unique(pred))))
        log(f"  {tag} (linkage+{len(K_AGG)} cuts, {time.time()-t:.0f}s)")
    # ---- Birch ----
    for k in (200, 800):
        t = time.time()
        pred = Birch(n_clusters=k, threshold=0.3).fit_predict(Z)
        rows.append(score(name, "birch", f"k={k}", pred, len(np.unique(pred))))
        log(f"  birch k={k} ({time.time()-t:.0f}s)")
    # ---- HDBSCAN ----
    try:
        import hdbscan
        for mcs in (5, 10, 20):
            t = time.time()
            pred = hdbscan.HDBSCAN(min_cluster_size=mcs, core_dist_n_jobs=8).fit_predict(Z)
            nk = len(set(pred)) - (1 if -1 in pred else 0)
            rows.append(score(name, "hdbscan", f"mcs={mcs}", pred, nk))
            log(f"  hdbscan mcs={mcs} -> {nk} ({time.time()-t:.0f}s)")
    except ImportError:
        log("  no hdbscan")
    # ---- Butina on raw binarized bits (native Tanimoto) ----
    if name in ("drfp", "ecfp4_diff", "atompair_diff"):
        try:
            from rdkit import DataStructs
            from rdkit.ML.Cluster import Butina
            bits = (np.abs(Xraw) > 0).astype(np.uint8)
            bvs = [DataStructs.CreateFromBitString("".join(map(str, r))) for r in bits]
            t0 = time.time(); dists = []
            for i in range(1, len(bvs)):
                dists.extend(1.0 - s for s in DataStructs.BulkTanimotoSimilarity(bvs[i], bvs[:i]))
            log(f"  butina distmat ({time.time()-t0:.0f}s)")
            for cut in (0.3, 0.5, 0.7):
                t = time.time()
                cs = Butina.ClusterData(dists, len(bvs), cut, isDistData=True, reordering=True)
                pred = np.empty(len(bvs), int)
                for ci, mem in enumerate(cs):
                    for m in mem: pred[m] = ci
                rows.append(score(name, "butina", f"cut={cut}", pred, len(cs)))
                log(f"  butina cut={cut} -> {len(cs)} ({time.time()-t:.0f}s)")
        except Exception as ex:
            log(f"  butina failed: {ex}")
    return rows

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    embs = args.only or ["ecfp4_diff", "ecfp4_concat", "ecfp4_drfp", "drfp",
                         "atompair_diff", "rdk_diff", "maccs_diff", "rxnfp"]
    out = os.path.join(RESULTS, "results.csv"); all_rows = []
    for e in embs:
        all_rows += run_embedding(e)
        pd.DataFrame(all_rows).to_csv(out, index=False)
    log(f"wrote {out} ({len(all_rows)} rows)")

if __name__ == "__main__":
    main()
