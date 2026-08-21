#!/usr/bin/env python
"""Is the fusion's annotation gain REAL or noise? Paired significance test.

Compares fusion vs ecfp4_diff on the SAME held-out queries (paired), EC3,
weighted-vote (M2, full coverage). Reports:
  - error-rate reduction (the honest reading of an accuracy delta)
  - McNemar exact test on discordant pairs (is the paired difference significant?)
  - bootstrap 95% CI on the accuracy difference
Env: palm.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data"); EMB = os.path.join(HERE, "embeddings")
sys.path.insert(0, os.path.join(HERE, "annotation"))
from annotate import weighted_vote
CONTINUOUS = {"descriptors_diff", "chemberta_diff"}
df = pd.read_parquet(os.path.join(DATA, "dataset.parquet"))

def view(name, rows):
    X = np.load(os.path.join(EMB, name + ".npy"))[rows]
    X = StandardScaler().fit_transform(X) if name in CONTINUOUS else X.astype(np.float32)
    X[~np.isfinite(X)] = 0.0
    return normalize(X).astype(np.float32)

def space(views, rows):
    X = np.hstack([view(v, rows) for v in views])
    Z = TruncatedSVD(n_components=min(256, X.shape[1]-1), random_state=0).fit_transform(X)
    z = np.linalg.norm(Z, axis=1) < 1e-9
    if z.any(): Z[z] = 0.0; Z[z, 0] = 1.0
    return normalize(Z).astype(np.float32)

def correct_vec(Z, rows, te, rf, K=15):
    lab = df["ec3"].to_numpy()[rows]
    nn = NearestNeighbors(n_neighbors=K, metric="cosine", n_jobs=8).fit(Z[rf])
    dist, nbr = nn.kneighbors(Z[te]); sim = np.clip(1 - dist, 0, None)
    LN = lab[rf][nbr]; yt = lab[te]
    pred = np.array([weighted_vote(LN[i], sim[i])[0] for i in range(len(te))])
    return pred == yt

def analyze(name, views_f, rows, seed=0, test_frac=0.2, B=2000):
    n = len(rows); rng = np.random.default_rng(seed); perm = rng.permutation(n)
    nt = int(test_frac * n); te = np.sort(perm[:nt]); rf = np.sort(perm[nt:])
    cf = correct_vec(space(views_f, rows), rows, te, rf)
    cs = correct_vec(space(["ecfp4_diff"], rows), rows, te, rf)
    accf, accs = cf.mean(), cs.mean()
    errf, errs = 1 - accf, 1 - accs
    b = int((cf & ~cs).sum())      # fusion right, single wrong
    c = int((~cf & cs).sum())      # fusion wrong, single right
    p = binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) > 0 else 1.0
    # bootstrap CI on paired accuracy difference
    diff = cf.astype(int) - cs.astype(int)
    boot = np.array([diff[rng.integers(0, nt, nt)].mean() for _ in range(B)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(config=name, n_test=nt,
                acc_fusion=round(accf, 4), acc_single=round(accs, 4),
                delta_acc=round(accf - accs, 4),
                err_reduction_rel=round((errs - errf) / errs, 4) if errs > 0 else 0,
                discordant_b_fusiononly=b, discordant_c_singleonly=c,
                mcnemar_p=float(f"{p:.2e}"),
                boot95_lo=round(lo, 4), boot95_hi=round(hi, 4),
                significant=(lo > 0 and p < 0.05))

mask = np.load(os.path.join(DATA, "cgr_mask.npy"))
rows_full = np.arange(len(df)); rows_cgr = np.where(mask)[0]
out = [
    analyze("full set: PhysChem+ChemBERTa+ECFP", ["descriptors_diff","chemberta_diff","ecfp4_diff"], rows_full),
    analyze("CGR subset: PhysChem+CGR+ChemBERTa", ["descriptors_diff","cgr_morgan","chemberta_diff"], rows_cgr),
]
res = pd.DataFrame(out)
res.to_csv(os.path.join(HERE, "annotation", "results", "fusion_significance.csv"), index=False)
print(res.to_string(index=False))
