#!/usr/bin/env python
"""Confidence intervals for the headline numbers -- which gaps are real?

The clustering (agglomerative) and kNN (leave-one-out) results are DETERMINISTIC
given the dataset, so re-seeding produces no variance; the honest uncertainty is
SAMPLING variance. We quantify it two ways:

  (A) Retrieval CIs -- exact bootstrap of the per-reaction LOO correctness
      indicators (B=1000). Puts a 95% CI on 1-NN / 15-NN / confidence-gated EC3
      accuracy for the top embeddings. This is the honest error bar on 0.87/0.90.

  (B) Clustering AMI CIs -- agglo-average is deterministic, so we subsample the
      reaction set (fixed n, B=30), rerun agglo-average-cosine at k=#EC3, and
      report EC3 AMI mean +/- 95% CI. Tells us whether the 0.61-0.63 "plateau"
      gaps between structural fingerprints are real or within noise.

Outputs: results/variance.csv, figures/fig_variance.png
"""
import os, sys, time
import numpy as np, pandas as pd
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_mutual_info_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "_lib"))
import _style; FS = _style.FS

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(HERE, "data"); RESULTS = os.path.join(HERE, "results")
EMB = os.path.join(HERE, "embeddings"); FIG = os.path.join(HERE, "figures", "benchmark")
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, file=sys.stderr, flush=True)

SEED = 0; rng = np.random.default_rng(SEED)
RETR_EMB = ["ecfp4_drfp", "ecfp6_diff", "ecfp4_diff", "secfp_diff",
            "descriptors_diff", "atompair_diff", "chemberta_diff", "rxnfp"]
CLUST_EMB = ["ecfp4_drfp", "secfp_diff", "ecfp6_diff", "descriptors_diff",
             "ecfp4_diff", "ecfp2_diff"]
B_RETR = 1000
B_CLUST = 30; N_SUB = 6000
GATE = 0.6

df = pd.read_parquet(os.path.join(DATA, "dataset.parquet"))
ec3 = df["ec3"].to_numpy(); N = len(df)
log(f"{N} reactions")

# same geometry as the benchmark: StandardScaler (continuous) -> L2 -> SVD256 -> L2
CONTINUOUS = {"rxnfp", "rxnfp_pretrained", "chemberta_diff", "descriptors_diff",
              "molformer_diff", "rxnfp_moldiff", "rxnfp_sidediff",
              "gin_diff", "unimol_diff", "biochem", "mordred_diff"}

def load(name, svd_dim=256):
    Xraw = np.load(os.path.join(EMB, name + ".npy")).astype(np.float32)
    X = StandardScaler().fit_transform(Xraw) if name in CONTINUOUS else Xraw
    X[~np.isfinite(X)] = 0.0
    X = normalize(X)
    d = min(svd_dim, X.shape[1] - 1)
    Z = TruncatedSVD(n_components=d, random_state=0).fit_transform(X)
    z = np.linalg.norm(Z, axis=1) < 1e-9
    if z.any(): Z[z] = 0.0; Z[z, 0] = 1.0
    return normalize(Z).astype(np.float32)

def ci95(x):
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))

# ============================================================ (A) retrieval CIs
from collections import Counter
rows_retr = []
for name in RETR_EMB:
    if not os.path.exists(os.path.join(EMB, name + ".npy")):
        log(f"  skip {name} (missing)"); continue
    Z = load(name)
    nn = NearestNeighbors(n_neighbors=16, metric="cosine", algorithm="brute").fit(Z)
    _, idx = nn.kneighbors(Z)
    idx = idx[:, 1:]                                   # drop self
    lab1 = ec3[idx[:, 0]]                              # 1-NN prediction
    # 15-NN vote + agreement
    lab15, agree = [], []
    for row in idx[:, :15]:
        c = Counter(ec3[row]); top, m = c.most_common(1)[0]
        lab15.append(top); agree.append(m / 15)
    lab15 = np.array(lab15); agree = np.array(agree)
    corr1 = (lab1 == ec3).astype(float)               # per-reaction 0/1
    corr15 = (lab15 == ec3).astype(float)
    conf = agree >= GATE
    # bootstrap the indicators
    b1, b15, bcov, bgate = [], [], [], []
    for _ in range(B_RETR):
        s = rng.integers(0, N, N)
        b1.append(corr1[s].mean()); b15.append(corr15[s].mean())
        cs = conf[s]; bcov.append(cs.mean())
        bgate.append(corr15[s][cs].mean() if cs.any() else np.nan)
    b1, b15, bcov, bgate = map(np.array, (b1, b15, bcov, bgate))
    lo1, hi1 = ci95(b1); lo15, hi15 = ci95(b15)
    lc, hc = ci95(bcov); lg, hg = ci95(bgate[~np.isnan(bgate)])
    rows_retr.append(dict(embedding=name,
        ec3_1nn=round(corr1.mean(),4), ec3_1nn_lo=round(lo1,4), ec3_1nn_hi=round(hi1,4),
        ec3_15nn=round(corr15.mean(),4), ec3_15nn_lo=round(lo15,4), ec3_15nn_hi=round(hi15,4),
        gated_acc=round(np.nanmean(corr15[conf]),4), gated_acc_lo=round(lg,4), gated_acc_hi=round(hg,4),
        coverage=round(conf.mean(),4), coverage_lo=round(lc,4), coverage_hi=round(hc,4)))
    log(f"  {name:16s} 1NN {corr1.mean():.3f} [{lo1:.3f},{hi1:.3f}]  "
        f"15NN {corr15.mean():.3f} [{lo15:.3f},{hi15:.3f}]  "
        f"gated {np.nanmean(corr15[conf]):.3f} [{lg:.3f},{hg:.3f}] @ {conf.mean():.2f}")
pd.DataFrame(rows_retr).to_csv(os.path.join(RESULTS, "variance_retrieval.csv"), index=False)
log("wrote results/variance_retrieval.csv")

# ============================================================ (B) clustering AMI CIs
log(f"clustering AMI subsample CIs (n={N_SUB}, B={B_CLUST}) ...")
rows_clu = []
for name in CLUST_EMB:
    if not os.path.exists(os.path.join(EMB, name + ".npy")):
        log(f"  skip {name}"); continue
    Z = load(name)
    amis = []
    for b in range(B_CLUST):
        sub = rng.choice(N, size=N_SUB, replace=False)
        y = ec3[sub]; k = len(np.unique(y))
        pred = AgglomerativeClustering(n_clusters=k, metric="cosine",
                                       linkage="average").fit_predict(Z[sub])
        amis.append(adjusted_mutual_info_score(y, pred))
    amis = np.array(amis); lo, hi = ci95(amis)
    rows_clu.append(dict(embedding=name, ec3_ami_mean=round(amis.mean(),4),
                         ec3_ami_std=round(amis.std(),4),
                         ec3_ami_lo=round(lo,4), ec3_ami_hi=round(hi,4)))
    log(f"  {name:16s} AMI {amis.mean():.4f} +/- {amis.std():.4f}  [{lo:.4f},{hi:.4f}]")
pd.DataFrame(rows_clu).to_csv(os.path.join(RESULTS, "variance_clustering.csv"), index=False)
log("wrote results/variance_clustering.csv")

# ============================================================ FIGURE
rr = pd.DataFrame(rows_retr); cc = pd.DataFrame(rows_clu)
fig, ax = plt.subplots(1, 2, figsize=(20, 8))
# (a) retrieval gated EC3 accuracy with CI
rr2 = rr.sort_values("gated_acc")
y = np.arange(len(rr2))
ax[0].errorbar(rr2.gated_acc, y,
               xerr=[rr2.gated_acc - rr2.gated_acc_lo, rr2.gated_acc_hi - rr2.gated_acc],
               fmt="o", color="#0072B2", ms=11, capsize=6, lw=2, label="gated EC3 acc")
ax[0].errorbar(rr2.ec3_15nn, y + 0.0,
               xerr=[rr2.ec3_15nn - rr2.ec3_15nn_lo, rr2.ec3_15nn_hi - rr2.ec3_15nn],
               fmt="s", color="#E69F00", ms=10, capsize=6, lw=2, label="15-NN EC3 acc")
ax[0].set_yticks(y); ax[0].set_yticklabels(rr2.embedding, fontsize=FS)
ax[0].set_xlabel("EC3 retrieval accuracy (95% CI)", fontsize=FS)
ax[0].set_title("(a) retrieval accuracy ± bootstrap CI", fontsize=FS)
ax[0].legend(fontsize=FS, frameon=False, loc="lower right")
# (b) clustering AMI with CI
cc2 = cc.sort_values("ec3_ami_mean")
yc = np.arange(len(cc2))
ax[1].errorbar(cc2.ec3_ami_mean, yc,
               xerr=[cc2.ec3_ami_mean - cc2.ec3_ami_lo, cc2.ec3_ami_hi - cc2.ec3_ami_mean],
               fmt="o", color="#009E73", ms=11, capsize=6, lw=2)
ax[1].set_yticks(yc); ax[1].set_yticklabels(cc2.embedding, fontsize=FS)
ax[1].set_xlabel(f"EC3 AMI (subsample n={N_SUB}, 95% CI)", fontsize=FS)
ax[1].set_title("(b) clustering AMI ± subsample CI", fontsize=FS)
for a in ax:
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
    a.grid(axis="x", alpha=0.25); a.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_variance.png"), dpi=300, bbox_inches="tight")
log("wrote figures/fig_variance.png")
