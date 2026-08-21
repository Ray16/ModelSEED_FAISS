#!/usr/bin/env python
"""Does the slide-11 fusion model give good EC ANNOTATION (not just clustering)?

Builds the multi-view fusion the same way the benchmark does (per-view standardize
continuous / L2-normalize -> concat -> SVD-256 -> L2) and runs the good annotation
rules (weighted vote, threshold+abstain, hierarchical back-off) under a
reference/test split, EC3. Compares each fusion against the best single fingerprint
(ecfp4_diff) evaluated on the SAME rows, so it's apples-to-apples.

Two configs:
  A) FULL set fusion, NO CGR (full coverage):  descriptors + chemberta + ecfp4
  B) CGR-subset fusion (slide-11 style):       descriptors + cgr_morgan + chemberta
     -> only the ~75% of reactions CGR maps; that is the annotation coverage cap.

Env: palm. Reuses helpers from annotate.py.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data"); EMB = os.path.join(HERE, "embeddings")
ADIR = os.path.join(HERE, "annotation")
ARES = os.path.join(ADIR, "results"); AFIG = os.path.join(ADIR, "figures")
sys.path.insert(0, os.path.join(HERE, "scripts")); import _style; FS = _style.FS
import matplotlib.pyplot as plt
sys.path.insert(0, ADIR)
from annotate import weighted_vote, acc_coverage_curve, acc_at_coverage, cov_at_accuracy

def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, file=sys.stderr, flush=True)

CONTINUOUS = {"descriptors_diff", "chemberta_diff", "rxnfp", "molformer_diff"}
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

def annotate_eval(tag, Z, rows, K=15, seed=0, test_frac=0.2):
    """Run M1/M2/M3/M4 at EC3 on rows; return dict + the M2 acc-cov curve."""
    lab3 = df["ec3"].to_numpy()[rows]
    LAB = {lvl: df[lvl].to_numpy()[rows] for lvl in ("ec3", "ec2", "ec1")}
    n = len(rows)
    rng = np.random.default_rng(seed); perm = rng.permutation(n)
    nt = int(test_frac * n); te = np.sort(perm[:nt]); rf = np.sort(perm[nt:])
    nn = NearestNeighbors(n_neighbors=50, metric="cosine", n_jobs=8).fit(Z[rf])
    dist, nbr = nn.kneighbors(Z[te]); sim = np.clip(1 - dist, 0, None)
    LN = {lvl: LAB[lvl][rf][nbr] for lvl in LAB}
    yt = {lvl: LAB[lvl][te] for lvl in LAB}
    ans = np.ones(nt, bool)
    # M1 equal-weight, M2 weighted
    p1 = np.array([weighted_vote(LN["ec3"][i, :K], np.ones(K))[0] for i in range(nt)])
    c2 = np.array([weighted_vote(LN["ec3"][i, :K], sim[i, :K]) for i in range(nt)], dtype=object)
    p2 = np.array([r[0] for r in c2]); f2 = np.array([r[1] for r in c2], float)
    corr1 = p1 == yt["ec3"]; corr2 = p2 == yt["ec3"]
    cov2, acc2 = acc_coverage_curve(f2, corr2, ans)
    # M3 threshold sweep -> best tau with coverage>=0.75
    best = None
    for tau in np.round(np.arange(0.3, 0.96, 0.05), 2):
        answ = np.zeros(nt, bool); corr = np.zeros(nt, bool)
        for i in range(nt):
            keep = sim[i] >= tau
            if keep.any():
                b, _, _ = weighted_vote(LN["ec3"][i][keep], sim[i][keep])
                answ[i] = True; corr[i] = (b == yt["ec3"][i])
        cov = answ.mean(); acc = corr[answ].mean() if answ.any() else 0.0
        if cov >= 0.75: best = (tau, cov, acc)
    # M4 back-off
    THETA = 0.5; correct_bo = np.zeros(nt, bool); answered = np.zeros(nt, bool)
    for i in range(nt):
        for lvl in ("ec3", "ec2", "ec1"):
            b, f, _ = weighted_vote(LN[lvl][i, :K], sim[i, :K])
            if f >= THETA:
                correct_bo[i] = (b == yt[lvl][i]); answered[i] = True; break
    return dict(tag=tag, n=n,
                m1_acc=round(corr1.mean(), 4),
                m2_acc=round(corr2.mean(), 4),
                m2_acc_at75=round(acc_at_coverage(f2, corr2, ans, 0.75)[0], 4),
                m3_tau=best[0] if best else None,
                m3_cov=round(best[1], 4) if best else None,
                m3_acc=round(best[2], 4) if best else None,
                m4_acc_reported=round(correct_bo.mean(), 4),
                m4_cov=round(answered.mean(), 4)), (cov2, acc2)

CONFIGS = [
    ("A: fusion PhysChem+ChemBERTa+ECFP (full set)", ["descriptors_diff", "chemberta_diff", "ecfp4_diff"], None),
    ("A-ref: ecfp4_diff (full set)",                 ["ecfp4_diff"], None),
    ("B: fusion PhysChem+CGR+ChemBERTa (CGR subset)", ["descriptors_diff", "cgr_morgan", "chemberta_diff"], "cgr"),
    ("B-ref: ecfp4_diff (CGR subset)",               ["ecfp4_diff"], "cgr"),
]
mask = np.load(os.path.join(DATA, "cgr_mask.npy"))
rows_full = np.arange(len(df)); rows_cgr = np.where(mask)[0]

rows_out = []; curves = {}
for tag, views, sub in CONFIGS:
    rows = rows_cgr if sub == "cgr" else rows_full
    Z = space(views, rows)
    row, curve = annotate_eval(tag, Z, rows)
    rows_out.append(row); curves[tag] = curve
    log(f"{tag}: M2={row['m2_acc']}  M3={row['m3_acc']}@{row['m3_cov']}  M4={row['m4_acc_reported']}")

res = pd.DataFrame(rows_out)
res.to_csv(os.path.join(ARES, "fusion_annotation.csv"), index=False)
print(res.to_string(index=False))

# figure: accuracy-vs-coverage, fusion vs ref, for both set + subset
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
for ax, (title, keys) in zip(axes, [
        ("Full set", [k for k in curves if k.startswith("A")]),
        ("CGR subset (slide-11 model)", [k for k in curves if k.startswith("B")])]):
    for k in keys:
        cov, acc = curves[k]
        lab = "fusion" if "fusion" in k else "ecfp4_diff (single)"
        ax.plot(cov, acc, lw=2.8, label=lab)
    ax.set_title(title, fontsize=FS); ax.set_xlabel("coverage", fontsize=FS)
    ax.set_ylabel("EC3 accuracy (answered)", fontsize=FS); ax.grid(alpha=0.3)
    ax.tick_params(labelsize=FS); ax.legend(fontsize=FS, frameon=False, loc="lower left")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.suptitle("Does the fusion model annotate better than a single fingerprint? (EC3)", fontsize=FS)
fig.tight_layout()
fig.savefig(os.path.join(AFIG, "fig_fusion_annotation.png"), dpi=300, bbox_inches="tight")
print("wrote fig_fusion_annotation.png")
