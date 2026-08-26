#!/usr/bin/env python
"""How much does taking only the FIRST EC understate retrieval accuracy?

6.5% of reactions carry >1 full EC (e.g. rxn00011 = 1.2.4.1;2.2.1.6;4.1.1.1). The
main benchmark scores against the first EC only, so a neighbour-vote that returns a
DIFFERENT-but-also-correct EC is counted wrong. Here we rescore leave-one-out
retrieval two ways for the top embeddings:

  first-EC   prediction == the reaction's first EC     (what the benchmark reports)
  any-match  prediction in the reaction's FULL EC set  (fair to multi-label reactions)

The gap is the label-noise the first-EC convention injects. Uses the `ec_all` column
added by build_dataset.py.

Outputs: results/multilabel_eval.csv, figures/fig_multilabel.png
"""
import os, sys, time
import numpy as np, pandas as pd
from collections import Counter
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "_lib"))
import _style; FS = _style.FS

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(HERE, "data"); RESULTS = os.path.join(HERE, "results")
EMB = os.path.join(HERE, "embeddings"); FIG = os.path.join(HERE, "figures")
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, file=sys.stderr, flush=True)

EMBEDDINGS = ["ecfp4_drfp", "ecfp6_diff", "ecfp4_diff", "descriptors_diff"]
CONTINUOUS = {"rxnfp", "rxnfp_pretrained", "chemberta_diff", "descriptors_diff",
              "molformer_diff", "rxnfp_moldiff", "rxnfp_sidediff",
              "gin_diff", "unimol_diff", "biochem", "mordred_diff"}

def make_space(name, svd_dim=256):
    """Benchmark geometry: StandardScaler (continuous) -> L2 -> SVD256 -> L2."""
    Xraw = np.load(os.path.join(EMB, name + ".npy")).astype(np.float32)
    X = StandardScaler().fit_transform(Xraw) if name in CONTINUOUS else Xraw
    X[~np.isfinite(X)] = 0.0
    X = normalize(X)
    d = min(svd_dim, X.shape[1] - 1)
    Z = TruncatedSVD(n_components=d, random_state=0).fit_transform(X)
    z = np.linalg.norm(Z, axis=1) < 1e-9
    if z.any(): Z[z] = 0.0; Z[z, 0] = 1.0
    return normalize(Z).astype(np.float32)

df = pd.read_parquet(os.path.join(DATA, "dataset.parquet"))
N = len(df)
ec3_single = df["ec3"].to_numpy()
ec_single = df["ec"].to_numpy()
# per-reaction FULL sets at each level, from ec_all
def prefix_set(ec_all, level):
    out = set()
    for full in str(ec_all).split(";"):
        p = full.split(".")
        if len(p) == 4:
            out.add(".".join(p[:level]) if level < 4 else full)
    return out
ec3_sets = [prefix_set(e, 3) for e in df["ec_all"]]
ecf_sets = [prefix_set(e, 4) for e in df["ec_all"]]
log(f"{N} reactions; multi-EC {(df.n_ec > 1).sum()} ({(df.n_ec > 1).mean()*100:.1f}%)")

rows = []
for name in EMBEDDINGS:
    p = os.path.join(EMB, name + ".npy")
    if not os.path.exists(p):
        log(f"  skip {name}"); continue
    Z = make_space(name)
    nn = NearestNeighbors(n_neighbors=16, metric="cosine", algorithm="brute").fit(Z)
    _, idx = nn.kneighbors(Z); idx = idx[:, 1:]        # drop self
    for k in (1, 15):
        nb = idx[:, :k]
        # majority-vote predictions (neighbour labels are single/first-EC)
        pred3 = np.array([Counter(ec3_single[r]).most_common(1)[0][0] for r in nb])
        predf = np.array([Counter(ec_single[r]).most_common(1)[0][0] for r in nb])
        ec3_first = np.mean(pred3 == ec3_single)
        ec3_any = np.mean([pred3[i] in ec3_sets[i] for i in range(N)])
        ecf_first = np.mean(predf == ec_single)
        ecf_any = np.mean([predf[i] in ecf_sets[i] for i in range(N)])
        rows.append(dict(embedding=name, k=k,
                         ec3_first=round(ec3_first, 4), ec3_any=round(ec3_any, 4),
                         ec3_gain=round(ec3_any - ec3_first, 4),
                         fullec_first=round(ecf_first, 4), fullec_any=round(ecf_any, 4),
                         fullec_gain=round(ecf_any - ecf_first, 4)))
        log(f"  {name:16s} {k:2d}-NN  EC3 {ec3_first:.3f}->{ec3_any:.3f} (+{ec3_any-ec3_first:.3f})  "
            f"fullEC {ecf_first:.3f}->{ecf_any:.3f} (+{ecf_any-ecf_first:.3f})")

out = pd.DataFrame(rows)
out.to_csv(os.path.join(RESULTS, "multilabel_eval.csv"), index=False)
log("wrote results/multilabel_eval.csv")

# ---- figure: first-EC vs any-match, 15-NN ----
d15 = out[out.k == 15]
fig, ax = plt.subplots(1, 2, figsize=(16, 7))
for a, (fcol, acol, ttl) in zip(ax, [("ec3_first", "ec3_any", "EC3"),
                                     ("fullec_first", "fullec_any", "full EC")]):
    x = np.arange(len(d15)); w = 0.36
    a.bar(x - w/2, d15[fcol], w, color="#8593A3", label="first-EC (benchmark)")
    a.bar(x + w/2, d15[acol], w, color="#0072B2", label="any-match (fair)")
    for xi in range(len(d15)):
        a.text(x[xi]-w/2, d15[fcol].iloc[xi]+0.01, f"{d15[fcol].iloc[xi]:.2f}", ha="center", va="bottom", fontsize=FS)
        a.text(x[xi]+w/2, d15[acol].iloc[xi]+0.01, f"{d15[acol].iloc[xi]:.2f}", ha="center", va="bottom", fontsize=FS)
    a.set_xticks(x); a.set_xticklabels(d15.embedding, rotation=20, ha="right", fontsize=FS)
    a.set_ylim(0, 1.0); a.set_title(f"{ttl} 15-NN accuracy", fontsize=FS)
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
    a.grid(axis="y", alpha=0.25); a.set_axisbelow(True)
ax[0].set_ylabel("retrieval accuracy", fontsize=FS)
ax[0].legend(fontsize=FS, frameon=False, loc="lower left")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_multilabel.png"), dpi=300, bbox_inches="tight")
log("wrote figures/fig_multilabel.png")
