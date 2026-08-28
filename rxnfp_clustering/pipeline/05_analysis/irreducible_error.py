#!/usr/bin/env python
"""Irreducible-error (Bayes-floor) probe for EC prediction from reaction structure.

Any predictor is a function structure -> EC. Where two reactions have the SAME
structure but DIFFERENT EC, no function can be right on both: that error is
irreducible under the representation, and it lower-bounds the error of every
embedding x algorithm. This script measures that floor two ways:

  (A) EXACT probe (representation-free): group reactions by identical rxn_smiles.
      Any group whose members disagree on EC is a hard collision -- literally the
      same reaction with different EC labels (one transformation catalysed by
      different enzyme classes, or a DB inconsistency). Reported per EC level.

  (B) NEAR probe (per embedding): in each embedding's kNN geometry (L2 -> SVD-256
      -> cosine, identical to knn_annotation.py), take every reaction's 1-NN
      and record (cosine similarity, EC-match?). Binning P(EC conflict | similarity)
      vs similarity shows the floor: as similarity -> 1, the residual conflict is
      the structure-identical Bayes error. Headline number = conflict rate among
      pairs with cosine >= 0.99 (near-identical structure), and the implied 1-NN
      accuracy ceiling (1 - overall 1-NN disagreement).

Outputs: results/irreducible_error.csv (summary), figures/fig_irreducible_error.png.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "_lib"))
import _style
FS = _style.FS

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(HERE, "data"); RESULTS = os.path.join(HERE, "results")
EMB = os.path.join(HERE, "embeddings"); FIG = os.path.join(HERE, "figures", "benchmark")
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, file=sys.stderr, flush=True)

df = pd.read_parquet(os.path.join(DATA, "dataset.parquet"))
N = len(df)
LEVELS = ["ec1", "ec2", "ec3", "ec"]           # ec = full 4-level
LEVEL_LABEL = {"ec1": "EC1", "ec2": "EC2", "ec3": "EC3", "ec": "full EC"}
CONTINUOUS = {"descriptors_diff", "mordred_diff", "chemberta_diff"}

# ---------------------------------------------------------------- (A) EXACT
log("(A) exact rxn_smiles collisions")
rows = []
g = df.groupby("rxn_smiles")
dup_mask = df.rxn_smiles.map(g.size()) >= 2       # reactions sharing a SMILES
n_dup = int(dup_mask.sum())
exact = {}
for lvl in LEVELS:
    # a reaction is "conflicted" if some other reaction shares its exact SMILES
    # but carries a different label at this level
    nun = g[lvl].transform("nunique")
    conflicted = int(((df.rxn_smiles.map(g.size()) >= 2) & (nun >= 2)).sum())
    exact[lvl] = conflicted / N
    rows.append(dict(probe="exact_smiles", level=lvl, metric="conflict_frac",
                     value=round(conflicted / N, 4), n_conflicted=conflicted,
                     n_with_twin=n_dup, floor_acc=round(1 - conflicted / N, 4)))
    log(f"  {LEVEL_LABEL[lvl]}: {conflicted}/{N} reactions have an exact structural "
        f"twin with a different label -> floor acc <= {1-conflicted/N:.4f}")

# ---------------------------------------------------------------- (B) NEAR
def make_space(name, Xraw):
    X = StandardScaler().fit_transform(Xraw) if name in CONTINUOUS else Xraw.astype(np.float32)
    X[~np.isfinite(X)] = 0.0
    X = normalize(X)
    d = min(256, X.shape[1] - 1)
    Z = TruncatedSVD(n_components=d, random_state=0).fit_transform(X)
    z = np.linalg.norm(Z, axis=1) < 1e-9
    if z.any(): Z[z] = 0.0; Z[z, 0] = 1.0
    return normalize(Z).astype(np.float32)

PROBE_EMB = ["ecfp4_diff", "descriptors_diff"]     # best structural + physchem
curves = {}
for name in PROBE_EMB:
    p = os.path.join(EMB, name + ".npy")
    if not os.path.exists(p):
        log(f"  skip {name}: no npy"); continue
    Z = make_space(name, np.load(p))
    nn = NearestNeighbors(n_neighbors=2, metric="cosine", n_jobs=6).fit(Z)
    dist, idx = nn.kneighbors(Z)
    sim = 1.0 - dist[:, 1]                          # cosine sim to 1-NN
    nbr = idx[:, 1]
    curves[name] = {"sim": sim}
    for lvl in LEVELS:
        lab = df[lvl].to_numpy()
        match = lab[nbr] == lab                     # 1-NN label agreement
        curves[name][lvl] = match
        overall_conflict = 1 - match.mean()         # 1-NN disagreement = acc ceiling gap
        hi = sim >= 0.99
        near_conflict = 1 - match[hi].mean() if hi.any() else np.nan
        rows.append(dict(probe=f"near_{name}", level=lvl, metric="1nn_conflict_frac",
                         value=round(overall_conflict, 4),
                         near_conflict_at_sim99=round(float(near_conflict), 4),
                         n_at_sim99=int(hi.sum()),
                         floor_acc=round(match.mean(), 4)))
        log(f"  {name} {LEVEL_LABEL[lvl]}: 1-NN acc ceiling {match.mean():.4f} | "
            f"conflict@cos>=.99 {near_conflict:.4f} (n={int(hi.sum())})")

pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "irreducible_error.csv"), index=False)
log("wrote results/irreducible_error.csv")

# ---------------------------------------------------------------- FIGURE
# P(EC conflict | 1-NN cosine similarity) vs similarity, for ecfp4_diff.
name = "ecfp4_diff"
if name in curves:
    sim = curves[name]["sim"]
    bins = np.linspace(sim.min(), 1.0, 13)
    ctr = 0.5 * (bins[:-1] + bins[1:])
    fig, ax = plt.subplots(figsize=(11, 7.5))
    colors = {"ec1": "#0072B2", "ec2": "#009E73", "ec3": "#D55E00"}
    for lvl in ["ec1", "ec2", "ec3"]:
        conflict = 1 - curves[name][lvl].astype(float)
        rate, keepx = [], []
        for lo, hi, c in zip(bins[:-1], bins[1:], ctr):
            m = (sim >= lo) & (sim < hi) if hi < 1.0 else (sim >= lo) & (sim <= hi)
            if m.sum() >= 20:
                rate.append(conflict[m].mean()); keepx.append(c)
        ax.plot(keepx, rate, marker="o", ms=8, lw=2.6, color=colors[lvl],
                label=LEVEL_LABEL[lvl])
    ax.set_xlabel("1-NN cosine similarity (structural identity)", fontsize=FS)
    ax.set_ylabel("P(EC label conflict with 1-NN)", fontsize=FS)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.grid(alpha=0.25); ax.set_axisbelow(True)
    ax.set_title("EC conflict vanishes as structure matches — the ceiling is sparsity",
                 fontsize=FS)
    ax.legend(fontsize=FS, frameon=False, title="EC level", title_fontsize=FS)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_irreducible_error.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    log("wrote figures/fig_irreducible_error.png")
