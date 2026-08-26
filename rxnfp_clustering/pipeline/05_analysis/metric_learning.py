#!/usr/bin/env python
"""Supervised hierarchical-EC metric learning -- can it break the 0.63/0.87 plateau?

The irreducible-error probe (26) showed the plateau is SPARSITY, not an information
ceiling: structure determines EC3 (96-99.6%) wherever a close neighbour exists, but
~35% of reactions sit in sparse regions with no close same-EC neighbour. A learned
metric that pulls same-EC reactions together should densify those neighbourhoods and
lift EC3 -- IF the headroom is real.

Method: take the baseline representation (ecfp4_diff -> L2 -> TruncatedSVD-256, fit on
TRAIN only), train a small MLP projection with a supervised-contrastive (SupCon) loss
on EC3 identity (a PK sampler guarantees in-batch positives). Compare the learned
128-d cosine space against the raw SVD-256 baseline, on:
  * retrieval  -- 1-NN and 15-NN EC1/EC2/EC3 accuracy (query -> reference vote)
  * clustering -- agglomerative-average-cosine EC3 AMI

Two splits, both leakage-safe (SVD + projection fit on TRAIN only):
  (R) HELD-OUT REACTIONS  -- 80/20 stratified by EC3; reference = train. The
      deployment number: does the metric improve retrieval/clustering on unseen
      reactions of seen classes?
  (C) HELD-OUT CLASSES    -- ~30 EC3 classes held out entirely; eval retrieval &
      clustering AMONG the held-out classes (leave-one-out). Tests whether the metric
      learned a transferable notion of EC-similarity vs. memorised the training classes.

Outputs: results/metric_learning.csv, figures/fig_metric_learning.png.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_mutual_info_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "_lib"))
import _style
FS = _style.FS

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(HERE, "data"); RESULTS = os.path.join(HERE, "results")
EMB = os.path.join(HERE, "embeddings"); FIG = os.path.join(HERE, "figures")
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, file=sys.stderr, flush=True)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
rng = np.random.default_rng(SEED); torch.manual_seed(SEED)

df = pd.read_parquet(os.path.join(DATA, "dataset.parquet"))
Xraw = np.load(os.path.join(EMB, "ecfp4_diff.npy")).astype(np.float32)
EC = {lvl: df[lvl].to_numpy() for lvl in ("ec1", "ec2", "ec3")}
N = len(df)

# ---------------------------------------------------------------- helpers
def svd_space(Xtrain, Xall, d=256):
    """L2 -> SVD (fit on train) -> L2. Returns the baseline geometry for Xall."""
    sv = TruncatedSVD(n_components=d, random_state=0)
    sv.fit(normalize(Xtrain))
    Z = sv.transform(normalize(Xall))
    z = np.linalg.norm(Z, axis=1) < 1e-9
    if z.any(): Z[z] = 0.0; Z[z, 0] = 1.0
    return normalize(Z).astype(np.float32)

def retrieval(Zref, ref_gid, Zq, q_gid, ks=(1, 15)):
    """majority-vote EC accuracy of queries against a reference set.
    ref_gid / q_gid map the rows of Zref / Zq back to GLOBAL dataset indices, so
    kNN's reference-local neighbour ids are translated before reading EC labels."""
    from collections import Counter
    ref_gid = np.asarray(ref_gid); q_gid = np.asarray(q_gid)
    loo = Zref is Zq
    kmax = max(ks) + (1 if loo else 0)
    nn = NearestNeighbors(n_neighbors=kmax, metric="cosine").fit(Zref)
    _, idx = nn.kneighbors(Zq)
    if loo: idx = idx[:, 1:]                  # drop self for in-set LOO
    out = {}
    for k in ks:
        gnb = ref_gid[idx[:, :k]]             # local ref ids -> global ids
        for lvl in ("ec1", "ec2", "ec3"):
            lab = EC[lvl]
            pred = np.array([Counter(lab[row]).most_common(1)[0][0] for row in gnb])
            out[f"{lvl}_{k}nn"] = round((pred == lab[q_gid]).mean(), 4)
    return out

def cluster_ami(Z, labels):
    """agglomerative average cosine at k = #distinct labels -> EC3 AMI."""
    k = len(np.unique(labels))
    pred = AgglomerativeClustering(n_clusters=k, metric="cosine",
                                   linkage="average").fit_predict(Z)
    return round(adjusted_mutual_info_score(labels, pred), 4)

# ---------------------------------------------------------------- model + SupCon
class Proj(nn.Module):
    def __init__(self, din, dh=512, dout=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, dh), nn.BatchNorm1d(dh), nn.ReLU(),
                                 nn.Linear(dh, dout))
    def forward(self, x):
        return nn.functional.normalize(self.net(x), dim=1)

def supcon(z, y, tau=0.1):
    """Supervised contrastive loss (Khosla 2020) over an in-batch PK sample."""
    y = y.view(-1, 1)
    pos = (y == y.t()).float()
    eye = torch.eye(z.size(0), device=z.device)
    pos = pos - eye                                   # exclude self
    logits = z @ z.t() / tau
    logits = logits - logits.max(dim=1, keepdim=True)[0].detach()
    exp = torch.exp(logits) * (1 - eye)               # exclude self in denom
    logp = logits - torch.log(exp.sum(1, keepdim=True) + 1e-9)
    npos = pos.sum(1)
    loss = -(pos * logp).sum(1) / npos.clamp(min=1)
    return loss[npos > 0].mean()

def train_proj(Ztr, ytr_ec3, epochs=80, P=48, K=8, lr=1e-3):
    """PK-sampled SupCon training on EC3 identity. Returns a fitted Proj (eval mode)."""
    labs, inv = np.unique(ytr_ec3, return_inverse=True)
    by_class = [np.where(inv == c)[0] for c in range(len(labs))]
    ok = [c for c in range(len(labs)) if len(by_class[c]) >= 2]   # need >=2 for positives
    Xt = torch.tensor(Ztr, device=DEV)
    yt = torch.tensor(inv, device=DEV)
    m = Proj(Ztr.shape[1]).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
    steps = max(1, len(Ztr) // (P * K))
    m.train()
    for ep in range(epochs):
        tot = 0.0
        for _ in range(steps):
            cls = rng.choice(ok, size=min(P, len(ok)), replace=False)
            idx = []
            for c in cls:
                pool = by_class[c]
                idx.extend(rng.choice(pool, size=min(K, len(pool)),
                                      replace=len(pool) < K))
            idx = np.array(idx)
            z = m(Xt[idx]); loss = supcon(z, yt[idx])
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        if ep % 20 == 0 or ep == epochs - 1:
            log(f"    epoch {ep:3d}  supcon={tot/steps:.4f}")
    m.eval()
    return m

@torch.no_grad()
def embed(m, Z):
    return m(torch.tensor(Z, device=DEV)).cpu().numpy()

# ---------------------------------------------------------------- (R) held-out reactions
log("=== (R) held-out reactions (80/20 stratified by EC3) ===")
ec3 = EC["ec3"]
labs, cnt = np.unique(ec3, return_counts=True)
test_idx = []
for lab in labs[cnt >= 5]:                            # stratify only classes with >=5
    ix = np.where(ec3 == lab)[0]
    n_te = max(1, int(round(0.2 * len(ix))))
    test_idx.extend(rng.choice(ix, size=n_te, replace=False))
test_idx = np.array(sorted(test_idx))
train_idx = np.setdiff1d(np.arange(N), test_idx)
log(f"  train {len(train_idx)}  test {len(test_idx)}")

Zbase = svd_space(Xraw[train_idx], Xraw)              # baseline geometry (fit on train)
m = train_proj(Zbase[train_idx], ec3[train_idx])
Zlearn = embed(m, Zbase)

rowsR = []
for tag, Z in [("baseline (SVD-256)", Zbase), ("metric-learned (128)", Zlearn)]:
    ret = retrieval(Z[train_idx], train_idx, Z[test_idx], test_idx)
    ami = cluster_ami(Z[test_idx], ec3[test_idx])     # cluster held-out reactions
    rowsR.append(dict(split="held-out reactions", space=tag, ec3_ami_test=ami, **ret))
    log(f"  {tag:22s} EC3 15-NN={ret['ec3_15nn']}  EC3 1-NN={ret['ec3_1nn']}  "
        f"clust EC3 AMI(test)={ami}")

# ---------------------------------------------------------------- (C) held-out classes
log("=== (C) held-out EC3 classes (metric never sees them) ===")
eligible = labs[cnt >= 10]
n_hold = max(5, int(round(0.2 * len(eligible))))
hold_cls = set(rng.choice(eligible, size=n_hold, replace=False))
held_idx = np.array([i for i in range(N) if ec3[i] in hold_cls])
seen_idx = np.array([i for i in range(N) if ec3[i] not in hold_cls])
log(f"  held-out {len(hold_cls)} classes, {len(held_idx)} reactions; train on {len(seen_idx)}")

ZbaseC = svd_space(Xraw[seen_idx], Xraw)
mC = train_proj(ZbaseC[seen_idx], ec3[seen_idx])
ZlearnC = embed(mC, ZbaseC)

rowsC = []
for tag, Z in [("baseline (SVD-256)", ZbaseC), ("metric-learned (128)", ZlearnC)]:
    # retrieval among held-out reactions only (their classes aren't in train ref) -> LOO
    ret = retrieval(Z[held_idx], held_idx, Z[held_idx], held_idx)
    ami = cluster_ami(Z[held_idx], ec3[held_idx])
    rowsC.append(dict(split="held-out classes", space=tag, ec3_ami_test=ami, **ret))
    log(f"  {tag:22s} EC3 15-NN={ret['ec3_15nn']}  EC3 1-NN={ret['ec3_1nn']}  "
        f"clust EC3 AMI(held)={ami}")

allrows = rowsR + rowsC
pd.DataFrame(allrows).to_csv(os.path.join(RESULTS, "metric_learning.csv"), index=False)
log("wrote results/metric_learning.csv")

# ---------------------------------------------------------------- FIGURE
fig, axes = plt.subplots(1, 2, figsize=(15, 7))
metrics = [("ec3_ami_test", "EC3 AMI (clustering)"), ("ec3_15nn", "EC3 15-NN acc"),
           ("ec3_1nn", "EC3 1-NN acc")]
for ax, rows, title in [(axes[0], rowsR, "Held-out reactions"),
                        (axes[1], rowsC, "Held-out EC3 classes")]:
    base = rows[0]; learn = rows[1]
    x = np.arange(len(metrics)); w = 0.36
    ax.bar(x - w/2, [base[k] for k, _ in metrics], w, color="#8593A3", label="baseline SVD-256")
    ax.bar(x + w/2, [learn[k] for k, _ in metrics], w, color="#0072B2", label="metric-learned")
    for xi, (k, _) in enumerate(metrics):
        for off, r in [(-w/2, base), (w/2, learn)]:
            ax.text(xi + off, r[k] + 0.01, f"{r[k]:.2f}", ha="center", va="bottom", fontsize=FS)
    ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in metrics], rotation=20, ha="right", fontsize=FS)
    ax.set_ylim(0, 1.0); ax.set_title(title, fontsize=FS)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.grid(axis="y", alpha=0.25); ax.set_axisbelow(True)
axes[0].set_ylabel("score", fontsize=FS)
axes[0].legend(fontsize=FS, frameon=False, loc="upper right")
fig.suptitle("Supervised metric learning vs the structural baseline", fontsize=FS)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_metric_learning.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
log("wrote figures/fig_metric_learning.png")
