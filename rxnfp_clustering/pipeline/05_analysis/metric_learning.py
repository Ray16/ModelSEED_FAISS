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

Run over N_SEEDS seeds (new split + init each) and report mean +/- std, because a
single run is noisy. Read the result TRANSFER-FIRST: the honest question is (C), and
the metric gives NO reliable gain on held-out CLASSES (Δ within noise). The large (R)
gain is in-distribution densification of classes the metric was trained on -- expected
by construction, not evidence of transfer.

Outputs: results/metric_learning.csv (mean/std per split x space),
         figures/fig_metric_learning.png (bars with std error bars).
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

# ---------------------------------------------------------------- multi-seed runs
N_SEEDS = 5
ec3 = EC["ec3"]
labs, cnt = np.unique(ec3, return_counts=True)
METRICS = [("ec3_ami_test", "EC3 AMI\n(clustering)"), ("ec3_15nn", "EC3 15-NN\nacc"),
           ("ec3_1nn", "EC3 1-NN\nacc")]
SPACES = ["baseline (SVD-256)", "metric-learned (128)"]

def run_R(rloc):
    """(R) held-out reactions: 80/20 stratified by EC3, reference = train."""
    test_idx = []
    for lab in labs[cnt >= 5]:
        ix = np.where(ec3 == lab)[0]
        test_idx.extend(rloc.choice(ix, size=max(1, int(round(0.2 * len(ix)))), replace=False))
    test_idx = np.array(sorted(test_idx)); train_idx = np.setdiff1d(np.arange(N), test_idx)
    Zbase = svd_space(Xraw[train_idx], Xraw)
    Zlearn = embed(train_proj(Zbase[train_idx], ec3[train_idx]), Zbase)
    out = {}
    for tag, Z in [(SPACES[0], Zbase), (SPACES[1], Zlearn)]:
        ret = retrieval(Z[train_idx], train_idx, Z[test_idx], test_idx)
        out[tag] = dict(ec3_ami_test=cluster_ami(Z[test_idx], ec3[test_idx]), **ret)
    return out

def run_C(rloc):
    """(C) held-out EC3 classes: metric never sees them; LOO among held-out."""
    eligible = labs[cnt >= 10]
    hold = set(rloc.choice(eligible, size=max(5, int(round(0.2 * len(eligible)))), replace=False))
    held = np.array([i for i in range(N) if ec3[i] in hold])
    seen = np.array([i for i in range(N) if ec3[i] not in hold])
    Zb = svd_space(Xraw[seen], Xraw)
    Zl = embed(train_proj(Zb[seen], ec3[seen]), Zb)
    out = {}
    for tag, Z in [(SPACES[0], Zb), (SPACES[1], Zl)]:
        ret = retrieval(Z[held], held, Z[held], held)
        out[tag] = dict(ec3_ami_test=cluster_ami(Z[held], ec3[held]), **ret)
    return out

# acc[split][space][metric] = list over seeds
acc = {"held-out reactions": {s: {k: [] for k, _ in METRICS} for s in SPACES},
       "held-out classes":   {s: {k: [] for k, _ in METRICS} for s in SPACES}}
for seed in range(N_SEEDS):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)   # reseed (train_proj uses global rng)
    log(f"=== seed {seed} ===")
    for split, fn in [("held-out reactions", run_R), ("held-out classes", run_C)]:
        res = fn(rng)
        for s in SPACES:
            for k, _ in METRICS:
                acc[split][s][k].append(res[s][k])
        log(f"  {split:20s} metric EC3 AMI={res[SPACES[1]]['ec3_ami_test']:.3f}  "
            f"15-NN={res[SPACES[1]]['ec3_15nn']:.3f}")

# ---- aggregate mean/std, write CSV ----
rows = []
for split in acc:
    for s in SPACES:
        row = dict(split=split, space=s, n_seeds=N_SEEDS)
        for k, _ in METRICS:
            a = np.array(acc[split][s][k])
            row[f"{k}_mean"] = round(a.mean(), 4); row[f"{k}_std"] = round(a.std(), 4)
        rows.append(row)
pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "metric_learning.csv"), index=False)
log("wrote results/metric_learning.csv")

# ---- conclusion (transfer-first framing) ----
def delta(split, k):
    d = np.array(acc[split][SPACES[1]][k]) - np.array(acc[split][SPACES[0]][k])
    return d.mean(), d.std()
for split in acc:
    dm, ds = delta(split, "ec3_ami_test")
    verdict = "REAL gain" if dm - 2 * ds > 0 else ("REAL loss" if dm + 2 * ds < 0 else "within noise")
    log(f"CONCLUSION {split:20s}: metric-vs-baseline EC3 AMI Δ = {dm:+.3f} ± {ds:.3f}  -> {verdict}")

# ---------------------------------------------------------------- FIGURE (with error bars)
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
handles = None
for ax, split, title in [(axes[0], "held-out reactions", "Held-out reactions"),
                         (axes[1], "held-out classes", "Held-out EC3 classes")]:
    x = np.arange(len(METRICS)); w = 0.36
    means = {s: [np.mean(acc[split][s][k]) for k, _ in METRICS] for s in SPACES}
    stds = {s: [np.std(acc[split][s][k]) for k, _ in METRICS] for s in SPACES}
    b1 = ax.bar(x - w/2, means[SPACES[0]], w, yerr=stds[SPACES[0]], capsize=6,
                color="#8593A3", error_kw=dict(lw=2))
    b2 = ax.bar(x + w/2, means[SPACES[1]], w, yerr=stds[SPACES[1]], capsize=6,
                color="#0072B2", error_kw=dict(lw=2))
    handles = [b1, b2]
    for xi, (k, _) in enumerate(METRICS):
        for off, s in [(-w/2, SPACES[0]), (w/2, SPACES[1])]:
            ax.text(xi + off, means[s][xi] + stds[s][xi] + 0.02, f"{means[s][xi]:.2f}",
                    ha="center", va="bottom", fontsize=FS)
    ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in METRICS], fontsize=FS)
    ax.set_ylim(0, 1.22); ax.set_title(title, fontsize=FS, pad=12)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.grid(axis="y", alpha=0.25); ax.set_axisbelow(True)
axes[0].set_ylabel("score", fontsize=FS)
fig.legend(handles, ["baseline SVD-256", "metric-learned"], fontsize=FS, frameon=False,
           loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_metric_learning.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
log(f"wrote figures/fig_metric_learning.png (mean±std over {N_SEEDS} seeds)")
