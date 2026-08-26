#!/usr/bin/env python
"""Validate the missing-EC predictions instead of assuming the benchmark number.

annotate_missing.py reports 11,513 confident EC3 calls, but their accuracy was only
*assumed* from the 0.90 held-out benchmark. Two honest checks:

  (A) Twin separation. Confident calls whose nearest labelled reaction is an EXACT
      structural twin (cos >= 0.999) are trivial label propagation, not prediction.
      We split them out so the headline is not inflated by them.

  (B) Distribution-matched accuracy. On the LABELLED set we run leave-one-out
      retrieval with the SAME ecfp4_drfp geometry + gate, and measure gated EC3
      accuracy as a function of nearest-neighbour similarity. We then reweight those
      per-band accuracies by the missing set's OWN similarity distribution -> an
      estimate of the accuracy the tool actually achieves on the 22,871 missing
      reactions (as opposed to the benchmark-average 0.90).

Outputs: results/prediction_validation.csv, figures/fig_prediction_validation.png
"""
import os, sys, time
import numpy as np, pandas as pd
from collections import Counter
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "_lib"))
import _style; FS = _style.FS

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(HERE, "data"); RESULTS = os.path.join(HERE, "results")
EMB = os.path.join(HERE, "embeddings"); FIG = os.path.join(HERE, "figures")
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, file=sys.stderr, flush=True)

GATE = 0.6
BANDS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.999, 1.0001]   # last bin = exact twins
BLAB = ["0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-0.95", "0.95-1.0", "twin=1.0"]

# ---- labelled reference geometry (ecfp4_drfp: pure structural -> L2 -> SVD256 -> L2) ----
df = pd.read_parquet(os.path.join(DATA, "dataset.parquet"))
ec3 = df["ec3"].to_numpy(); N = len(df)
from sklearn.decomposition import TruncatedSVD
Re = np.load(os.path.join(EMB, "ecfp4_diff.npy")).astype(np.float32)
Rd = np.load(os.path.join(EMB, "drfp.npy")).astype(np.float32)
X = normalize(np.hstack([Re, Rd]).astype(np.float32))
Z = TruncatedSVD(n_components=256, random_state=0).fit_transform(X)
Z = normalize(Z).astype(np.float32)
log(f"labelled geometry {Z.shape}")

nn = NearestNeighbors(n_neighbors=16, metric="cosine", algorithm="brute").fit(Z)
dist, idx = nn.kneighbors(Z); dist, idx = dist[:, 1:], idx[:, 1:]   # drop self
top1_sim = 1.0 - dist[:, 0]
pred3, agree = [], []
for row in idx[:, :15]:
    c = Counter(ec3[row]); t, m = c.most_common(1)[0]
    pred3.append(t); agree.append(m / 15)
pred3 = np.array(pred3); agree = np.array(agree)
correct = (pred3 == ec3)
conf = agree >= GATE

# ---- (B) labelled gated EC3 accuracy per similarity band ----
band = np.digitize(top1_sim, BANDS) - 1
rows = []
lab_acc = {}
for b in range(len(BLAB)):
    sel = conf & (band == b)
    acc = correct[sel].mean() if sel.any() else np.nan
    lab_acc[b] = acc
    rows.append(dict(sim_band=BLAB[b], labelled_gated_acc=round(acc, 4) if sel.any() else np.nan,
                     labelled_n=int(sel.sum())))

# ---- missing predictions: confident, in-domain, per band + twin split ----
pred = pd.read_csv(os.path.join(RESULTS, "predicted_missing_ec.tsv"), sep="\t")
mconf = pred[(pred.confident) & (~pred.ood)].copy()
mband = np.digitize(mconf.top1_sim.to_numpy(), BANDS) - 1
n_conf = len(mconf)
n_twin = int((mconf.top1_sim >= 0.999).sum())
n_genuine = n_conf - n_twin

expected_acc = 0.0; w_total = 0.0
for b in range(len(BLAB)):
    w = int((mband == b).sum())
    rows[b]["missing_n"] = w
    rows[b]["missing_frac"] = round(w / n_conf, 4)
    if w and not np.isnan(lab_acc[b]):
        expected_acc += w * lab_acc[b]; w_total += w
expected_acc = expected_acc / w_total if w_total else np.nan
# genuine-only (exclude twin band) expected accuracy
exp_gen, wg = 0.0, 0.0
for b in range(len(BLAB) - 1):                      # exclude last (twin) band
    w = int((mband == b).sum())
    if w and not np.isnan(lab_acc[b]): exp_gen += w * lab_acc[b]; wg += w
exp_gen = exp_gen / wg if wg else np.nan

pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "prediction_validation.csv"), index=False)
log("wrote results/prediction_validation.csv")
log("=" * 64)
log(f"confident missing-EC calls: {n_conf:,}")
log(f"  exact-twins (trivial propagation, cos=1.0): {n_twin:,} ({n_twin/n_conf*100:.1f}%)")
log(f"  genuine novel predictions:                  {n_genuine:,} ({n_genuine/n_conf*100:.1f}%)")
log(f"distribution-matched expected EC3 accuracy (ALL confident):     {expected_acc:.3f}")
log(f"distribution-matched expected EC3 accuracy (GENUINE non-twin):  {exp_gen:.3f}")
log("(vs the benchmark-average 0.90 that was previously assumed)")

# ---------------------------------------------------------------- FIGURE
res = pd.DataFrame(rows)
fig, ax = plt.subplots(1, 2, figsize=(18, 7))
# (a) labelled gated accuracy vs band + missing distribution overlay
x = np.arange(len(BLAB))
ax[0].bar(x, res.labelled_gated_acc, color="#009E73", alpha=0.85, label="labelled gated EC3 acc")
for xi in x:
    if not np.isnan(res.labelled_gated_acc[xi]):
        ax[0].text(xi, res.labelled_gated_acc[xi] + 0.01, f"{res.labelled_gated_acc[xi]:.2f}",
                   ha="center", va="bottom", fontsize=FS)
ax[0].set_xticks(x); ax[0].set_xticklabels(BLAB, rotation=25, ha="right", fontsize=FS)
ax[0].set_ylabel("gated EC3 accuracy (labelled LOO)", fontsize=FS)
ax[0].set_ylim(0, 1.05); ax[0].set_title("(a) accuracy vs neighbour similarity", fontsize=FS)
ax2 = ax[0].twinx()
ax2.plot(x, res.missing_frac, "o--", color="#D55E00", lw=2.5, ms=10, label="missing-set weight")
ax2.set_ylabel("fraction of confident missing calls", fontsize=FS, color="#D55E00")
ax2.tick_params(axis="y", labelcolor="#D55E00")
ax[0].legend(fontsize=FS, frameon=False, loc="lower right")
# (b) twin vs genuine + expected accuracy
cats = ["exact-twin\n(cos=1.0)", "genuine\nnovel"]
vals = [n_twin, n_genuine]
ax[1].bar(cats, vals, color=["#8593A3", "#0072B2"])
for i, v in enumerate(vals):
    ax[1].text(i, v + max(vals)*0.01, f"{v:,}", ha="center", va="bottom", fontsize=FS)
ax[1].set_ylabel("confident predictions", fontsize=FS)
ax[1].set_title(f"(b) expected EC3 acc: all {expected_acc:.2f} / genuine {exp_gen:.2f}", fontsize=FS)
for a in (ax[1],):
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
    a.grid(axis="y", alpha=0.25); a.set_axisbelow(True)
for sp in ("top",): ax[0].spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_prediction_validation.png"), dpi=300, bbox_inches="tight")
log("wrote figures/fig_prediction_validation.png")
