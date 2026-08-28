#!/usr/bin/env python
"""
Visualise the ground-truth adjudication: when our model and CARE disagree on a LABELLED
reaction, which matches ModelSEED's curated EC? — and how much of our model's edge is
near-duplicate leakage (LOO doesn't remove twins).

Panel A: among disagreements (leakage-corrected: our model has no near-duplicate twin),
         stacked bars EC1/EC2/EC3 = our model correct / CARE correct / both wrong.
Panel B: honesty — the near-duplicate leakage fraction, and how our model's numbers deflate
         when those twins are removed (EC3 disagreement win 81->73, overall acc 93->90).

Reads results/care_who_is_right.tsv + recomputes our model's LOO top-1 similarity.
Env: palm.
"""
import os, sys
import numpy as np, pandas as pd
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "_lib"))
import _style  # noqa

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(HERE, "results"); FIG = os.path.join(HERE, "figures")
C_US, C_CARE, C_BOTH = "#0072B2", "#D55E00", "#bbbbbb"


def main():
    w = pd.read_csv(os.path.join(RES, "care_who_is_right.tsv"), sep="\t")
    # recompute our-model LOO top-1 similarity (near-duplicate detector)
    Re = np.load(os.path.join(HERE, "embeddings", "ecfp4_diff.npy")).astype(np.float32)
    Rd = np.load(os.path.join(HERE, "embeddings", "drfp.npy")).astype(np.float32)
    R = normalize(np.hstack([Re, Rd]))
    nn = NearestNeighbors(n_neighbors=2, metric="cosine", algorithm="brute").fit(R)
    dist, _ = nn.kneighbors(R)
    w["us_top1sim"] = 1.0 - dist[:, 1]

    conf = (w.us_conf >= 0.6) & (w.care_conf >= 0.6) & (w.care_top1sim < 0.999)
    nolk = conf & (w.us_top1sim < 0.99)                       # our model has no near-dup twin

    def outcome(mask, L):
        d = w[mask & (w[f"us_{L}"] != w[f"care_{L}"])]
        n = len(d)
        u = (d[f"us_{L}"] == d[f"truth_{L}"]).mean() * 100
        c = (d[f"care_{L}"] == d[f"truth_{L}"]).mean() * 100
        return u, c, 100 - u - c, n

    lv = ["ec1", "ec2", "ec3"]
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(15, 6.3), gridspec_kw={"width_ratios": [1.35, 1]})

    # ---- Panel A: leakage-corrected disagreement outcomes ----
    us = [outcome(nolk, L)[0] for L in lv]
    ca = [outcome(nolk, L)[1] for L in lv]
    bw = [outcome(nolk, L)[2] for L in lv]
    ns = [outcome(nolk, L)[3] for L in lv]
    y = np.arange(3)[::-1]
    axa.barh(y, us, color=C_US, label="our model correct")
    axa.barh(y, ca, left=us, color=C_CARE, label="CARE correct")
    axa.barh(y, bw, left=np.add(us, ca), color=C_BOTH, label="both wrong")
    for yi, u, c in zip(y, us, ca):
        axa.text(u / 2, yi, f"{u:.0f}%", ha="center", va="center", color="white", fontweight="bold")
        axa.text(u + c / 2 + 1, yi, f"{c:.0f}%", ha="center", va="center", color="white", fontweight="bold")
    axa.set_yticks(y); axa.set_yticklabels([f"{L.upper()}\n(n={n})" for L, n in zip(lv, ns)])
    axa.set_xlim(0, 100); axa.set_xlabel("% of disagreements matching ground truth")
    axa.set_title("When they disagree, who's right?\n(labelled reactions, near-dup leakage removed)")
    axa.legend(loc="lower center", bbox_to_anchor=(0.5, -0.34), ncol=3, frameon=False)

    # ---- Panel B: the leakage effect ----
    frac_nd = (w.us_top1sim >= 0.99).mean() * 100
    ec3_all = outcome(conf, "ec3"); ec3_nl = outcome(nolk, "ec3")
    acc_all = (w[conf].us_ec3 == w[conf].truth_ec3).mean() * 100
    acc_nl = (w[nolk].us_ec3 == w[nolk].truth_ec3).mean() * 100
    cats = ["near-dup\nin ref", "EC3-disagree\nwin", "overall\nEC3 acc"]
    before = [frac_nd, ec3_all[0], acc_all]
    after = [np.nan, ec3_nl[0], acc_nl]
    x = np.arange(3); wd = 0.38
    axb.bar(x[0], before[0], wd * 2, color="#888")
    axb.bar(x[1:] - wd / 2, before[1:], wd, color=C_US, alpha=0.45, label="LOO (with leakage)")
    axb.bar(x[1:] + wd / 2, after[1:], wd, color=C_US, label="near-dup removed")
    axb.text(x[0], before[0] + 2, f"{before[0]:.0f}%", ha="center")
    for xi, v in zip(x[1:] - wd / 2, before[1:]):
        axb.text(xi, v + 2, f"{v:.0f}", ha="center", fontsize=13)
    for xi, v in zip(x[1:] + wd / 2, after[1:]):
        axb.text(xi, v + 2, f"{v:.0f}", ha="center", fontsize=13)
    axb.set_xticks(x); axb.set_xticklabels(cats)
    axb.set_ylim(0, 112); axb.set_ylabel("%")
    axb.set_title("Near-duplicate leakage & its effect")
    axb.legend(loc="upper left", frameon=False, fontsize=15)

    fig.tight_layout(w_pad=2)
    fig.savefig(os.path.join(FIG, "care_who_is_right.png"))
    fig.savefig(os.path.join(FIG, "mcb_care_who_is_right.png"))
    print(f"near-dup fraction {frac_nd:.0f}%; EC3 disagreement win {ec3_all[0]:.0f}%->{ec3_nl[0]:.0f}%; "
          f"overall acc {acc_all:.0f}%->{acc_nl:.0f}%")
    print("wrote figures/care_who_is_right.png")


if __name__ == "__main__":
    main()
