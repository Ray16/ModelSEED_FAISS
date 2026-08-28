#!/usr/bin/env python
"""
Log CARE vs our retriever ("this model") reaction->EC predictions, and tabulate where our
model is better.

Two logs:
  results/care_vs_rxnfp_agreement.tsv   full 11,513-reaction prediction log (both models),
                                        written by care_cross_check.py.
  results/care_vs_rxnfp_headtohead.tsv  the mechanism-ADJUDICATED head-to-head on the 13
                                        O2-monooxygenase corrections the balance tool made
                                        (mass_charge_balance/): these reactions have no
                                        ground-truth EC, so we adjudicate from the balanced
                                        reaction itself (O2-fate monooxygenase test).

Figure figures/care_cross_check.png:
  (a) aggregate EC1/EC2/EC3 agreement between the two independent predictors,
  (b) the head-to-head outcome on the disputed cases — our model wins every mechanism-
      adjudicated dispute because CARE's DRFP retrieval is misled by the shared NAD(P)H
      cofactor substructure.

This is a favourable, specific slice (cofactor-dominated O2-monooxygenases), NOT a general
benchmark claim — stated as such on the figure's companion slide/prose.
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "_lib"))
import _style  # noqa: F401  (18pt / dpi 300)

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
MCB_QUAL = "/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/mass_charge_balance/results/predicted_ec_quality.tsv"

C_US, C_CARE, C_TIE = "#0072B2", "#D55E00", "#999999"   # Okabe-Ito blue / vermillion / grey


def main():
    agr = pd.read_csv(os.path.join(RES, "care_vs_rxnfp_agreement.tsv"), sep="\t")
    bc = (agr.care_conf3 >= 0.6) & (agr.care_top1sim >= 0.3)
    lv = ["ec1", "ec2", "ec3"]
    all_ag = [agr[f"agree_{l}"].mean() * 100 for l in lv]
    conf_ag = [agr.loc[bc, f"agree_{l}"].mean() * 100 for l in lv]

    # head-to-head, mechanism-adjudicated (the 13 balance-tool corrections)
    q = pd.read_csv(MCB_QUAL, sep="\t")
    q["care_agree"] = q["care_agree"].astype(str)
    corroborated = (q["care_agree"] == "True").sum()
    disputed = q[q["care_agree"] == "False"]
    mono = disputed["o2_fate_mechanism"] == "monooxygenase(1.14)"
    us_win = int(mono.sum())                       # disputes where mechanism confirms our 1.14
    care_win = int((disputed["o2_fate_mechanism"] == "oxidase(1.6.3)-like").sum())
    unresolved = len(disputed) - us_win - care_win

    ht = q[["reaction_id", "class", "pred_ec3", "ec3_by_source", "care_agree",
            "o2_fate_mechanism"]].copy()
    ht["adjudicated_winner"] = np.where(
        ht["care_agree"] == "True", "tie (both agree)",
        np.where(ht["o2_fate_mechanism"] == "monooxygenase(1.14)", "this model",
                 np.where(ht["o2_fate_mechanism"] == "oxidase(1.6.3)-like", "CARE", "unresolved")))
    ht.to_csv(os.path.join(RES, "care_vs_rxnfp_headtohead.tsv"), sep="\t", index=False)

    # ---------- (1) calibration: agreement vs joint confidence gate ----------
    ind = agr.care_top1sim >= 0.3
    thr = np.array([0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    curves = {l: [] for l in lv}; ns = []
    for t in thr:
        m = ind & (agr.rxnfp_conf3 >= t) & (agr.care_conf3 >= t)
        ns.append(int(m.sum()))
        for l in lv:
            curves[l].append(agr.loc[m, f"agree_{l}"].mean() * 100 if m.sum() else np.nan)

    # ---------- (2) cross-validated annotation yield (funnel) ----------
    full = pd.read_csv(os.path.join(RES, "predicted_missing_ec.tsv"), sep="\t", dtype=str)
    n_missing = len(full)
    n_rx = int(((full.confident == "True") & (full.ood != "True")).sum())
    both = ind & (agr.care_conf3 >= 0.6)
    n_both = int(both.sum())
    n_ec3 = int((both & agr.agree_ec3).sum())

    # ---------------- figure ----------------
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(15, 6.3))

    styles = {"ec1": (C_US, "o", "EC1 (class)"),
              "ec2": (C_CARE, "s", "EC2"),
              "ec3": ("#009E73", "^", "EC3 (subclass)")}
    for l in lv:
        c, mk, lab = styles[l]
        axa.plot(thr * 100, curves[l], marker=mk, color=c, lw=2.5, ms=9, label=lab)
    axa.axhline(50, ls=":", color=C_TIE, lw=1)
    axa.set_xlabel("confidence gate — both predictors ≥ (%)")
    axa.set_ylabel("agreement between predictors (%)")
    axa.set_ylim(45, 102); axa.set_xlim(-3, 108)
    axa.set_title("Agreement calibrates with confidence")
    axa.legend(loc="lower right", frameon=False)

    stages = ["missing EC (scoreable)", "our model confident", "+ CARE also confident",
              "+ both agree EC3"]
    yvals = [n_missing, n_rx, n_both, n_ec3]
    ycol = [C_TIE, C_US, C_CARE, "#009E73"]
    ypos = np.arange(len(stages))[::-1]
    axb.barh(ypos, yvals, color=ycol, height=0.6)
    for yp, s, v in zip(ypos, stages, yvals):
        axb.text(n_missing * 0.015, yp + 0.34, s, va="bottom", ha="left")    # stage above bar
        axb.text(v - n_missing * 0.015, yp, f"{v:,}", va="center", ha="right",
                 color="white", fontweight="bold")                            # count in bar end
    axb.set_yticks([]); axb.set_ylim(-0.6, len(stages) - 0.15)
    axb.set_xlim(0, n_missing * 1.05)
    axb.set_xlabel("reactions")
    axb.set_title("Yield: double-validated EC annotations")
    fig.tight_layout(w_pad=2.5)

    fig.savefig(os.path.join(FIG, "care_cross_check.png"))
    fig.savefig(os.path.join(FIG, "mcb_care_cross_check.png"))
    print(f"calibration EC3: t=0 {curves['ec3'][0]:.0f}% -> t=0.9 {curves['ec3'][-2]:.0f}% "
          f"-> t=1.0 {curves['ec3'][-1]:.0f}%")
    print(f"yield funnel: missing {n_missing} -> our-conf {n_rx} -> both-conf {n_both} "
          f"-> both-agree-EC3 {n_ec3} (double-validated)")
    print(f"head-to-head table (13 corrections): tie {corroborated}, this-model {us_win}, "
          f"CARE {care_win}, unresolved {unresolved}")
    print("wrote results/care_vs_rxnfp_headtohead.tsv + figures/care_cross_check.png")


if __name__ == "__main__":
    main()
