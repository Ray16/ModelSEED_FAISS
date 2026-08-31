#!/usr/bin/env python
"""
Figure for the fair external benchmark (care_task2_benchmark.py): our reaction->EC
retriever vs CARE's task-2 baselines on CARE's OWN Rhea/SwissProt ground truth.

Two panels:
  (a) EC3 top-1 accuracy, grouped by CARE difficulty split (easy/medium/hard),
      one bar per method. The head-to-head "is the method better?" answer.
  (b) accuracy vs EC level (EC1->full) at top-1, pooled over splits (mean), showing
      how the gap behaves as the label gets finer.

Reads results/care_task2_benchmark.csv. Writes figures/care/care_task2_benchmark.png.
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "_lib"))
import _style  # noqa: F401  (18pt / dpi 300)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results", "care_task2_benchmark.csv")
OUT = os.path.join(ROOT, "figures", "care", "care_task2_benchmark.png")

df = pd.read_csv(RES)
df["label"] = df.method.map({
    "ours": "Ours (ECFP4-diff‖DRFP)", "Similarity": "CARE Similarity (DRFP)",
    "CREEP": "CARE CREEP (learned)", "CLIPZyme": "CLIPZyme", "ChatGPT": "ChatGPT",
    "random": "random",
}).fillna(df.method)
# distinguish our two references
df.loc[df.method == "ours", "label"] = "Ours [" + df.reference + "]"

# method display order + colours (Okabe-Ito)
ORDER = ["Ours [CARE-train]", "Ours [ModelSEED]", "CARE Similarity (DRFP)",
         "CARE CREEP (learned)", "CLIPZyme", "ChatGPT", "random"]
COL = {"Ours [CARE-train]": "#0072B2", "Ours [ModelSEED]": "#56B4E9",
       "CARE Similarity (DRFP)": "#D55E00", "CARE CREEP (learned)": "#009E73",
       "CLIPZyme": "#E69F00", "ChatGPT": "#CC79A7", "random": "#999999"}
SPLITS = ["easy", "medium", "hard"]

present = [m for m in ORDER if m in set(df.label)]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(20, 8),
                               gridspec_kw={"width_ratios": [1.55, 1]})

# ---- (a) EC3 top-1, grouped by split ----
d = df[(df.level == "ec3") & (df.k == 1)]
nb, ng = len(present), len(SPLITS)
w = 0.8 / nb
x = np.arange(ng)
for j, m in enumerate(present):
    vals = [d[(d.label == m) & (d.split == s)].acc.mean() for s in SPLITS]
    axA.bar(x + (j - (nb - 1) / 2) * w, vals, w, color=COL[m], label=m,
            edgecolor="white", linewidth=0.6)
axA.set_xticks(x); axA.set_xticklabels([s.capitalize() for s in SPLITS])
axA.set_ylabel("EC3 top-1 accuracy")
axA.set_xlabel("CARE difficulty split")
axA.set_ylim(0, 1.02)
axA.grid(axis="y", ls=":", alpha=0.5)
axA.set_title("(a) EC3 subclass, top-1")
# the 'hard' split holds out entire EC3 classes -> train-restricted retrievers
# cannot hit them (0.0). Annotate so the collapse reads as coverage, not a bug.
axA.annotate("hard split holds out\nwhole EC3 classes\n(0% present in CARE-train)",
             xy=(2, 0.02), xytext=(1.5, 0.52),
             ha="center", va="center", fontsize=18,
             arrowprops=dict(arrowstyle="->", color="#555", lw=1.5))
axA.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4,
           frameon=False)

# ---- (b) robustness across difficulty: EC1 top-1 vs split ----
# the story panel: train-restricted methods fall off easy->hard; our deployed
# ModelSEED-reference tool stays flat because ModelSEED covers 98% of classes.
d1 = df[(df.level == "ec1") & (df.k == 1)]
for m in present:
    ys = [d1[(d1.label == m) & (d1.split == s)].acc.mean() for s in SPLITS]
    lw = 4.0 if m == "Ours [ModelSEED]" else 2.5
    axB.plot(range(len(SPLITS)), ys, "o-", color=COL[m], lw=lw, ms=10, label=m,
             zorder=5 if m == "Ours [ModelSEED]" else 3)
axB.set_xticks(range(len(SPLITS))); axB.set_xticklabels([s.capitalize() for s in SPLITS])
axB.set_ylabel("EC1 top-1 accuracy")
axB.set_xlabel("CARE difficulty split")
axB.set_ylim(0, 1.02)
axB.grid(axis="y", ls=":", alpha=0.5)
axB.set_title("(b) robustness: EC1 top-1 vs difficulty")

fig.savefig(OUT, dpi=300, bbox_inches="tight")
print("wrote", OUT)

# also print the headline table
piv = (d.pivot_table(index="label", columns="split", values="acc")
       .reindex(index=[m for m in ORDER if m in d.label.values], columns=SPLITS))
print("\n=== EC3 top-1 accuracy (fair external ground truth) ===")
print(piv.round(3).to_string())
