#!/usr/bin/env python
"""Visualize the correction log: how many reactions fixed per category, colored by
confidence, plus the glycosyl stereo-verifiability breakdown. Reads corrections_log.tsv
+ couple_agent_corrections.tsv."""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = "/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/mass_charge_balance"
FIG = "/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/figures"

log = pd.read_csv(f"{OUT}/corrections_log.tsv", sep="\t", dtype=str, keep_default_na=False)

# confidence class per row
def conf_class(r):
    c = r["confidence"]
    if r["category"] == "proton_water": return "deterministic"
    if c in ("high", "deterministic"): return "high (EC-corroborated)"
    if c == "agent": return "agent (name-based)"
    return "candidate (review/low)"
log["cc"] = log.apply(conf_class, axis=1)

COLORS = {"deterministic": "#1b5e20", "high (EC-corroborated)": "#2e7d32",
          "agent (name-based)": "#1565c0", "candidate (review/low)": "#f9a825"}
ORDER = ["deterministic", "high (EC-corroborated)", "agent (name-based)", "candidate (review/low)"]

plt.rcParams.update({"font.size": 11})
fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True,
                               gridspec_kw={"width_ratios": [2.4, 1]})

# Panel A: fixes per category, stacked by confidence
cats = (log.groupby("category").size().sort_values(ascending=True))
ct = pd.crosstab(log["category"], log["cc"]).reindex(cats.index)
bottom = pd.Series(0, index=ct.index)
for cc in ORDER:
    if cc not in ct: continue
    axA.barh(ct.index, ct[cc], left=bottom, color=COLORS[cc], edgecolor="white", label=cc)
    bottom = bottom + ct[cc].fillna(0)
for i, cat in enumerate(ct.index):
    axA.text(cats[cat] + 3, i, str(int(cats[cat])), va="center", fontsize=9, fontweight="bold")
axA.set_xlabel("reactions corrected (mass+charge balance verified)")
total = len(log)
axA.set_title(f"A  Corrections by category & confidence  —  {total} reactions, all balance-verified",
              fontsize=12, fontweight="bold")
axA.legend(handles=[Patch(color=COLORS[c], label=c) for c in ORDER if c in ct],
           loc="lower right", fontsize=9, title="confidence")
axA.set_xlim(0, cats.max() * 1.12)

# Panel B: glycosyl stereo verifiability
try:
    ag = pd.read_csv(f"{OUT}/couple_agent_corrections.tsv", sep="\t", dtype=str, keep_default_na=False)
    st = ag[ag["stereo_tier"].isin(["structure-stereo-defined", "structure-stereo-undefined", "unverifiable"])]
    counts = st["stereo_tier"].value_counts()
    lbl = {"structure-stereo-defined": "stereo-verifiable\n(epimer checkable)",
           "structure-stereo-undefined": "scaffold OK,\nstereo undefined",
           "unverifiable": "unverifiable\n(rests on name)"}
    cols = {"structure-stereo-defined": "#2e7d32", "structure-stereo-undefined": "#f9a825",
            "unverifiable": "#c62828"}
    order = ["structure-stereo-defined", "structure-stereo-undefined", "unverifiable"]
    vals = [counts.get(k, 0) for k in order]
    axB.bar(range(3), vals, color=[cols[k] for k in order], edgecolor="black", linewidth=0.5)
    axB.set_xticks(range(3)); axB.set_xticklabels([lbl[k] for k in order], fontsize=9)
    for i, v in enumerate(vals):
        axB.text(i, v + 0.3, str(v), ha="center", fontweight="bold")
    axB.set_ylabel("glycosyl corrections")
    axB.set_title("B  Glycosyl fixes:\nbalance is stereo-blind", fontsize=12, fontweight="bold")
    axB.set_ylim(0, max(vals) * 1.2 + 1)
except FileNotFoundError:
    pass

fig.savefig(f"{OUT}/corrections_summary.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{FIG}/mcb_corrections_summary.png", dpi=300, bbox_inches="tight")
print(f"wrote corrections_summary.png  (total {total})")
