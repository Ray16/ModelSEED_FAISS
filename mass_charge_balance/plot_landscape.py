#!/usr/bin/env python
"""Visualize the ModelSEED mass/charge imbalance landscape.

Story: the ~48% "imbalanced" fraction is not one problem. Recast every reaction
by *actionability*: already OK, must-refuse (no formula), deterministically
fixable (H+/H2O/charge), or genuinely hard (heavy-atom skeleton error).
"""
import json
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = "/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/mass_charge_balance"
FIGROOT = "/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/figures"

df = pd.read_csv(f"{OUT}/landscape.tsv", sep="\t", keep_default_na=False)
N = len(df)

# --- action-class mapping -------------------------------------------------
CLASS = {
    "OK":               ("Already balanced",        "#2e7d32"),
    "A_no_formula":     ("Unscoreable (no formula)", "#9e9e9e"),
    "B_unknown_charge": ("Deterministic fix",        "#1565c0"),
    "C_H_only":         ("Deterministic fix",        "#1565c0"),
    "D_H_and_O":        ("Deterministic fix",        "#1565c0"),
    "E_charge_only":    ("Deterministic fix",        "#1565c0"),
    "F_heavy_imbalance":("Hard (heavy-atom error)",  "#e65100"),
}
df["action"] = df["bucket"].map(lambda b: CLASS[b][0])
action_color = {v[0]: v[1] for v in CLASS.values()}
ACTION_ORDER = ["Already balanced", "Unscoreable (no formula)",
                "Deterministic fix", "Hard (heavy-atom error)"]

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold"})
fig = plt.figure(figsize=(15, 10), constrained_layout=True)
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15])

# =========================================================================
# Panel A: single stacked bar of ALL reactions by actionability
# =========================================================================
axA = fig.add_subplot(gs[0, :])
counts = df["action"].value_counts()
seg_centers = {}
left = 0
for act in ACTION_ORDER:
    c = int(counts.get(act, 0))
    axA.barh(0, c, left=left, color=action_color[act], edgecolor="white", height=0.6)
    seg_centers[act] = left + c / 2
    if c / N > 0.10:   # only the two wide segments get inline labels
        axA.text(left + c / 2, 0, f"{act}\n{c:,}  ({100*c/N:.1f}%)",
                 ha="center", va="center", color="white", fontsize=11, fontweight="bold")
    left += c
axA.set_xlim(0, N)
axA.set_ylim(-0.6, 1.25)
axA.set_yticks([])
axA.set_xlabel(f"Reactions (n = {N:,} active, non-obsolete)")
axA.set_title("A  All ModelSEED reactions by actionability — \"48% imbalanced\" is mostly unscoreable R-groups, "
              "not fixable errors", pad=14)
# Callouts for the two thin segments, placed above the bar and pulled apart
det = int(counts.get("Deterministic fix", 0))
hard = int(counts.get("Hard (heavy-atom error)", 0))
axA.annotate(f"Deterministic fix\n{det:,}  ({100*det/N:.1f}%)",
             xy=(seg_centers["Deterministic fix"], 0.30), xytext=(0.72*N, 0.95),
             fontsize=10, color="#1565c0", fontweight="bold", ha="center",
             arrowprops=dict(arrowstyle="->", color="#1565c0", lw=1.5))
axA.annotate(f"Hard (heavy-atom)\n{hard:,}  ({100*hard/N:.1f}%)",
             xy=(seg_centers["Hard (heavy-atom error)"], 0.30), xytext=(0.93*N, 0.95),
             fontsize=10, color="#e65100", fontweight="bold", ha="center",
             arrowprops=dict(arrowstyle="->", color="#e65100", lw=1.5))

# =========================================================================
# Panel B: imbalanced-only, per-bucket bar (log-ish), colored by action
# =========================================================================
axB = fig.add_subplot(gs[1, 0])
imb = df[df["bucket"] != "OK"]
order = ["A_no_formula", "F_heavy_imbalance", "E_charge_only",
         "D_H_and_O", "B_unknown_charge", "C_H_only"]
labels = {"A_no_formula": "A: no/generic\nformula",
          "F_heavy_imbalance": "F: heavy-atom\nimbalance",
          "E_charge_only": "E: charge only\n(mass OK)",
          "D_H_and_O": "D: H + O off\n(add H2O/H+)",
          "B_unknown_charge": "B: unknown\ncharge",
          "C_H_only": "C: H only\n(add H+)"}
vals = [(df["bucket"] == b).sum() for b in order]
cols = [CLASS[b][1] for b in order]
bars = axB.bar(range(len(order)), vals, color=cols, edgecolor="black", linewidth=0.5)
axB.set_yscale("log")
axB.set_xticks(range(len(order)))
axB.set_xticklabels([labels[b] for b in order], fontsize=9)
axB.set_ylabel("Reactions (log scale)")
axB.set_title("B  Imbalanced reactions by defect type")
for bar, v in zip(bars, vals):
    axB.text(bar.get_x() + bar.get_width()/2, v*1.1, f"{v:,}",
             ha="center", va="bottom", fontsize=9, fontweight="bold")
axB.set_ylim(1, max(vals)*2)

# =========================================================================
# Panel C: charge-imbalance magnitude distribution (E + any nonzero charge)
# =========================================================================
axC = fig.add_subplot(gs[1, 1])
ch = df[(df["charge_imbalance"] != 0)]["charge_imbalance"].astype(float)
ch = ch[np.abs(ch) <= 10]  # ignore sentinel-driven extremes for readability
bins = np.arange(-10.5, 11.5, 1)
axC.hist(ch, bins=bins, color="#1565c0", edgecolor="white")
axC.axvline(0, color="black", lw=0.8)
axC.set_xlabel("Net charge imbalance (RHS − LHS)")
axC.set_ylabel("Reactions")
axC.set_title(f"C  Charge imbalance is small & integer\n(|Δq| ≤ 10 shown; n = {len(ch):,}) → proton-count fixable")
axC.set_xticks(range(-10, 11, 2))

fig.savefig(f"{OUT}/imbalance_landscape.png", dpi=300, bbox_inches="tight")
fig.savefig(f"{FIGROOT}/mcb_imbalance_landscape.png", dpi=300, bbox_inches="tight")
print("wrote imbalance_landscape.png")

# =========================================================================
# Second figure: F heavy-atom element signatures (what's actually missing)
# =========================================================================
fh = df[df["bucket"] == "F_heavy_imbalance"]
sig = Counter()
for e in fh["elem_imbalance"]:
    d = json.loads(e) if e else {}
    key = tuple(sorted(set(d) - {"H"}))
    if key:
        sig[key] += 1
top = sig.most_common(14)

fig2, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
names = ["+".join(k) for k, _ in top][::-1]
vv = [v for _, v in top][::-1]
# color: O-only or O+P (likely water/phosphate) blue-ish = more fixable
def sigcolor(name):
    els = set(name.split("+"))
    if els <= {"O"}:
        return "#42a5f5"       # just oxygen -> often water
    if els <= {"O", "P"}:
        return "#7e57c2"       # phosphate bookkeeping
    return "#e65100"           # carbon skeleton etc -> genuinely hard
cols = [sigcolor(n) for n in names]
ax.barh(range(len(names)), vv, color=cols, edgecolor="black", linewidth=0.5)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names)
ax.set_xlabel("Reactions")
ax.set_title("F: heavy-atom imbalance signatures (non-H elements off)\n"
             f"n = {len(fh):,} reactions", fontweight="bold")
for i, v in enumerate(vv):
    ax.text(v + max(vv)*0.01, i, str(v), va="center", fontsize=9)
legend = [Patch(color="#42a5f5", label="O only (candidate: water)"),
          Patch(color="#7e57c2", label="O+P (phosphate bookkeeping)"),
          Patch(color="#e65100", label="C-containing (skeleton — hard)")]
ax.legend(handles=legend, loc="lower right", fontsize=9)
fig2.savefig(f"{OUT}/heavy_imbalance_signatures.png", dpi=300, bbox_inches="tight")
fig2.savefig(f"{FIGROOT}/mcb_heavy_imbalance_signatures.png", dpi=300, bbox_inches="tight")
print("wrote heavy_imbalance_signatures.png")
