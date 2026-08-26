#!/usr/bin/env python
"""Replot the 3-view fusion ranking as a LANDSCAPE figure (no recomputation).
Reads results/triple_combos.csv (produced by triple_combos.py) and writes a
wide, slide-friendly figures/fig_triple_combos.png. All text 18pt (import _style).
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "_lib"))
import _style  # 18pt fonts, dpi 300
FS = _style.FS

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(HERE, "results"); FIG = os.path.join(HERE, "figures")
res = pd.read_csv(os.path.join(RESULTS, "triple_combos.csv"))

TOP = 12
top = res.head(TOP).iloc[::-1]                 # best at top after invert
y = np.arange(len(top)); h = 0.40
fig, ax = plt.subplots(figsize=(20, 9))        # landscape (aspect ~2.2)
ax.barh(y + h/2, top.ec3_ami,     h, color="#0072B2", label="EC3 AMI (cluster)")
ax.barh(y - h/2, top.full_ec_1nn, h, color="#E69F00", label="full-EC 1-NN (retrieval)")
for yi, (a, f) in enumerate(zip(top.ec3_ami, top.full_ec_1nn)):
    ax.text(a + 0.004, yi + h/2, f"{a:.3f}", va="center", fontsize=FS)
    ax.text(f + 0.004, yi - h/2, f"{f:.3f}", va="center", fontsize=FS)
ax.set_yticks(y); ax.set_yticklabels(top.combo, fontsize=FS)
ax.tick_params(axis="x", labelsize=FS)
ax.set_xlim(0, max(top.ec3_ami.max(), top.full_ec_1nn.max()) + 0.10)
ax.set_xlabel("score", fontsize=FS)
ax.set_title(f"Top {TOP} three-view fusion combos, ranked by EC3 AMI", fontsize=FS)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.grid(axis="x", alpha=0.25); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=FS, loc="upper center",
          bbox_to_anchor=(0.5, -0.12), ncol=2)
fig.tight_layout()
out = os.path.join(FIG, "fig_triple_combos.png")
fig.savefig(out, dpi=300, bbox_inches="tight"); print("wrote", out)
