"""Figure: does the native amine (putrescine) rescue glutamate engagement on the GS fold?
Per enzyme, pooled 3-seed distribution of gamma-carboxylate O -> ATP Pgamma distance,
no-amine vs +putrescine, with the 4 A near-attack cutoff.
"""
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('am', HERE / 'analyze_matrix.py')
am = importlib.util.module_from_spec(spec); spec.loader.exec_module(am)

plt.rcParams.update({'font.size': 18, 'savefig.dpi': 300, 'figure.dpi': 300})
C_FREE, C_PUT = '#999999', '#0072B2'  # grey = no amine, blue = +putrescine
LABEL = {'P78061': 'PuuA (P78061)', 'Q9KZC7': 'GlnA3 (Q9KZC7)'}

fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), sharey=True)
for ax, acc in zip(axes, ('P78061', 'Q9KZC7')):
    dfree = np.array([p['d_gamma'] for s in am.SEEDS for p in am.poses_for(f'{acc}_free', s)])
    dput = np.array([p['d_gamma'] for s in am.SEEDS for p in am.poses_for(f'{acc}_put', s)])
    bins = np.linspace(2.5, 9, 24)
    ax.hist(dfree, bins=bins, color=C_FREE, alpha=0.8,
            label=f'no amine        n<4A={int((dfree<4).sum())}/150')
    ax.hist(dput, bins=bins, color=C_PUT, alpha=0.75,
            label=f'+ putrescine   n<4A={int((dput<4).sum())}/150')
    ax.axvline(4.0, color='black', ls='--', lw=2, label='4 A near-attack cutoff')
    ax.set_title(LABEL[acc], fontsize=18)
    ax.set_xlabel('gamma-carboxylate O to ATP P$\\gamma$ (A)', fontsize=18)
    ax.legend(fontsize=18, loc='upper right')
axes[0].set_ylabel('poses (of 150)', fontsize=18)
fig.tight_layout()
fig.savefig(HERE / 'matrix_amine_rescue.png', bbox_inches='tight')
print('wrote', HERE / 'matrix_amine_rescue.png')
