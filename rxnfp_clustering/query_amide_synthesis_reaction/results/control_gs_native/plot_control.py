"""Figure: native-substrate (L-glutamate) positive control vs the adipate target on
the GS fold. Per enzyme, distance of the attacking gamma-carboxylate oxygen to the ATP
gamma-phosphorus across 50 poses, with the 4 A near-attack cutoff. Tests whether the
step-1 attack-angle/engagement proxy transfers from ANL to the GS fold.
"""
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 18, 'savefig.dpi': 300, 'figure.dpi': 300})

RES = Path('results')
CTRL = RES / 'control_gs_native'
C_GLU, C_ADI = '#0072B2', '#D55E00'  # Okabe-Ito blue (native), vermillion (target)

spec = importlib.util.spec_from_file_location('ac', CTRL / 'analyze_control.py')
ac = importlib.util.module_from_spec(spec); spec.loader.exec_module(ac)
man = json.loads((CTRL / 'manifest.json').read_text())
LABEL = {'P78061': 'PuuA (P78061)', 'Q9KZC7': 'GlnA3 (Q9KZC7)'}

fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), sharey=True)
for ax, item in zip(axes, man['inputs']):
    acc, name = item['accession'], item['name']
    seed = CTRL / 'outputs' / 'by_seed' / name / 's42'
    poses = [ac.one_model(p, item['acid_chain']) for p in sorted(seed.glob(f'{name}_model_*.pdb'))]
    dglu = np.array([p['d_gamma'] for p in poses])
    done = json.loads((RES / 'cofold_step1' / 'state' / f'{acc}_2mg.done.json').read_text())
    dadi = np.array([p['distance_A'] for p in done['poses']])
    bins = np.linspace(2.5, 10, 24)
    ax.hist(dadi, bins=bins, color=C_ADI, alpha=0.75, label=f'adipate (target)  n<4A={int((dadi<4).sum())}')
    ax.hist(dglu, bins=bins, color=C_GLU, alpha=0.75, label=f'L-glutamate (native)  n<4A={int((dglu<4).sum())}')
    ax.axvline(4.0, color='black', ls='--', lw=2, label='4 A near-attack cutoff')
    ax.set_title(LABEL[acc], fontsize=18)
    ax.set_xlabel('gamma-carboxylate O to ATP P$\\gamma$ (A)', fontsize=18)
    ax.legend(fontsize=18, loc='upper right')
axes[0].set_ylabel('poses (of 50)', fontsize=18)
fig.tight_layout()
fig.savefig(CTRL / 'control_glutamate_vs_adipate.png', bbox_inches='tight')
print('wrote', CTRL / 'control_glutamate_vs_adipate.png')
