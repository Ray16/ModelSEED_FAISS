#!/usr/bin/env python
"""Two figures for the slide deck.

fig_families.png     the shortlist collapsed to protein families, which is how
                     the result should be read: one family found sixteen times
                     is a convergence, not sixteen independent hits.
fig_substrate_gap.png the chemical point of the whole project - the native
                     substrate of the top enzymes is the same STRUCTURAL CLASS
                     as the nylon-6,6 diad, differing in chain length and an
                     N-hydroxy group. Drawn, because the argument is structural.

Project rule: every text element >= 18 pt, dpi 300, no gray explanatory
captions baked in.
"""
from __future__ import annotations

import sys

import pandas as pd

from .paths import ROOT, FIG, RES
sys.path.insert(0, str(ROOT / '_lib'))
import _style  # noqa: F401,E402   18 pt / dpi 300
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from rdkit import Chem
from rdkit.Chem import Draw, AllChem


NIS = '#D55E00'
OTHER = '#0072B2'
INK = '#252C34'

from .families import SECONDARY_DOMAINS


def families_figure():
    families = pd.read_csv(RES / 'enzyme_families.tsv', sep='\t')
    families = families.dropna(subset=['family'])
    families = families[~families.family.isin(SECONDARY_DOMAINS)]
    families = families[families.n_enzymes >= 2].head(8).iloc[::-1]
    if families.empty:
        (FIG / 'fig_families.png').unlink(missing_ok=True)
        print('No families to plot', flush=True)
        return

    fig, ax = plt.subplots(figsize=(15, 8.5))
    colors = [NIS if 'IucA' in f else OTHER for f in families.family]
    bars = ax.barh(range(len(families)), families.n_enzymes, color=colors, height=0.68)
    ax.set_yticks(range(len(families)))
    ax.set_yticklabels([f if len(f) < 44 else f[:41] + '...' for f in families.family])
    ax.set_xlabel('Candidate enzymes in family', labelpad=12)
    ax.set_xlim(0, families.n_enzymes.max() * 1.28)
    for bar, cosine in zip(bars, families.best_cosine):
        ax.text(bar.get_width() + families.n_enzymes.max() * 0.025,
                bar.get_y() + bar.get_height() / 2,
                f'best cosine {cosine:.2f}', va='center', fontsize=18, color=INK)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', color='#DDE2E7', lw=0.8)
    ax.set_axisbelow(True)
    ax.legend([Patch(facecolor=NIS), Patch(facecolor=OTHER)],
              ['NIS synthetase (IucA/IucC)', 'Other families'],
              loc='lower right', frameon=False, fontsize=18)
    for text in fig.findobj(match=plt.Text):
        if text.get_text():
            assert text.get_fontsize() >= 18
    fig.savefig(FIG / 'fig_families.png', dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    print('Saved fig_families.png', flush=True)


def substrate_gap_figure():
    """Native monomer vs the nylon-6,6 diad: same class, different dimensions."""
    # Short two-line titles: the full IUPAC names are wide enough to collide
    # between panels, and the comparison is about chain length, not nomenclature.
    panels = [
        ('BibC native monomer\nsuccinyl-N-hydroxy-cadaverine  (C4+C5)',
         'OC(=O)CCC(=O)N(O)CCCCCN'),
        ('PubC native monomer\nsuccinyl-N-hydroxy-putrescine  (C4+C4)',
         'OC(=O)CCC(=O)N(O)CCCCN'),
        ('Nylon-6,6 diad (this work)\nadipoyl-hexamethylenediamine  (C6+C6)',
         'OC(=O)CCCCC(=O)NCCCCCCN'),
    ]
    mols, legends = [], []
    for legend, smiles in panels:
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, smiles
        AllChem.Compute2DCoords(mol)
        mols.append(mol)
        legends.append(legend)

    image = Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(620, 330),
                                 legends=['', '', ''], returnPNG=False)
    raw = FIG / '_substrate_structures.png'
    image.save(raw)

    fig, axes = plt.subplots(1, 3, figsize=(22, 6.6))
    fig.subplots_adjust(wspace=0.08)
    for ax, (legend, smiles) in zip(axes, panels):
        mol = Chem.MolFromSmiles(smiles)
        AllChem.Compute2DCoords(mol)
        panel = Draw.MolToImage(mol, size=(620, 330))
        ax.imshow(panel)
        ax.axis('off')
        title_color = NIS if 'Nylon' in legend else INK
        ax.set_title(legend, fontsize=18, color=title_color, pad=16,
                     linespacing=1.5,
                     fontweight='semibold' if 'Nylon' in legend else 'normal')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [ax.title.get_window_extent(renderer) for ax in axes]
    for left, right in zip(boxes, boxes[1:]):
        assert left.x1 < right.x0, 'panel titles overlap'
    for text in fig.findobj(match=plt.Text):
        if text.get_text():
            assert text.get_fontsize() >= 18
    fig.savefig(FIG / 'fig_substrate_gap.png', dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    raw.unlink(missing_ok=True)
    print('Saved fig_substrate_gap.png', flush=True)


from .freshness import tracked_stage

@tracked_stage('deckfigs')
def main():
    from .paths import ensure_directories
    ensure_directories()
    families_figure()
    substrate_gap_figure()


if __name__ == '__main__':
    main()
