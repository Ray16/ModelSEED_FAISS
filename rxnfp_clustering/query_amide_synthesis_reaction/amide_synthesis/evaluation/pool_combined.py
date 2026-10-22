"""Pool-wide step-1 (activation) + step-2 (accommodation) view across ALL 124 candidates.

Step-2 was folded for every candidate, not just step-1 survivors, because the two
half-reactions are independent (BacD activates adipate best yet cannot fit HMDA; PuuA
is a weak activator yet accommodates the amine). This joins:
  - step-1 geometry  : results/top10_cofold_expanded/step1_pool_ranking.tsv (all 124)
  - step-2 geometry  : computed here from BOTH step-2 runs
        panel 13    -> results/top10_cofold_step2
        expanded 111-> results/top10_cofold_expanded/step2

and renders the joint picture two ways, per the transferability caveat:
  WITHIN-FAMILY (conservative): candidates ranked by the validated step-1 activation
      proxy inside each structural family (GS vs GS, NIS vs NIS, ...), with step-2
      reachability shown alongside. This is the defensible reading.
  CROSS-FAMILY (hypothesis): a single pooled scatter (step-1 activation vs step-2
      reachability), explicitly flagged as hypothesis generation because the step-1
      proxy is calibrated only within the ANL/adenylating superfamily.

Both readouts are structural feasibility, not measured activity.
Enzymes are always labelled with gene name AND UniProt accession (homologs without a
gene name carry the accession + organism).
"""
import csv
import json
from pathlib import Path

from amide_synthesis.infrastructure.paths import RES
from amide_synthesis.evaluation.step2_geometry import GENE, summarise

EXP = RES / 'top10_cofold_expanded'
STEP1_RANK = EXP / 'step1_pool_ranking.tsv'
STEP2_PANEL = RES / 'top10_cofold_step2'
STEP2_EXPANDED = EXP / 'step2'
OUT = EXP

FAM_COLOUR = {'GS': '#0072B2', 'NIS': '#D55E00', 'Mur': '#009E73', 'BacD': '#CC79A7', '?': '#999999'}


def org_short(organism):
    parts = (organism or '').split()
    return (parts[0][0] + '.' + parts[1]) if len(parts) >= 2 else (organism or '?')[:14]


def build_label(acc, organism):
    """Always carries the UniProt accession; gene name when known, else organism."""
    gene = GENE.get(acc)
    return f'{gene} ({acc})' if gene else f'{acc} ({org_short(organism)})'


def step2_geometry():
    """reachable_fraction_3p5A, near_attack, weighted distance per accession, both runs."""
    out = {}
    for run in (STEP2_PANEL, STEP2_EXPANDED):
        manifest = json.loads((run / 'manifest.json').read_text())
        for item in manifest['inputs']:
            row, _d, _a = summarise(item, run=run)
            out[row['accession']] = row
    return out


def collect():
    step1 = {r['accession']: r for r in csv.DictReader(STEP1_RANK.open(), delimiter='\t')}
    step2 = step2_geometry()
    rows = []
    for acc, s1 in step1.items():
        s2 = step2.get(acc)
        if s2 is None:
            continue
        conclusive = s1['conclusive'] == 'True'
        rows.append({
            'accession': acc,
            'enzyme': build_label(acc, s1.get('organism', '')),
            'gene': s1.get('gene', '') or GENE.get(acc, ''),
            'struct_family': s1.get('struct_family', '?'),
            'organism': s1.get('organism', ''),
            'expression': s1.get('expression', 'ok'),
            'status': s1.get('status', ''),
            'route': s2['route'],
            # step-1 (activation): validated discriminator
            'step1_candidate_angle_deg': float(s1['candidate_angle_deg']) if s1['candidate_angle_deg'] else None,
            'step1_conclusive': conclusive,
            'step1_weighted_angle_deg': float(s1['weighted_attack_angle_deg']),
            'step1_near_attack': float(s1['near_attack_fraction']),
            'step1_n_within_4A': int(s1['n_within_4A']),
            # step-2 (accommodation): steric feasibility
            'step2_amine_reachable': s2['reachable_fraction_3p5A'],
            'step2_near_attack': s2['near_attack_fraction'],
            'step2_weighted_distance_A': s2['weighted_distance_A'],
            'step2_min_distance_A': s2['min_distance_A'],
        })
    return rows


# --------------------------------------------------------------------------- figures

def _init_mpl():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 18, 'savefig.dpi': 300, 'figure.dpi': 300})
    return plt


def within_family_figure(rows, path, per_family=8):
    """Conservative view: within each family, top step-1 activators (conclusive), with
    step-2 reachability as a paired bar. Families in separate blocks; never a shared
    leaderboard (proxy calibrated within ANL only)."""
    plt = _init_mpl()
    import numpy as np

    blocks, labels, seps = [], [], []
    for fam in ('GS', 'NIS', 'BacD', 'Mur'):
        fam_rows = [r for r in rows if r['struct_family'] == fam and r['step1_conclusive']]
        fam_rows = sorted(fam_rows, key=lambda r: r['step1_candidate_angle_deg'])[-per_family:]
        if not fam_rows:
            continue
        seps.append(len(labels) - 0.5)
        for r in fam_rows:
            blocks.append(r)
            labels.append(f"[{fam}] {r['enzyme']}")

    y = np.arange(len(labels)); h = 0.4
    fig, ax = plt.subplots(figsize=(14, 0.62 * len(labels) + 3.2))
    # step-1 activation angle rescaled to [0,1] over [90,180] so both bars share an axis.
    s1_frac = [(r['step1_candidate_angle_deg'] - 90) / 90 for r in blocks]
    s2_frac = [r['step2_amine_reachable'] for r in blocks]
    ax.barh(y + h / 2, s1_frac, height=h, color='#56B4E9',
            label='Step 1: adipate activation angle (90→180° ⇒ 0→1)')
    ax.barh(y - h / 2, s2_frac, height=h, color='#E69F00',
            label='Step 2: HMDA amine reachable (<3.5 Å) fraction')
    ax.set_yticks(y); ax.set_yticklabels(labels)
    for s in seps[1:]:
        ax.axhline(s, color='#bbbbbb', lw=1)
    for i, r in enumerate(blocks):
        ax.text(s1_frac[i] + 0.012, y[i] + h / 2, f"{r['step1_candidate_angle_deg']:.0f}°",
                va='center', fontsize=18)
    ax.set_xlim(0, 1.15); ax.set_xlabel('Normalised score (see legend)')
    ax.set_title('Within-family: step-1 activation (validated) + step-2 accommodation')
    # Legend outside the axes (below) so it never covers a bar.
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.045), ncol=1,
              fontsize=18, frameon=False)
    fig.tight_layout(); fig.savefig(path, bbox_inches='tight'); plt.close(fig)


def cross_family_figure(rows, path):
    """Hypothesis view: pooled scatter of step-1 activation vs step-2 accommodation,
    coloured by family. Flagged hypothesis because the step-1 proxy is not calibrated
    across superfamilies."""
    plt = _init_mpl()

    conc = [r for r in rows if r['step1_conclusive']]
    fig, ax = plt.subplots(figsize=(12.5, 10))
    # Upper-right quadrant = both half-reactions feasible.
    ax.axvline(150, color='#888888', ls='--', lw=1.3)
    ax.axhline(0.3, color='#888888', ls='--', lw=1.3)
    for r in conc:
        col = FAM_COLOUR.get(r['struct_family'], '#999999')
        ax.scatter(r['step1_candidate_angle_deg'], r['step2_amine_reachable'],
                   s=210, color=col, edgecolor=('black' if r['expression'] == 'hard' else 'white'),
                   lw=1.6, zorder=3)
    # Label only the strong corner (both feasible) to avoid clutter.
    for r in conc:
        if r['step1_candidate_angle_deg'] >= 150 and r['step2_amine_reachable'] >= 0.3:
            ax.annotate(r['enzyme'], (r['step1_candidate_angle_deg'], r['step2_amine_reachable']),
                        xytext=(7, 5), textcoords='offset points', fontsize=18)
    ax.set_xlabel('Step 1: activation angle among engaged poses (°)')
    ax.set_ylabel('Step 2: HMDA amine reachable (<3.5 Å) fraction')
    ax.set_title('Cross-family joint feasibility (HYPOTHESIS; top-right = both feasible)')
    handles = [plt.Line2D([], [], marker='o', ls='', color=FAM_COLOUR[f], markersize=14)
               for f in ('GS', 'NIS', 'BacD', 'Mur')]
    handles.append(plt.Line2D([], [], marker='o', ls='', mfc='#999999', mec='black', mew=1.6,
                              markersize=14))
    ax.legend(handles, ['GS', 'NIS', 'BacD', 'Mur', 'hard-host (codon-opt)'],
              loc='upper center', bbox_to_anchor=(0.5, -0.09), ncol=5, fontsize=18,
              frameon=False, handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout(); fig.savefig(path, bbox_inches='tight'); plt.close(fig)


def main():
    rows = collect()
    rows.sort(key=lambda r: (not r['step1_conclusive'],
                             -(r['step1_candidate_angle_deg'] or 0)))
    fields = list(rows[0])
    with (OUT / 'pool_combined.tsv').open('w', newline='') as h:
        w = csv.DictWriter(h, fieldnames=fields, delimiter='\t')
        w.writeheader(); w.writerows(rows)
    within_family_figure(rows, OUT / 'pool_combined_within_family.png')
    cross_family_figure(rows, OUT / 'pool_combined_cross_family.png')

    conc = [r for r in rows if r['step1_conclusive']]
    print(f'combined: {len(rows)} candidates; step-1 conclusive: {len(conc)}')
    print(f'wrote {OUT / "pool_combined.tsv"}')
    print(f'      {OUT / "pool_combined_within_family.png"}')
    print(f'      {OUT / "pool_combined_cross_family.png"}')
    import collections
    for fam in ('GS', 'NIS', 'BacD', 'Mur'):
        fr = sorted([r for r in conc if r['struct_family'] == fam],
                    key=lambda r: -r['step1_candidate_angle_deg'])[:5]
        if fr:
            print(f'\n-- {fam} within-family top (step-1 activation; step-2 reach) --')
            for r in fr:
                print(f"   s1_angle={r['step1_candidate_angle_deg']:5.1f}  "
                      f"s2_reach={r['step2_amine_reachable']:.2f}  "
                      f"{r['enzyme']:34s} {r['status']:12s} "
                      f"{'HARD' if r['expression']=='hard' else ''}")
    # Both-feasible corner (hypothesis cross-family readout)
    both = [r for r in conc if r['step1_candidate_angle_deg'] >= 150 and r['step2_amine_reachable'] >= 0.3]
    print(f'\nboth-feasible corner (step1>=150 & step2 reach>=0.3): {len(both)}')
    for r in sorted(both, key=lambda r: -(r['step1_candidate_angle_deg'])):
        print(f"   {r['enzyme']:34s} [{r['struct_family']}] "
              f"s1={r['step1_candidate_angle_deg']:.0f}  s2={r['step2_amine_reachable']:.2f}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
