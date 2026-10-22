"""Pool-wide step-1 activation ranking across the growing candidate pool.

Aggregates step-1 geometry (in-line attack angle A_deg and near-attack fraction,
the validated activation discriminator) for every folded ensemble in the pool:
the expanded runs (campaign/ expressible, campaign_hard/ hard-host) plus the reused
reference folds (from the panel geometry_ranking). Partial-safe: includes whatever
is done so far, so it can be rolled out periodically while the fold proceeds.

Enzymes are labelled "Gene/accession (organism)"; references carry gene names,
homologs carry accession + organism. Hard-host candidates are flagged (expressibility
is an annotation, not a filter).
"""
import csv
import json
import re
from pathlib import Path

from amide_synthesis.infrastructure.paths import RES

EXP = RES / 'top10_cofold_expanded'            # analysis home: pool.tsv, output ranking tsv/png
PANEL_GEOM = RES / 'cofold_step1/geometry_ranking.tsv'

GENE = {'P78061': 'PuuA', 'P39641': 'BacD', 'Q9KZC7': 'GlnA3', 'Q9RDS6': 'GlnA2sc',
        'I3R584': 'GlnA2hf', 'Q9WY79': 'MurE', 'E2JA32': 'DdaG', 'Q2FW70': 'SfnaD',
        'Q2G1M9': 'SbnE', 'A0A0F7RJ52': 'AsbA', 'Q2G1N1': 'SbnC', 'Q2G1M8': 'SbnF',
        'Q81RQ8': 'AsbB'}
STRUCT = {'rxn05123': 'GS', 'rxn22343': 'Mur', 'rxn09897': 'BacD',
          **{r: 'NIS' for r in ('rxn46045', 'rxn42510', 'rxn47618', 'rxn21038',
                                 'rxn42395', 'rxn45007', 'rxn21036')}}
FAM_COLOUR = {'GS': '#0072B2', 'NIS': '#D55E00', 'Mur': '#009E73', 'BacD': '#CC79A7'}
HARD = re.compile(r'Streptomyces|Mycobacter|Corynebact|Nocardia|Rhodococc|Actinomyc|'
                  r'Frankia|Bifidobact|Halo|Natrono|Archae|Thermo|Pyro|Sulfolob|Methano|'
                  r'Geobacillus|Deinococ', re.I)


def org_short(organism):
    parts = (organism or '').split()
    return (parts[0][0] + '.' + parts[1]) if len(parts) >= 2 else (organism or '?')[:12]


import numpy as np

# All step-1 ensembles live in one folder now; done.json carry per-pose geometry.
POSE_RUNS = [RES / 'cofold_step1']


def pose_metrics(poses):
    d = np.array([p['distance_A'] for p in poses], float)
    a = np.array([p['angle_deg'] for p in poses], float)
    inv = 1 / d
    within = d < 4.0
    return {
        'weighted_attack_angle_deg': round(float((a * inv).sum() / inv.sum()), 1),  # A
        'near_attack_fraction': round(float((within & (a > 150)).mean()), 3),
        'min_distance_A': round(float(d.min()), 2),
        'n_within_4A': int(within.sum()),
        # Manuscript candidate workflow: mean in-line angle among poses within 4 A,
        # reported only when >= 5 are retained; otherwise inconclusive.
        'mean_angle_within_4A': round(float(a[within].mean()), 1) if within.sum() else 0.0,
    }


def collect():
    pool = {r['accession']: r for r in csv.DictReader((EXP / 'pool.tsv').open(), delimiter='\t')}
    geom = {}
    for run in POSE_RUNS:
        for p in (run / 'state').glob('*.done.json'):
            d = json.loads(p.read_text())
            if 'poses' in d:
                geom.setdefault(d['accession'], pose_metrics(d['poses']))
    rows = []
    for acc, g in geom.items():
        meta = pool.get(acc, {})
        rxn = meta.get('rxn_id', '')
        organism = meta.get('organism', '')
        conclusive = g['n_within_4A'] >= 5
        rows.append({
            'accession': acc,
            'label': f"{GENE.get(acc, acc)} ({org_short(organism)})",
            'gene': GENE.get(acc, ''), 'rxn_id': rxn, 'struct_family': STRUCT.get(rxn, '?'),
            'organism': organism, 'expression': 'hard' if HARD.search(organism or '') else 'ok',
            'status': meta.get('status', 'reuse'), 'identity_to_ref': meta.get('identity', ''),
            'candidate_angle_deg': g['mean_angle_within_4A'] if conclusive else '',
            'conclusive': conclusive,
            'weighted_attack_angle_deg': g['weighted_attack_angle_deg'],
            'near_attack_fraction': g['near_attack_fraction'],
            'n_within_4A': g['n_within_4A'], 'min_distance_A': g['min_distance_A'],
        })
    # Surface by the candidate-workflow angle (conclusive first, highest mean in-line
    # angle among engaged poses). A and near-attack kept as context columns.
    rows.sort(key=lambda r: (not r['conclusive'],
                             -(r['candidate_angle_deg'] if r['conclusive'] else -1)))
    return rows


def figure(rows, path, per_family=10):
    """Family-stratified: rank conclusive candidates WITHIN each structural family.

    The proxy was validated within one superfamily (ANL/SfaB); cross-family
    comparison is not calibrated, so families are shown in separate blocks and never
    on a shared leaderboard.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 18, 'savefig.dpi': 300, 'figure.dpi': 300})

    labels, values, colours, edges, sep = [], [], [], [], []
    for fam in ('GS', 'NIS', 'BacD', 'Mur'):
        fam_rows = [r for r in rows if r['struct_family'] == fam and r['conclusive']]
        fam_rows = sorted(fam_rows, key=lambda r: r['candidate_angle_deg'])[-per_family:]
        if not fam_rows:
            continue
        sep.append(len(labels) - 0.5)
        for r in fam_rows:
            labels.append(f"[{fam}] {r['label']}")
            values.append(r['candidate_angle_deg'])
            colours.append(FAM_COLOUR[fam])
            edges.append('black' if r['expression'] == 'hard' else 'none')
    fig, ax = plt.subplots(figsize=(10.5, 0.46 * len(labels) + 2.2))
    ax.barh(range(len(labels)), values, color=colours, edgecolor=edges, linewidth=1.6)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    for s in sep[1:]:
        ax.axhline(s, color='#cccccc', lw=1)
    ax.set_xlabel('Mean in-line attack angle among engaged poses (<4Å) (°)')
    ax.set_xlim(90, 180)
    ax.set_title('Step-1 candidates ranked WITHIN family\n(proxy validated in ANL only; '
                 'cross-family not comparable)', fontsize=15)
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAM_COLOUR[f]) for f in ('GS', 'NIS', 'BacD', 'Mur')]
    handles.append(plt.Line2D([], [], marker='s', ls='', mfc='white', mec='black', mew=1.6, markersize=13))
    ax.legend(handles, ['GS', 'NIS', 'BacD', 'Mur', 'hard-host (codon-opt)'],
              loc='lower right', fontsize=12)
    fig.tight_layout(); fig.savefig(path, bbox_inches='tight'); plt.close(fig)


def main():
    rows = collect()
    with (EXP / 'step1_pool_ranking.tsv').open('w', newline='') as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]), delimiter='\t')
        w.writeheader(); w.writerows(rows)
    figure(rows, EXP / 'step1_pool_ranking.png')
    conc = [r for r in rows if r['conclusive']]
    import collections
    print(f'folded: {len(rows)} candidates; conclusive (>=5 poses <4A): {len(conc)}')
    print('conclusive by family:', dict(collections.Counter(r['struct_family'] for r in conc)))
    for fam in ('GS', 'NIS', 'BacD', 'Mur'):
        fr = sorted([r for r in conc if r['struct_family'] == fam],
                    key=lambda r: -r['candidate_angle_deg'])[:5]
        if fr:
            print(f'  -- {fam} (within-family top) --')
            for r in fr:
                print(f"    angle={r['candidate_angle_deg']:5.1f}  na={r['near_attack_fraction']:.2f}  "
                      f"{r['label']:30s} {r['status']:12s} {'HARD' if r['expression']=='hard' else ''}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
