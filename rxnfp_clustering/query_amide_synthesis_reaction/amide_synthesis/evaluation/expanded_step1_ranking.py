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

EXP = RES / 'top10_cofold_expanded'
PANEL_GEOM = RES / 'top10_cofold/campaign/geometry_ranking.tsv'

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


def collect():
    pool = {r['accession']: r for r in csv.DictReader((EXP / 'pool.tsv').open(), delimiter='\t')}
    geom = {}
    # expanded runs
    for run in ('campaign', 'campaign_hard'):
        for p in (EXP / run / 'state').glob('*.done.json'):
            d = json.loads(p.read_text())
            geom[d['accession']] = d
    # reused references
    if PANEL_GEOM.exists():
        for r in csv.DictReader(PANEL_GEOM.open(), delimiter='\t'):
            geom.setdefault(r['accession'], {'A_deg': float(r['A_deg']),
                'near_attack_fraction': float(r['near_attack_fraction']),
                'min_distance_A': float(r['min_distance_A']), 'Dh_A': float(r['Dh_A'])})
    rows = []
    for acc, g in geom.items():
        meta = pool.get(acc, {})
        rxn = meta.get('rxn_id', '')
        organism = meta.get('organism', '')
        rows.append({
            'accession': acc,
            'label': f"{GENE.get(acc, acc)} ({org_short(organism)})",
            'gene': GENE.get(acc, ''), 'rxn_id': rxn, 'struct_family': STRUCT.get(rxn, '?'),
            'organism': organism, 'expression': 'hard' if HARD.search(organism or '') else 'ok',
            'status': meta.get('status', 'reuse'),
            'identity_to_ref': meta.get('identity', ''),
            'near_attack_fraction': round(float(g['near_attack_fraction']), 3),
            'weighted_attack_angle_deg': round(float(g['A_deg']), 1),
            'min_distance_A': round(float(g['min_distance_A']), 2),
        })
    rows.sort(key=lambda r: (-r['near_attack_fraction'], -r['weighted_attack_angle_deg']))
    return rows


def figure(rows, path, top=25):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 18, 'savefig.dpi': 300, 'figure.dpi': 300})
    top_rows = rows[:top][::-1]
    y = range(len(top_rows))
    colours = [FAM_COLOUR.get(r['struct_family'], '#888888') for r in top_rows]
    fig, ax = plt.subplots(figsize=(10, 0.46 * len(top_rows) + 2.2))
    ax.barh(list(y), [r['near_attack_fraction'] for r in top_rows], color=colours,
            edgecolor=['black' if r['expression'] == 'hard' else 'none' for r in top_rows],
            linewidth=1.6)
    ax.set_yticks(list(y))
    ax.set_yticklabels([r['label'] for r in top_rows])
    ax.set_xlabel('Step-1 near-attack fraction (activation)')
    ax.set_xlim(0, 1)
    ax.set_title(f'Top {len(top_rows)} activators of {len(rows)} folded candidates')
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAM_COLOUR.values()]
    handles.append(plt.Line2D([], [], marker='s', ls='', mfc='white', mec='black', mew=1.6,
                              markersize=13))
    ax.legend(handles, list(FAM_COLOUR) + ['hard-host (codon-opt)'], loc='lower right',
              fontsize=13)
    fig.tight_layout(); fig.savefig(path, bbox_inches='tight'); plt.close(fig)


def main():
    rows = collect()
    with (EXP / 'step1_pool_ranking.tsv').open('w', newline='') as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]), delimiter='\t')
        w.writeheader(); w.writerows(rows)
    figure(rows, EXP / 'step1_pool_ranking.png')
    hits = [r for r in rows if r['near_attack_fraction'] >= 0.3]
    print(f'folded so far: {len(rows)} candidates')
    print(f'strong activators (near-attack >= 0.30): {len(hits)}')
    for r in rows[:12]:
        print(f"  {r['near_attack_fraction']:.2f}  {r['label']:28s} {r['struct_family']:4s} "
              f"{r['status']:12s} {'HARD' if r['expression']=='hard' else ''}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
