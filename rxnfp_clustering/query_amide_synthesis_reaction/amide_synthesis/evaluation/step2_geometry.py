"""Step-2 near-attack geometry: can the C6 diamine reach the acyl carbon?

For each step-2 cofold (acyl intermediate + HMDA + Mg, +ADP on the acyl-phosphate
route) this measures whether an HMDA amine is positioned to attack the activated
carbonyl: the amine-N -> carbonyl-C distance and the Burgi-Dunitz angle (N...C=O).
These describe accommodation of the diamine, not reaction rate -- step-2 geometry
was non-predictive of activity on the experimental panel, where step-1 adenylation
geometry carried the signal. The readout answers the size/shape question the user
posed: native amine substrates here range from ammonia-sized to large polyamines,
so the point is whether HMDA specifically fits for near-attack.

Atoms are identified topologically (Boltz names ligand atoms by SMILES order):
  acyl carbonyl C = carbon bonded to a phosphorus-bridging ester oxygen and carrying
                    at least two oxygen neighbours (excludes the ribose C5' and the
                    distal free carboxylate carbon);
  carbonyl O      = that carbon's oxygen neighbour not bonded to phosphorus;
  nucleophile N   = the HMDA nitrogen closest to the acyl carbon (whichever amine is
                    presented); HMDA is the ligand chain with 6 C and 2 N.
"""
import csv
import json
import math
from pathlib import Path

import gemmi
import numpy as np

from amide_synthesis.infrastructure.paths import RES

RUN = RES / 'top10_cofold_step2'
BD_LOW, BD_HIGH, NEAR_A = 90.0, 120.0, 3.5

# Enzyme/gene name per UniProt accession; always label as "Gene (Accession)".
GENE = {'P78061': 'PuuA', 'P39641': 'BacD', 'Q9KZC7': 'GlnA3', 'Q9RDS6': 'GlnA2sc',
        'I3R584': 'GlnA2hf', 'Q9WY79': 'MurE', 'E2JA32': 'DdaG', 'Q2FW70': 'SfnaD',
        'Q2G1M9': 'SbnE', 'A0A0F7RJ52': 'AsbA', 'Q2G1N1': 'SbnC', 'Q2G1M8': 'SbnF',
        'Q81RQ8': 'AsbB'}


def label(accession):
    return f'{GENE.get(accession, accession)} ({accession})'


def bonded(a, b, cutoff):
    return a.pos.dist(b.pos) <= cutoff


def reactive_atoms(model):
    """Return (acyl carbon, carbonyl oxygen, [HMDA nitrogens]) by topology."""
    hmda_n = None
    acyl_c = carbonyl_o = None
    for chain in model:
        heavy = [a for r in chain for a in r if a.element.name != 'H']
        c = sum(a.element.name == 'C' for a in heavy)
        n = sum(a.element.name == 'N' for a in heavy)
        if len(heavy) == 8 and c == 6 and n == 2:                     # HMDA
            hmda_n = [a for a in heavy if a.element.name == 'N']
        if any(a.element.name == 'P' for a in heavy) and c == 6 and n == 0:
            acyl_c, carbonyl_o = _acyl_site(heavy)                    # acyl-phosphate (C6,P,no N)
        elif any(a.element.name == 'P' for a in heavy) and c == 16:
            acyl_c, carbonyl_o = _acyl_site(heavy)                    # acyl-AMP
    if hmda_n is None or acyl_c is None:
        raise ValueError('Could not locate HMDA or acyl carbonyl')
    return acyl_c, carbonyl_o, hmda_n


def _acyl_site(heavy):
    phosphorus = [a for a in heavy if a.element.name == 'P']
    oxygens = [a for a in heavy if a.element.name == 'O']
    carbons = [a for a in heavy if a.element.name == 'C']
    bridging = [o for o in oxygens if any(bonded(o, p, 1.9) for p in phosphorus)]
    for carbon in carbons:
        o_neighbours = [o for o in oxygens if bonded(o, carbon, 1.8)]
        if len(o_neighbours) >= 2 and any(bonded(carbon, o, 1.8) for o in bridging):
            terminal = [o for o in o_neighbours
                        if not any(bonded(o, p, 1.9) for p in phosphorus)]
            if terminal:
                return carbon, terminal[0]
    raise ValueError('Acyl carbonyl carbon not found')


def pose_geometry(model):
    acyl_c, carbonyl_o, nitrogens = reactive_atoms(model)
    nitrogen = min(nitrogens, key=lambda a: a.pos.dist(acyl_c.pos))
    distance = nitrogen.pos.dist(acyl_c.pos)
    u = np.array([nitrogen.pos.x - acyl_c.pos.x, nitrogen.pos.y - acyl_c.pos.y,
                  nitrogen.pos.z - acyl_c.pos.z])
    v = np.array([carbonyl_o.pos.x - acyl_c.pos.x, carbonyl_o.pos.y - acyl_c.pos.y,
                  carbonyl_o.pos.z - acyl_c.pos.z])
    angle = math.degrees(math.acos(float(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)),
                                                  -1, 1))))
    return distance, angle


def summarise(item, run=RUN):
    name = item['name']
    out = run / 'outputs' / name / 'by_seed' / name / 's42'
    distances, angles = [], []
    for pdb in sorted(out.glob(f'{name}_model_*.pdb')):
        d, a = pose_geometry(gemmi.read_structure(str(pdb))[0])
        distances.append(d); angles.append(a)
    d = np.array(distances); a = np.array(angles)
    if d.shape != (50,):
        raise ValueError(f'{name}: expected 50 poses, got {d.shape[0]}')
    inv = 1 / d
    near = (d < NEAR_A) & (a >= BD_LOW) & (a <= BD_HIGH)
    row = {
        'name': name, 'accession': item['accession'],
        'enzyme': label(item['accession']), 'route': item['route'],
        'weighted_distance_A': float((d * inv).sum() / inv.sum()),
        'harmonic_distance_A': float(50 / inv.sum()),
        'min_distance_A': float(d.min()), 'median_distance_A': float(np.median(d)),
        'mean_BurgiDunitz_deg': float(a.mean()),
        # Reachability: an HMDA amine within 3.5 A of the acyl carbon, angle-free.
        # This is the robust accommodation signal; the Burgi-Dunitz gate below is stricter
        # and fragile on static cofold poses, so it is reported as secondary.
        'reachable_fraction_3p5A': float((d < NEAR_A).mean()),
        'near_attack_fraction': float(near.mean()),
        'n_models': 50,
    }
    return row, d, a


def main():
    manifest = json.loads((RUN / 'manifest.json').read_text())
    rows, poses = [], {}
    for item in manifest['inputs']:
        row, d, a = summarise(item)
        rows.append(row); poses[row['name']] = (d, a)
    rows.sort(key=lambda r: (-r['reachable_fraction_3p5A'], -r['near_attack_fraction'],
                             r['min_distance_A']))
    with (RUN / 'step2_near_attack.tsv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader(); writer.writerows(rows)
    summary = {
        'n_enzymes': len(rows),
        'enzymes_amine_reachable': sum(r['reachable_fraction_3p5A'] > 0 for r in rows),
        'enzymes_with_any_near_attack': sum(r['near_attack_fraction'] > 0 for r in rows),
        'median_min_distance_A': float(np.median([r['min_distance_A'] for r in rows])),
        'interpretation': (
            'Primary accommodation signal = reachable_fraction_3p5A: poses with an HMDA '
            'amine-N within 3.5 A of the acyl carbonyl-C (distance only). near_attack_fraction '
            'adds a Burgi-Dunitz angle gate [90,120] deg, which is stricter and fragile on '
            'static cofold poses. Both are steric feasibility for the C6 diamine, not catalytic '
            'rate; step-2 geometry did not rank activity on the experimental panel (step-1 did). '
            'Output lacks hydrogens, so the closest amine-N is used (neutral vs protonated amine '
            'not distinguishable geometrically).'),
        'enzymes': rows,
    }
    (RUN / 'step2_near_attack.json').write_text(json.dumps(summary, indent=2))
    figure(rows, poses, RUN / 'step2_near_attack.png')
    facet_figure(rows, poses, RUN / 'step2_pose_clouds.png')
    print(json.dumps(summary, indent=2))
    return 0


def _route_colour(route):
    return '#0072B2' if route == 'acyl_adenylate' else '#D55E00'


def figure(rows, poses, path):
    """Two panels: per-enzyme near-attack summary, and the full pose distribution."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size': 18, 'savefig.dpi': 300, 'figure.dpi': 300})

    fig, (bar, scat) = plt.subplots(1, 2, figsize=(17, 0.52 * len(rows) + 2.5),
                                    gridspec_kw={'width_ratios': [1, 1.25]})

    order = sorted(rows, key=lambda r: r['reachable_fraction_3p5A'])
    labels = [r['enzyme'] for r in order]
    bar.barh(range(len(order)), [r['reachable_fraction_3p5A'] for r in order],
             color=[_route_colour(r['route']) for r in order])
    bar.set_yticks(range(len(order))); bar.set_yticklabels(labels)
    bar.set_xlabel('HMDA amine reachable (<3.5 Å) fraction'); bar.set_xlim(0, 1)
    bar.set_title('Fraction of poses with amine in reach')

    # Near-attack region: d < 3.5 A and Burgi-Dunitz angle in [90, 120] deg.
    scat.axhspan(BD_LOW, BD_HIGH, xmax=(NEAR_A - 2.5) / (9 - 2.5), color='#CCCCCC', alpha=0.5,
                 zorder=0)
    scat.axvline(NEAR_A, color='#888888', ls='--', lw=1.5, zorder=1)
    for r in rows:
        d, a = poses[r['name']]
        scat.scatter(d, a, s=26, alpha=0.5, color=_route_colour(r['route']),
                     edgecolor='none', zorder=2)
    for r in rows:  # weighted summary marker per enzyme
        scat.scatter(r['weighted_distance_A'], r['mean_BurgiDunitz_deg'], s=150, marker='D',
                     color=_route_colour(r['route']), edgecolor='black', lw=1.2, zorder=3)
    scat.set_xlabel('Amine N - acyl C distance (Å)')
    scat.set_ylabel('Bürgi–Dunitz angle N···C=O (°)')
    scat.set_xlim(2.5, 9); scat.set_ylim(0, 180)
    scat.set_title('All poses (●) and weighted summary (◆)')
    handles = [plt.Line2D([], [], marker='o', ls='', color='#0072B2'),
               plt.Line2D([], [], marker='o', ls='', color='#D55E00'),
               plt.Rectangle((0, 0), 1, 1, color='#CCCCCC', alpha=0.7)]
    scat.legend(handles, ['acyl-adenylate', 'acyl-phosphate', 'near-attack region'],
                loc='upper right', fontsize=14)
    fig.tight_layout(); fig.savefig(path, bbox_inches='tight'); plt.close(fig)


def facet_figure(rows, poses, path):
    """Small-multiples: each enzyme's 50-pose cloud in the distance-angle plane."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 18, 'savefig.dpi': 300, 'figure.dpi': 300})
    ncol = 4
    nrow = -(-len(rows) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.6 * nrow),
                             sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, r in zip(axes, rows):
        d, a = poses[r['name']]
        ax.axhspan(BD_LOW, BD_HIGH, xmax=(NEAR_A - 2.5) / (9 - 2.5), color='#CCCCCC', alpha=0.5)
        ax.axvline(NEAR_A, color='#888888', ls='--', lw=1.2)
        ax.scatter(d, a, s=30, alpha=0.6, color=_route_colour(r['route']), edgecolor='none')
        ax.set_title(f"{r['enzyme']}  reach {r['reachable_fraction_3p5A']*100:.0f}%",
                     fontsize=16)
        ax.set_xlim(2.5, 9); ax.set_ylim(0, 180)
    for ax in axes[len(rows):]:
        ax.axis('off')
    fig.supxlabel('Amine N - acyl C distance (Å)', fontsize=18)
    fig.supylabel('Bürgi–Dunitz angle (°)', fontsize=18)
    fig.tight_layout(); fig.savefig(path, bbox_inches='tight'); plt.close(fig)


if __name__ == '__main__':
    raise SystemExit(main())
