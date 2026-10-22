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


def summarise(item):
    name = item['name']
    out = RUN / 'outputs' / name / 'by_seed' / name / 's42'
    distances, angles = [], []
    for pdb in sorted(out.glob(f'{name}_model_*.pdb')):
        d, a = pose_geometry(gemmi.read_structure(str(pdb))[0])
        distances.append(d); angles.append(a)
    d = np.array(distances); a = np.array(angles)
    if d.shape != (50,):
        raise ValueError(f'{name}: expected 50 poses, got {d.shape[0]}')
    inv = 1 / d
    near = (d < NEAR_A) & (a >= BD_LOW) & (a <= BD_HIGH)
    return {
        'name': name, 'accession': item['accession'], 'route': item['route'],
        'weighted_distance_A': float((d * inv).sum() / inv.sum()),
        'harmonic_distance_A': float(50 / inv.sum()),
        'min_distance_A': float(d.min()), 'median_distance_A': float(np.median(d)),
        'mean_BurgiDunitz_deg': float(a.mean()),
        'near_attack_fraction': float(near.mean()),
        'n_models': 50,
    }


def main():
    manifest = json.loads((RUN / 'manifest.json').read_text())
    rows = [summarise(item) for item in manifest['inputs']]
    rows.sort(key=lambda r: (-r['near_attack_fraction'], r['min_distance_A']))
    with (RUN / 'step2_near_attack.tsv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader(); writer.writerows(rows)
    summary = {
        'n_enzymes': len(rows),
        'enzymes_with_any_near_attack': sum(r['near_attack_fraction'] > 0 for r in rows),
        'median_min_distance_A': float(np.median([r['min_distance_A'] for r in rows])),
        'interpretation': (
            'Near-attack fraction = poses with amine-N to carbonyl-C < 3.5 A and '
            'Burgi-Dunitz angle in [90,120] deg. Measures whether HMDA is sterically '
            'accommodated for attack, not catalytic rate; step-2 geometry did not rank '
            'activity on the experimental panel (step-1 adenylation did).'),
        'enzymes': rows,
    }
    (RUN / 'step2_near_attack.json').write_text(json.dumps(summary, indent=2))
    figure(rows, RUN / 'step2_near_attack.png')
    print(json.dumps(summary, indent=2))
    return 0


def figure(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 18, 'savefig.dpi': 300, 'figure.dpi': 300})
    order = sorted(rows, key=lambda r: r['near_attack_fraction'])
    labels = [f"{r['accession']} ({'AMP' if r['route'] == 'acyl_adenylate' else 'P'})" for r in order]
    values = [r['near_attack_fraction'] for r in order]
    colours = ['#0072B2' if r['route'] == 'acyl_adenylate' else '#D55E00' for r in order]
    fig, ax = plt.subplots(figsize=(8.5, 0.55 * len(order) + 2))
    ax.barh(range(len(order)), values, color=colours)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(labels)
    ax.set_xlabel('HMDA near-attack fraction'); ax.set_xlim(0, 1)
    handles = [plt.Rectangle((0, 0), 1, 1, color='#0072B2'),
               plt.Rectangle((0, 0), 1, 1, color='#D55E00')]
    ax.legend(handles, ['acyl-adenylate', 'acyl-phosphate'], loc='lower right')
    fig.tight_layout(); fig.savefig(path, bbox_inches='tight'); plt.close(fig)


if __name__ == '__main__':
    raise SystemExit(main())
