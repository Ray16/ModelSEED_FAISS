"""Combine step-1 (acid activation) and step-2 (amine accommodation) per enzyme.

An enzyme is only a plausible adipate+HMDA catalyst if both half-reactions look
feasible: adipate positioned to be activated (step 1) AND HMDA able to reach the
activated carbonyl (step 2). This joins the two geometry runs on accession and
renders them together -- paired near-attack fractions, and a continuous
quality scatter (step-1 weighted attack angle vs step-2 weighted amine distance).

Both readouts are structural feasibility, not activity: step-2 geometry did not rank
activity on the experimental panel, so the joint view flags candidates worth testing,
not a predicted ranking.
"""
import csv
import json
from pathlib import Path

from amide_synthesis.infrastructure.paths import RES
from amide_synthesis.evaluation.step2_geometry import GENE, label

STEP1 = RES / 'cofold_step1/geometry_ranking.tsv'
STEP2 = RES / 'cofold_step2/step2_near_attack.json'
OUT = RES / 'cofold_step2'


def load():
    step1 = {r['accession']: r for r in csv.DictReader(STEP1.open(), delimiter='\t')}
    step2 = {r['accession']: r for r in json.loads(STEP2.read_text())['enzymes']}
    rows = []
    for acc, s2 in step2.items():
        s1 = step1.get(acc)   # step-2 folder now holds all 124; this view keeps the panel 13
        if s1 is None:
            continue
        rows.append({
            'accession': acc, 'gene': GENE.get(acc, acc), 'enzyme': label(acc),
            'route': s2['route'],
            'step1_near_attack': float(s1['near_attack_fraction']),
            'step1_weighted_angle_deg': float(s1['A_deg']),
            'step1_harmonic_distance_A': float(s1['Dh_A']),
            'step2_amine_reachable': s2['reachable_fraction_3p5A'],
            'step2_near_attack': s2['near_attack_fraction'],
            'step2_weighted_distance_A': s2['weighted_distance_A'],
        })
    rows.sort(key=lambda r: (-(r['step1_near_attack'] + r['step2_amine_reachable']),
                             -r['step1_weighted_angle_deg']))
    return rows


def _colour(route):
    return '#0072B2' if route == 'acyl_adenylate' else '#D55E00'


def figure(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size': 18, 'savefig.dpi': 300, 'figure.dpi': 300})

    fig, (bars, scat) = plt.subplots(1, 2, figsize=(18, 0.62 * len(rows) + 3),
                                     gridspec_kw={'width_ratios': [1, 1.15]})

    # Paired near-attack fractions per enzyme.
    order = rows[::-1]
    y = np.arange(len(order)); h = 0.4
    bars.barh(y + h / 2, [r['step1_near_attack'] for r in order], height=h,
              color='#56B4E9', label='step 1: adipate in-line activation (<4Å, >150°)')
    bars.barh(y - h / 2, [r['step2_amine_reachable'] for r in order], height=h,
              color='#E69F00', label='step 2: HMDA amine reachable (<3.5Å)')
    bars.set_yticks(y); bars.set_yticklabels([r['enzyme'] for r in order])
    bars.set_xlabel('Fraction of poses'); bars.set_xlim(0, 1)
    bars.set_title('Both half-reactions per enzyme')
    bars.legend(loc='lower right', fontsize=13)

    # Continuous joint quality: step-1 angle (ideal -> 180) vs step-2 amine distance (ideal -> small).
    for r in rows:
        scat.scatter(r['step1_weighted_angle_deg'], r['step2_weighted_distance_A'],
                     s=170, color=_colour(r['route']), edgecolor='black', lw=1.2, zorder=3)
        scat.annotate(r['gene'], (r['step1_weighted_angle_deg'], r['step2_weighted_distance_A']),
                      xytext=(7, 4), textcoords='offset points', fontsize=15)
    scat.axvline(150, color='#888888', ls='--', lw=1.3)
    scat.axhline(3.5, color='#888888', ls='--', lw=1.3)
    scat.set_xlabel('Step 1: weighted attack angle (°, ideal → 180)')
    scat.set_ylabel('Step 2: weighted amine–carbonyl distance (Å, ideal → small)')
    scat.set_title('Joint quality (best: upper-right)')
    scat.invert_yaxis()  # small amine distance (better) at top
    handles = [plt.Line2D([], [], marker='o', ls='', color='#0072B2', markersize=12),
               plt.Line2D([], [], marker='o', ls='', color='#D55E00', markersize=12)]
    scat.legend(handles, ['acyl-adenylate', 'acyl-phosphate'], loc='lower left', fontsize=15)
    fig.tight_layout(); fig.savefig(path, bbox_inches='tight'); plt.close(fig)


def main():
    rows = load()
    with (OUT / 'step1_step2_combined.tsv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader(); writer.writerows(rows)
    figure(rows, OUT / 'step1_step2_combined.png')
    print(f"Wrote combined step-1/step-2 view for {len(rows)} enzymes to {OUT}")
    for r in rows:
        print(f"  {r['gene']:8s} {r['accession']:11s} step1_near={r['step1_near_attack']:.2f} "
              f"step2_near={r['step2_near_attack']:.2f}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
