"""Analyze the GS native-control matrix: enzyme x (+/- putrescine) x seeds 42/1/7.

Reuses the glutamate geometry from analyze_control.py (gamma-carboxylate O -> Pgamma ->
O3B angle, <4A engagement). Pools the 3 seeds (150 poses/condition) and also reports the
per-seed engagement count so seed variance is visible. Question: does adding the native
amine (putrescine) rescue glutamate engagement, and is the free-amine deficit seed-robust?
"""
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE.parent.parent  # results/
spec = importlib.util.spec_from_file_location('ac', HERE.parent / 'analyze_control.py')
ac = importlib.util.module_from_spec(spec); spec.loader.exec_module(ac)

SEEDS = [42, 1, 7]
LABEL = {'P78061': 'PuuA (P78061)', 'Q9KZC7': 'GlnA3 (Q9KZC7)'}


def poses_for(name, seed):
    pred = HERE / 'stage' / f'{name}_s{seed}' / f'boltz_results_{name}' / 'predictions' / name
    models = sorted(pred.glob(f'{name}_model_*.pdb'))
    return [ac.one_model(p, 'E') for p in models]


def main():
    man = json.loads((HERE / 'manifest.json').read_text())
    out = {}
    print(f'{"enzyme":16s} {"cond":5s} {"seed":>5s} {"n<4A/50":>8s} {"medD":>6s} {"candAng":>8s}')
    for acc in ('P78061', 'Q9KZC7'):
        out[acc] = {}
        for cond in ('free', 'put'):
            name = f'{acc}_{cond}'
            per_seed, allposes = [], []
            for s in SEEDS:
                ps = poses_for(name, s)
                if len(ps) != 50:
                    print(f'  WARN {name} s{s}: {len(ps)} models')
                allposes += ps
                d = np.array([p['d_gamma'] for p in ps]) if ps else np.array([])
                n4 = int((d < 4).sum()) if ps else 0
                md = round(float(np.median(d)), 2) if ps else None
                a = np.array([p['ang_gamma'] for p in ps]) if ps else np.array([])
                ca = round(float(a[d < 4].mean()), 1) if ps and (d < 4).any() else None
                per_seed.append({'seed': s, 'n_within_4A': n4, 'median_A': md, 'cand_angle': ca})
                print(f'{LABEL[acc]:16s} {cond:5s} {s:5d} {str(n4)+"/50":>8s} {str(md):>6s} {str(ca):>8s}')
            d = np.array([p['d_gamma'] for p in allposes]); a = np.array([p['ang_gamma'] for p in allposes])
            w = d < 4
            agg = {'n_poses': len(allposes), 'n_within_4A': int(w.sum()),
                   'frac_within_4A': round(float(w.mean()), 3),
                   'median_A': round(float(np.median(d)), 2), 'min_A': round(float(d.min()), 2),
                   'near_attack_fraction': round(float((w & (a > 150)).mean()), 3),
                   'candidate_angle_deg': round(float(a[w].mean()), 1) if w.any() else None,
                   'per_seed': per_seed}
            out[acc][cond] = agg
    (HERE / 'matrix_geometry.json').write_text(json.dumps(out, indent=2) + '\n')
    print('\n=== POOLED (3 seeds, 150 poses) ===')
    print(f'{"enzyme":16s} {"cond":5s} {"n<4A/150":>9s} {"frac<4A":>8s} {"medD":>6s} {"candAng":>8s} {"nearAtk":>8s}')
    for acc in ('P78061', 'Q9KZC7'):
        for cond in ('free', 'put'):
            g = out[acc][cond]
            print(f'{LABEL[acc]:16s} {cond:5s} {str(g["n_within_4A"])+"/150":>9s} '
                  f'{g["frac_within_4A"]:8.3f} {g["median_A"]:6.2f} {str(g["candidate_angle_deg"]):>8s} '
                  f'{g["near_attack_fraction"]:8.3f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
