"""Geometry for the GS native-substrate positive control (L-glutamate on PuuA/GlnA3).

Replicates the step-1 in-line attack-angle metric atom-for-atom (nucleophile oxygen ->
Pgamma -> O3B leaving oxygen; distance filter <4A; mean in-line angle among engaged
poses, conclusive only if >=5), but for L-glutamate instead of the symmetric adipate.

Glutamate has two carboxylates; only the gamma (side-chain) carboxylate is phosphorylated
by GS. So we report TWO readings:
  - any-O : closest glutamate oxygen to Pgamma (exact parallel to the adipate rule, which
            is unambiguous only because adipate is symmetric), and WHICH carboxylate it is;
  - gamma : closest gamma-carboxylate oxygen (the mechanistically reactive one).
The fraction of poses whose closest oxygen is the gamma-carboxylate is itself diagnostic:
it says whether Boltz even places the correct carboxylate at the phosphorus.
"""
import json
import math
from pathlib import Path

import gemmi
import numpy as np

HERE = Path(__file__).resolve().parent
BOND = 1.7  # A, heavy-atom bond cutoff


def atoms_of(chain):
    return [a for r in chain for a in r if a.element.name != 'H']


def classify_glutamate(glu_atoms):
    """Return (gamma_carboxyl_O positions, alpha_carboxyl_O positions, all_O positions)."""
    Os = [a for a in glu_atoms if a.element.name == 'O']
    Cs = [a for a in glu_atoms if a.element.name == 'C']
    Ns = [a for a in glu_atoms if a.element.name == 'N']
    assert len(Os) == 4 and len(Cs) == 5 and len(Ns) == 1, \
        f'unexpected glutamate composition O{len(Os)} C{len(Cs)} N{len(Ns)}'
    N = Ns[0].pos
    # carboxyl carbons: C with >=2 O neighbours within bond distance
    carboxyl = []
    for c in Cs:
        near_O = [o for o in Os if c.pos.dist(o.pos) < BOND]
        if len(near_O) >= 2:
            carboxyl.append((c, near_O))
    assert len(carboxyl) == 2, f'expected 2 carboxyl C, got {len(carboxyl)}'
    # alpha-carboxyl C is the one closest (through space) to the amino N; gamma is the far one
    carboxyl.sort(key=lambda cg: cg[0].pos.dist(N))
    (alpha_C, alpha_O), (gamma_C, gamma_O) = carboxyl[0], carboxyl[1]
    return ([o.pos for o in gamma_O], [o.pos for o in alpha_O], [o.pos for o in Os])


def angle_at_P(o_pos, p_pos, leaving_pos):
    u = np.array([o_pos.x - p_pos.x, o_pos.y - p_pos.y, o_pos.z - p_pos.z])
    v = np.array([leaving_pos.x - p_pos.x, leaving_pos.y - p_pos.y, leaving_pos.z - p_pos.z])
    return math.degrees(math.acos(float(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1))))


def one_model(path, acid_chain):
    model = gemmi.read_structure(str(path))[0]
    nuc = {a.name: a.pos for r in model['B'] for a in r if a.element.name != 'H'}
    for req in ('PG', 'O3B'):
        if req not in nuc:
            raise ValueError(f'missing ATP atom {req} in {path.name}')
    PG, O3B = nuc['PG'], nuc['O3B']
    gamma_O, alpha_O, all_O = classify_glutamate(atoms_of(model[acid_chain]))
    # any-O reading (parallel to adipate)
    o_any = min(all_O, key=lambda o: o.dist(PG))
    d_any = o_any.dist(PG)
    is_gamma = min(gamma_O, key=lambda o: o.dist(o_any)).dist(o_any) < 1e-6
    # gamma reading (mechanistic)
    o_g = min(gamma_O, key=lambda o: o.dist(PG))
    return {
        'model': path.name,
        'd_any': d_any, 'ang_any': angle_at_P(o_any, PG, O3B), 'closest_is_gamma': is_gamma,
        'd_gamma': o_g.dist(PG), 'ang_gamma': angle_at_P(o_g, PG, O3B),
    }


def summarise(poses, dkey, akey):
    d = np.array([p[dkey] for p in poses], float)
    a = np.array([p[akey] for p in poses], float)
    inv = 1 / d
    within = d < 4.0
    n = int(within.sum())
    return {
        'n_models': len(poses),
        'weighted_attack_angle_A_deg': round(float((a * inv).sum() / inv.sum()), 1),
        'near_attack_fraction': round(float((within & (a > 150)).mean()), 3),
        'n_within_4A': n,
        'min_distance_A': round(float(d.min()), 2),
        'conclusive': n >= 5,
        'candidate_angle_deg': round(float(a[within].mean()), 1) if n else None,
    }


def main():
    manifest = json.loads((HERE / 'manifest.json').read_text())
    # adipate step-1 reference for the same two enzymes (from step1_pool_ranking.tsv)
    ref = {'P78061': {'candidate_angle_deg': 162.0, 'near_attack_fraction': 0.10, 'n_within_4A': 7},
           'Q9KZC7': {'candidate_angle_deg': 159.2, 'near_attack_fraction': 0.68, 'n_within_4A': 37}}
    out = {}
    for item in manifest['inputs']:
        name = item['name']
        seed_dir = HERE / 'outputs' / 'by_seed' / name / 's42'
        models = sorted(seed_dir.glob(f'{name}_model_*.cif')) or sorted(seed_dir.glob(f'{name}_model_*.pdb'))
        if len(models) != 50:
            print(f'WARN {name}: found {len(models)} models (expected 50)')
            if not models:
                continue
        poses = [one_model(p, item['acid_chain']) for p in models]
        rec = {
            'accession': item['accession'], 'name': name, 'n_models': len(poses),
            'frac_closest_is_gamma': round(float(np.mean([p['closest_is_gamma'] for p in poses])), 3),
            'any_O': summarise(poses, 'd_any', 'ang_any'),
            'gamma_O': summarise(poses, 'd_gamma', 'ang_gamma'),
            'adipate_reference': ref.get(item['accession']),
        }
        out[item['accession']] = rec
    (HERE / 'control_geometry.json').write_text(json.dumps(out, indent=2) + '\n')
    # console summary
    print(f'{"enzyme":22s} {"metric":8s} {"cand_angle":>10s} {"n<4A":>5s} {"near_atk":>8s} {"concl":>6s}')
    for acc, rec in out.items():
        tag = f"{rec['name']}"
        for mk in ('any_O', 'gamma_O'):
            s = rec[mk]
            print(f'{tag:22s} {mk:8s} {str(s["candidate_angle_deg"]):>10s} '
                  f'{s["n_within_4A"]:5d} {s["near_attack_fraction"]:8.3f} {str(s["conclusive"]):>6s}')
            tag = ''
        r = rec['adipate_reference']
        print(f'{"  (adipate ref)":22s} {"":8s} {str(r["candidate_angle_deg"]):>10s} '
              f'{r["n_within_4A"]:5d} {r["near_attack_fraction"]:8.3f}')
        print(f'  -> closest-O is the gamma-carboxylate in {rec["frac_closest_is_gamma"]*100:.0f}% of poses')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
