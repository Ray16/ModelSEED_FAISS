"""Build step-2 (aminolysis) cofold inputs for the ten-reaction panel.

Step 1 (already modelled) places ATP + adipate for acid activation. Step 2 models the
amide-forming half-reaction: the activated acyl intermediate is attacked by the amine.
Two mechanisms, so two intermediates and two metal counts:

  acyl-adenylate route (ANL/NIS adenylating ligases, 1 Mg)
      intermediate = adipoyl-AMP (mixed carboxylic-phosphoric anhydride with AMP)
      leaving group AMP is part of the intermediate; PPi already departed.
  acyl-phosphate route (glutamine-synthetase-like and Mur/ATP-grasp ligases, 2 Mg)
      intermediate = adipoyl-phosphate; ADP retained (post-transfer Michaelis complex),
      two Mg(2+) coordinate ADP and the acyl-phosphate.

The nucleophile is hexamethylenediamine with ONE primary amine left neutral (-NH2) so it
can attack; the distal amine stays protonated (-NH3+), as in the reference step-2 panel
(/homes/rzhu/Enzyme_agent/exp_data/amide_synthetase/boltz2/step2_1mg). The amine is
placed by a pocket constraint derived from where adipate actually bound in that enzyme's
own step-1 pose, not a hand-picked residue list.

Honesty note: on the experimental amide-synthetase panel, step-2 intermediate-state
geometry did NOT separate actives (Spearman ~0.11), while step-1 adenylation geometry did
(AUROC ~0.98). Step 2 here tests whether the amine can be accommodated in a productive
near-attack pose (completeness of the mechanism), and is not advanced as a ranking signal.
"""
import csv
import hashlib
import json
from pathlib import Path

import gemmi
import yaml

from amide_synthesis.infrastructure.paths import RES

# Heavy-atom signature (C, N, O, P) of each activated acyl intermediate, used by the
# step-2 runner to confirm the modelled complex carries the right species.
INTERMEDIATE_HEAVY = {'acyl_adenylate': (16, 5, 10, 1), 'acyl_phosphate': (6, 0, 7, 1)}

# Activated acyl intermediates and the neutral-amine nucleophile.
ADIPOYL_AMP = 'Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)([O-])OC(=O)CCCCC(=O)[O-])[C@@H](O)[C@H]1O'
ADIPOYL_PHOSPHATE = 'O=C([O-])CCCCC(=O)OP(=O)([O-])[O-]'
HMDA_MONONEUTRAL = 'NCCCCCC[NH3+]'  # one -NH2 nucleophile, one -NH3+ spectator

POCKET_RADIUS_A = 7.0
POCKET_MAX_CONTACTS = 20

# (accession, step-1 run basename, step-1 outputs root). 2mg = acyl-phosphate route.
PANEL = [
    ('I3R584', 'I3R584_2mg', 'campaign'), ('Q9KZC7', 'Q9KZC7_2mg', 'campaign'),
    ('P39641', 'P39641_2mg', 'campaign'),
    ('P78061', 'P78061_2mg', 'activation'), ('Q9RDS6', 'Q9RDS6_2mg', 'activation'),
    ('Q9WY79', 'Q9WY79_2mg', 'activation'),
    ('E2JA32', 'E2JA32_1mg', 'activation'), ('Q2FW70', 'Q2FW70_1mg', 'activation'),
    ('Q2G1M9', 'Q2G1M9_1mg', 'activation'), ('A0A0F7RJ52', 'A0A0F7RJ52_1mg', 'activation'),
    ('Q2G1N1', 'Q2G1N1_1mg', 'activation'), ('Q2G1M8', 'Q2G1M8_1mg', 'activation'),
    ('Q81RQ8', 'Q81RQ8_1mg', 'activation'),
]


def step1_paths(name, where=None):
    # All step-1 ensembles (panel + reused + expanded) now live in one folder.
    base = RES / 'cofold_step1'
    return base / 'inputs' / f'{name}.yaml', base / 'outputs' / name


def adipate_chain(model):
    """The ligand chain whose residue is the adipate dianion (6 C, 4 O, 10 heavy)."""
    for chain in model:
        heavy = [a for r in chain for a in r if a.element.name != 'H']
        if (len(heavy) == 10 and sum(a.element.name == 'C' for a in heavy) == 6
                and sum(a.element.name == 'O' for a in heavy) == 4):
            return chain
    raise ValueError('adipate chain not found')


def representative_pose(out_dir, name):
    """Pick the step-1 model where an adipate oxygen sits closest to the ATP phosphorus."""
    best, best_d = None, float('inf')
    for pdb in sorted((out_dir / 'by_seed' / name / 's42').glob(f'{name}_model_*.pdb')):
        model = gemmi.read_structure(str(pdb))[0]
        phosphorus = [a.pos for r in model['B'] for a in r if a.element.name == 'P']
        oxygens = [a.pos for r in adipate_chain(model) for a in r if a.element.name == 'O']
        if not phosphorus or not oxygens:
            continue
        d = min(o.dist(p) for o in oxygens for p in phosphorus)
        if d < best_d:
            best, best_d = model, d
    if best is None:
        raise ValueError('no usable step-1 pose for ' + name)
    return best


def pocket_contacts(model):
    """Protein residues whose nearest heavy atom is within POCKET_RADIUS_A of adipate."""
    acid_atoms = [a.pos for r in adipate_chain(model) for a in r if a.element.name != 'H']
    scored = []
    for residue in model['A']:
        atoms = [a.pos for a in residue if a.element.name != 'H']
        if not atoms:
            continue
        d = min(p.dist(q) for p in atoms for q in acid_atoms)
        if d <= POCKET_RADIUS_A:
            scored.append((d, residue.seqid.num))
    scored.sort()
    return sorted(num for _, num in scored[:POCKET_MAX_CONTACTS])


def build_one(accession, name, where):
    in_yaml, out_dir = step1_paths(name, where)
    step1 = yaml.safe_load(in_yaml.read_text())
    protein = next(s['protein'] for s in step1['sequences'] if 'protein' in s)
    two_mg = name.endswith('_2mg')
    contacts = pocket_contacts(representative_pose(out_dir, name))

    ligands = [{'ligand': {'id': 'B', 'smiles': ADIPOYL_PHOSPHATE if two_mg else ADIPOYL_AMP}},
               {'ligand': {'id': 'C', 'smiles': HMDA_MONONEUTRAL}}]
    chain = 'D'
    if two_mg:  # acyl-phosphate route keeps ADP as the post-transfer partner
        ligands.append({'ligand': {'id': chain, 'ccd': 'ADP'}}); chain = chr(ord(chain) + 1)
    for _ in range(2 if two_mg else 1):
        ligands.append({'ligand': {'id': chain, 'ccd': 'MG'}}); chain = chr(ord(chain) + 1)

    doc = {
        'version': 1,
        'sequences': [{'protein': {'id': 'A', 'sequence': protein['sequence'],
                                   'msa': protein['msa']}}] + ligands,
        'constraints': [{'pocket': {'binder': 'C',
                                    'contacts': [['A', n] for n in contacts],
                                    'max_distance': 10.0, 'force': True}}],
    }
    return doc, two_mg, contacts, protein


def main():
    out = RES / 'cofold_step2'
    (out / 'inputs').mkdir(parents=True, exist_ok=True)
    for sub in ('outputs', 'state', 'logs', 'evidence'):
        (out / sub).mkdir(parents=True, exist_ok=True)
    rows, manifest_inputs = [], []
    for accession, name, where in PANEL:
        doc, two_mg, contacts, protein = build_one(accession, name, where)
        path = out / 'inputs' / f'{name}.yaml'
        route = 'acyl_phosphate' if two_mg else 'acyl_adenylate'
        header = (f'# STEP 2 aminolysis ({"2mg acyl-phosphate+ADP" if two_mg else "1mg acyl-adenylate"}): '
                  'A=protein B=acyl-intermediate C=HMDA(one -NH2 neutral) '
                  f'{"D=ADP E,F=Mg" if two_mg else "D=Mg"}\n')
        path.write_text(header + yaml.safe_dump(doc, sort_keys=False))
        sequence = protein['sequence']
        manifest_inputs.append({
            'accession': accession, 'name': name, 'route': route,
            'input': str(path), 'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'msa': protein['msa'], 'msa_sha256': hashlib.sha256(Path(protein['msa']).read_bytes()).hexdigest(),
            'sequence': sequence, 'sequence_sha256': hashlib.sha256(sequence.encode()).hexdigest(),
            'mg_count': 2 if two_mg else 1,
            'intermediate_heavy': list(INTERMEDIATE_HEAVY[route]),
            'adp': two_mg, 'n_pocket_contacts': len(contacts)})
        rows.append({'accession': accession, 'name': name, 'route': route,
                     'mg_count': 2 if two_mg else 1,
                     'intermediate': 'adipoyl_phosphate+ADP' if two_mg else 'adipoyl_AMP',
                     'n_pocket_contacts': len(contacts),
                     'pocket_contacts': ';'.join(map(str, contacts))})
    with (out / 'step2_manifest.tsv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader(); writer.writerows(rows)
    manifest = {
        'scope': 'Step-2 aminolysis: acyl intermediate + HMDA (one -NH2 neutral) + Mg (+ADP on '
                 'the acyl-phosphate route). Tests whether the C6 diamine is accommodated for '
                 'near-attack; not advanced as an activity ranking (step-2 geometry was '
                 'non-predictive on the experimental panel).',
        'inputs': manifest_inputs,
        'settings': {'mode': 'exact', 'seed': 42, 'diffusion_samples': 50,
                     'max_parallel_samples': 41, 'recycling_steps': 3, 'sampling_steps': 200},
        'expected_models': 50 * len(manifest_inputs)}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Wrote {len(rows)} step-2 inputs + manifest to {out}')
    for r in rows:
        print(f"  {r['name']:16s} {r['route']:14s} Mg={r['mg_count']} "
              f"pocket={r['n_pocket_contacts']} residues")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
