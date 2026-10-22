"""Build step-2 (aminolysis) inputs for every NEW candidate in the expanded pool.

Same chemistry as step2_build (acyl-adenylate+1Mg or acyl-phosphate+ADP+2Mg per
mechanism; HMDA with one neutral amine; amine placed by a pocket derived from the
enzyme's own step-1 adipate pose) but driven by pool.tsv over the expanded step-1
runs (campaign/ expressible, campaign_hard/ hard-host). The 13 references already
have step-2, so only the 111 new (status new / new_hardhost) are built here.
"""
import csv
import hashlib
import json
from pathlib import Path

import yaml

from amide_synthesis.infrastructure.paths import RES
from amide_synthesis.structure_modeling.step2_build import (
    ADIPOYL_AMP, ADIPOYL_PHOSPHATE, HMDA_MONONEUTRAL, INTERMEDIATE_HEAVY,
    representative_pose, pocket_contacts)

EXP = RES / 'top10_cofold_expanded'
OUT = RES / 'cofold_step2'
# All step-1 ensembles consolidated into one folder (no per-increment subdirs).
STEP1_RUN = {'new': RES / 'cofold_step1', 'new_hardhost': RES / 'cofold_step1'}


def build():
    (OUT / 'inputs').mkdir(parents=True, exist_ok=True)
    for sub in ('outputs', 'state', 'logs', 'evidence'):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    pool = [r for r in csv.DictReader((EXP / 'pool.tsv').open(), delimiter='\t')
            if r['status'] in ('new', 'new_hardhost')]
    manifest_inputs, rows, skipped = [], [], []
    for r in pool:
        acc, mg = r['accession'], int(r['mg_count'])
        name = f'{acc}_{mg}mg'
        run = STEP1_RUN[r['status']]
        step1_yaml = run / 'inputs' / f'{name}.yaml'
        step1_out = run / 'outputs' / name
        if not step1_yaml.exists() or not step1_out.exists():
            skipped.append(name); continue
        two_mg = r['family'] == 'atp_adp_ligase'
        route = 'acyl_phosphate' if two_mg else 'acyl_adenylate'
        protein = next(s['protein'] for s in yaml.safe_load(step1_yaml.read_text())['sequences']
                       if 'protein' in s)
        try:
            contacts = pocket_contacts(representative_pose(step1_out, name))
        except Exception as exc:  # noqa: BLE001 - record and skip un-poseable
            skipped.append(f'{name}:{exc}'); continue

        ligands = [{'ligand': {'id': 'B', 'smiles': ADIPOYL_PHOSPHATE if two_mg else ADIPOYL_AMP}},
                   {'ligand': {'id': 'C', 'smiles': HMDA_MONONEUTRAL}}]
        chain = 'D'
        if two_mg:
            ligands.append({'ligand': {'id': chain, 'ccd': 'ADP'}}); chain = chr(ord(chain) + 1)
        for _ in range(mg):
            ligands.append({'ligand': {'id': chain, 'ccd': 'MG'}}); chain = chr(ord(chain) + 1)
        doc = {'version': 1,
               'sequences': [{'protein': {'id': 'A', 'sequence': protein['sequence'],
                                          'msa': protein['msa']}}] + ligands,
               'constraints': [{'pocket': {'binder': 'C',
                                           'contacts': [['A', n] for n in contacts],
                                           'max_distance': 10.0, 'force': True}}]}
        path = OUT / 'inputs' / f'{name}.yaml'
        path.write_text(f'# STEP2 expanded {route} mg={mg}\n' + yaml.safe_dump(doc, sort_keys=False))
        seq = protein['sequence']
        manifest_inputs.append({
            'accession': acc, 'name': name, 'route': route,
            'input': str(path), 'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'msa': protein['msa'], 'msa_sha256': hashlib.sha256(Path(protein['msa']).read_bytes()).hexdigest(),
            'sequence': seq, 'sequence_sha256': hashlib.sha256(seq.encode()).hexdigest(),
            'mg_count': mg, 'intermediate_heavy': list(INTERMEDIATE_HEAVY[route]),
            'adp': two_mg, 'n_pocket_contacts': len(contacts)})
        rows.append({'accession': acc, 'name': name, 'route': route, 'status': r['status'],
                     'n_pocket_contacts': len(contacts)})
    manifest = {'scope': 'Step-2 aminolysis for expanded pool new candidates',
                'inputs': manifest_inputs,
                'settings': {'mode': 'exact', 'seed': 42, 'diffusion_samples': 50,
                             'max_parallel_samples': 41, 'recycling_steps': 3, 'sampling_steps': 200},
                'expected_models': 50 * len(manifest_inputs)}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    with (OUT / 'step2_manifest.tsv').open('w', newline='') as h:
        if rows:
            w = csv.DictWriter(h, fieldnames=list(rows[0]), delimiter='\t'); w.writeheader(); w.writerows(rows)
    print(f'built step-2 inputs for {len(manifest_inputs)} new candidates; skipped {len(skipped)}')
    if skipped:
        print('skipped:', skipped[:10])
    return len(manifest_inputs)


if __name__ == '__main__':
    raise SystemExit(0 if build() else 1)
