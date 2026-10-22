"""Explicit preparation and read-only verification of structural campaigns.

Preparation creates a new run from declared sequences, metals, reaction mechanism
and existing MSAs. It does not retrieve MSAs, reserve GPUs or launch inference.
"""
from pathlib import Path
import csv
import hashlib
import json
import math


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(selection, run_dir):
    import yaml
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    selection, run_dir = Path(selection).resolve(), Path(run_dir).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError('Campaign output must be new or empty; frozen runs cannot be overwritten')
    rows = list(csv.DictReader(selection.open(), delimiter='\t'))
    required = {'accession', 'sequence', 'mg_count', 'phosphate', 'msa'}
    if not rows or not required <= rows[0].keys():
        raise ValueError('Selection requires accession, sequence, mg_count, phosphate and msa')
    acid = '[O-]C(=O)CCCCC(=O)[O-]'
    molecule = Chem.MolFromSmiles(acid)
    assert Chem.GetFormalCharge(molecule) == -2
    assert rdMolDescriptors.CalcMolFormula(molecule) == 'C6H8O4-2'
    prepared = []; names = set()
    for row in rows:
        accession, sequence = row['accession'], row['sequence']
        if not accession or not all(c.isalnum() or c in '_.-' for c in accession):
            raise ValueError('Invalid accession for an input filename')
        if not sequence or set(sequence) - set('ACDEFGHIKLMNPQRSTVWY'):
            raise ValueError('Expected a complete, unambiguous protein sequence')
        mg = int(row['mg_count']); phosphate = row['phosphate']
        if mg < 1 or phosphate not in {'PA', 'PG'}:
            raise ValueError('Declare a positive Mg count and PA or PG mechanism')
        msa = Path(row['msa'])
        msa = msa if msa.is_absolute() else selection.parent / msa
        msa = msa.resolve()
        with msa.open() as handle:
            alignment = list(csv.DictReader(handle))
        if len(alignment) < 2:
            raise ValueError('Query-only MSA is not accepted')
        query = ''.join(c for c in alignment[0]['sequence'] if not c.islower()).replace('-', '')
        if query != sequence:
            raise ValueError('MSA query differs from selected protein')
        name = f'{accession}_{mg}mg'
        if name in names:
            raise ValueError('Duplicate campaign input: '+name)
        names.add(name)
        ids = iter('BCDEFGHIJKLMNOPQRSTUVWXYZ')
        sequences = [{'protein': {'id': 'A', 'sequence': sequence, 'msa': str(msa)}},
                     {'ligand': {'id': next(ids), 'ccd': 'ATP'}}]
        sequences += [{'ligand': {'id': next(ids), 'ccd': 'MG'}} for _ in range(mg)]
        acid_chain = next(ids)
        sequences.append({'ligand': {'id': acid_chain, 'smiles': acid}})
        prepared.append((row, name, msa, acid_chain, {'version': 1, 'sequences': sequences}))
    for name in ['inputs', 'outputs', 'state', 'logs', 'evidence']:
        (run_dir/name).mkdir(parents=True, exist_ok=True)
    inputs = []
    for row, name, msa, chain, config in prepared:
        path = run_dir/'inputs'/(name+'.yaml'); path.write_text(yaml.safe_dump(config, sort_keys=False))
        inputs.append(dict(accession=row['accession'], name=name, sequence_length=len(row['sequence']),
            sequence_sha256=hashlib.sha256(row['sequence'].encode()).hexdigest(),
            mg_count=int(row['mg_count']), phosphate=row['phosphate'],
            leaving_oxygen='O3A' if row['phosphate']=='PA' else 'O3B',
            mechanism_family=row.get('family', 'declared'), acid_chain=chain, acid_smiles=acid,
            input=str(path), input_sha256=digest(path), msa=str(msa), msa_sha256=digest(msa),
            metal_assignment_basis=row.get('metal_assignment_basis', 'Explicit user-supplied assignment')))
    manifest = dict(scope='Monomeric adipate acid activation; no HMDA or activated intermediate',
        selection=str(selection), selection_sha256=digest(selection), inputs=inputs,
        settings=dict(mode='exact', seed=42, diffusion_samples=50, max_parallel_samples=41,
                      recycling_steps=3, sampling_steps=200), expected_models=50*len(inputs))
    (run_dir/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def verify(run_dir):
    """Re-read all actual models and compare geometry with saved completion rows."""
    from .poses import inspect
    run_dir = Path(run_dir).resolve()
    manifest = json.loads((run_dir/'manifest.json').read_text())
    failures = []; checked = 0; models = 0
    for item in manifest['inputs']:
        try:
            done = json.loads((run_dir/'state'/(item['name']+'.done.json')).read_text())
            actual = inspect(item, run_dir/'outputs'/item['name'],
                             include_near_attack='near_attack_fraction' in done)
            for key, value in actual.items():
                if key == 'poses':
                    continue
                expected = done[key]
                if isinstance(value, float):
                    if not math.isclose(value, expected, rel_tol=1e-11, abs_tol=1e-11):
                        raise ValueError('Saved geometry differs: '+key)
                elif value != expected:
                    raise ValueError('Saved assembly metadata differs: '+key)
            checked += 1; models += actual['n_models']
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            failures.append({'input': item['name'], 'error': str(exc)})
    return dict(run_dir=str(run_dir), expected_ensembles=len(manifest['inputs']),
                checked_ensembles=checked, checked_models=models, failures=failures,
                ok=not failures and checked==len(manifest['inputs']))


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    inspect_parser = commands.add_parser('inspect', help='read-only validation of saved structures and scores')
    inspect_parser.add_argument('--run-dir', required=True, type=Path)
    inspect_parser.add_argument('--output', type=Path, help='optional verification JSON outside frozen inputs')
    prepare_parser = commands.add_parser('prepare', help='create new inputs; no prediction or reservation')
    prepare_parser.add_argument('--selection', required=True, type=Path)
    prepare_parser.add_argument('--run-dir', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command=='prepare':
        result = prepare(args.selection, args.run_dir)
    else:
        result = verify(args.run_dir)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    return 0 if result.get('ok', True) else 1
