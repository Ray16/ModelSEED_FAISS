#!/usr/bin/env python
"""Build the pinned ModelSEED reaction corpus and 4096-component fingerprints.

Normalize stored structures, preserve reaction stoichiometry, and publish the
reaction table, embedding and content-hash manifest together. Zero-vector
reactions are excluded by search, not silently treated as neighbors.
"""
from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator as fpg
from drfp import DrfpEncoder

from amide_synthesis.reaction_search.normalize import normalize

RDLogger.DisableLog('rdApp.*')
from amide_synthesis.infrastructure.paths import HERE, DB, RES

HPLUS = 'cpd00067'
from amide_synthesis.infrastructure.workers import worker_count, describe
WORKERS = worker_count()


def compound_tables():
    """Only the real tables; the refresh added *.provenance.tsv sidecars."""
    return [p for p in sorted(glob.glob(str(DB / 'compound_*.tsv')))
            if 'provenance' not in Path(p).name]


def composition(mol):
    atoms = Counter()
    for atom in mol.GetAtoms():
        atoms[atom.GetSymbol()] += 1
        atoms['H'] += atom.GetTotalNumHs()
    return atoms, Chem.GetFormalCharge(mol)


def _normalize_block(items):
    """Worker: (cpd_id, raw_smiles) -> (cpd_id, canonical, composition, flags)."""
    out = []
    for cpd, raw in items:
        mol = Chem.MolFromSmiles(raw)
        if mol is None:
            continue
        fixed = normalize(mol)
        canonical = Chem.MolToSmiles(fixed)
        repaired = canonical != Chem.MolToSmiles(mol)
        charge_changed = repaired and Chem.GetFormalCharge(fixed) != Chem.GetFormalCharge(mol)
        out.append((cpd, canonical, composition(fixed), repaired, charge_changed))
    return out


def load_compounds():
    """Repaired SMILES plus the per-compound composition needed to audit balance."""
    raw_rows = []
    for path in compound_tables():
        for row in csv.DictReader(open(path), delimiter='\t'):
            raw = row.get('smiles', '')
            if raw and raw != 'null':
                raw_rows.append((row['id'], raw))

    smiles, comp = {}, {}
    audit = {'repaired': 0, 'charge_changed': 0}
    block = 1000
    chunks = [raw_rows[i:i + block] for i in range(0, len(raw_rows), block)]
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for result in pool.map(_normalize_block, chunks):
            for cpd, canonical, counts, repaired, charge_changed in result:
                smiles[cpd] = canonical
                comp[cpd] = counts
                audit['repaired'] += repaired
                audit['charge_changed'] += charge_changed
    print(f"repaired {audit['repaired']:,} structures "
          f"({audit['charge_changed']:,} of them changed formal charge)", flush=True)
    return smiles, comp, audit


def all_full_ecs(value):
    found = []
    for token in (value or '').replace('|', ';').split(';'):
        token = token.strip()
        parts = token.split('.')
        if len(parts) == 4 and all(p.isdigit() for p in parts):
            found.append(token)
    return list(dict.fromkeys(found))


def net_stoich(pairs):
    totals = defaultdict(float)
    for coef, cpd in pairs:
        if cpd != HPLUS:
            totals[cpd] += coef
    return {k: v for k, v in totals.items() if abs(v) > 1e-9}


def reaction_smiles(stoich, smiles):
    left, right = [], []
    for cpd, coef in sorted(stoich.items()):
        if cpd not in smiles:
            return None, None
        mult = min(4, max(1, int(round(abs(coef)))))
        (left if coef < 0 else right).extend([(smiles[cpd], cpd)] * mult)
    if not left or not right:
        return None, None
    return '.'.join(s for s, _ in left) + '>>' + '.'.join(s for s, _ in right), (left, right)


def balance_audit(sides, comp):
    """Heavy-atom balance (repair-invariant) and the proton delta repair implies."""
    left, right = sides
    lhs, rhs = Counter(), Counter()
    lhs_charge = rhs_charge = 0
    for side, atoms, is_left in ((left, lhs, True), (right, rhs, False)):
        for _, cpd in side:
            if cpd not in comp:
                return None, None, None
            counts, charge = comp[cpd]
            atoms.update(counts)
            if is_left:
                lhs_charge += charge
            else:
                rhs_charge += charge
    heavy = {el for el in set(lhs) | set(rhs) if el != 'H' and rhs[el] != lhs[el]}
    h_delta = int(rhs['H'] - lhs['H'])
    charge_delta = int(rhs_charge - lhs_charge)
    # Protons are the only free variable once H+ is stripped, so an imbalance is
    # repairable ONLY if the hydrogen and charge deltas agree. Recording this
    # separately keeps "balanced after adding H+" distinct from "simply wrong".
    proton_fixable = (not heavy) and (h_delta == charge_delta)
    return (not heavy), h_delta, proton_fixable


def git_state():
    def run(*args):
        return subprocess.run(args, cwd=DB.parent, capture_output=True,
                              text=True).stdout.strip()
    return {'branch': run('git', 'rev-parse', '--abbrev-ref', 'HEAD'),
            'commit': run('git', 'rev-parse', 'HEAD'),
            'commit_date': run('git', 'log', '-1', '--format=%ad', '--date=short')}


def _encode_block(reactions):
    """Top-level so ProcessPoolExecutor can pickle it."""
    return np.array(DrfpEncoder.encode(reactions, n_folded_length=2048),
                    dtype=np.float32)


_MORGAN = None


def _ecfp_block(reactions):
    """Worker: ECFP4 count-difference vectors for a block of reactions."""
    global _MORGAN
    if _MORGAN is None:
        _MORGAN = fpg.GetMorganGenerator(radius=2, fpSize=2048)
    cache = {}

    def side(parts):
        vec = np.zeros(2048, np.float32)
        for smi in parts:
            if smi not in cache:
                cache[smi] = Chem.MolFromSmiles(smi)
            mol = cache[smi]
            if mol is not None:
                vec += _MORGAN.GetCountFingerprintAsNumPy(mol).astype(np.float32)
        return vec

    out = np.zeros((len(reactions), 2048), np.float32)
    for i, rxn in enumerate(reactions):
        lhs, rhs = rxn.split('>>')
        out[i] = side(rhs.split('.')) - side(lhs.split('.'))
    return out


def main():
    from amide_synthesis.infrastructure.paths import ensure_directories
    ensure_directories()
    smiles, comp, audit = load_compounds()
    print(f'{len(smiles):,} compounds with a parseable structure', flush=True)

    rows = []
    for path in sorted(glob.glob(str(DB / 'reaction_*.tsv'))):
        for row in csv.DictReader(open(path), delimiter='\t'):
            if row.get('is_obsolete') == '1' or row.get('is_transport') == '1':
                continue
            pairs = []
            for part in row.get('stoichiometry', '').split(';'):
                field = part.split(':')
                if len(field) >= 2 and field[1]:
                    try:
                        pairs.append((float(field[0]), field[1]))
                    except ValueError:
                        pass
            stoich = net_stoich(pairs)
            if not stoich:
                continue
            rxn, sides = reaction_smiles(stoich, smiles)
            if rxn is None:
                continue
            heavy_ok, h_delta, proton_fixable = balance_audit(sides, comp)
            ecs = all_full_ecs(row.get('ec_numbers'))
            primary = ecs[0] if ecs else ''
            rows.append({
                'rxn_id': row['id'], 'rxn_smiles': rxn,
                'ec': primary,
                'ec1': primary.split('.')[0] if primary else '',
                'ec2': '.'.join(primary.split('.')[:2]) if primary else '',
                'ec3': '.'.join(primary.split('.')[:3]) if primary else '',
                'ec_all': ';'.join(ecs), 'n_ec': len(ecs),
                'status': row.get('status', ''), 'n_cpd': len(stoich),
                'heavy_balanced': bool(heavy_ok) if heavy_ok is not None else False,
                'implied_h_delta': int(h_delta) if h_delta is not None else 0,
                'proton_fixable': bool(proton_fixable) if proton_fixable is not None
                                  else False})

    corpus = pd.DataFrame(rows).drop_duplicates('rxn_id').reset_index(drop=True)
    print(f'{len(corpus):,} scoreable reactions '
          f'({int((corpus.ec != "").sum()):,} with a full 4-level EC); '
          f'{int((~corpus.heavy_balanced).sum()):,} not heavy-atom balanced, '
          f'{int((~corpus.proton_fixable).sum()):,} not fixable by protons alone',
          flush=True)

    # ---- ECFP4-diff || DRFP, identical recipe to pipeline/01_embed.
    # Both halves are pure-CPU and embarrassingly parallel; fan them out.
    reactions = corpus.rxn_smiles.to_numpy()
    block = 500
    chunks = [list(reactions[i:i + block]) for i in range(0, len(reactions), block)]
    print(f'ECFP4 + DRFP on {describe(WORKERS)}', flush=True)

    diff = np.zeros((len(reactions), 2048), np.float32)
    drfp = np.zeros((len(reactions), 2048), np.float32)
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        jobs = {}
        for i, chunk in enumerate(chunks):
            jobs[pool.submit(_ecfp_block, chunk)] = ('ecfp', i)
            jobs[pool.submit(_encode_block, chunk)] = ('drfp', i)
        done = 0
        for future in as_completed(jobs):
            kind, i = jobs[future]
            target = diff if kind == 'ecfp' else drfp
            target[i * block:i * block + len(chunks[i])] = future.result()
            done += 1
            if done % 40 == 0:
                print(f'  {done}/{len(jobs)} blocks', flush=True)

    embedding = np.hstack([diff, drfp])

    # Data and embeddings first, manifest LAST: later stages trust the manifest's
    # hash, so a half-finished run cannot masquerade as complete.
    # Publish atomically: readers either see the previous complete pair or the
    # new one, never a half-written file. np.save appends .npy, so name the
    # temporary file with that suffix already present.
    corpus_tmp = HERE / 'data/.modelseed_master.parquet.tmp'
    embedding_tmp = HERE / 'embeddings/.ecfp4_drfp_master_tmp.npy'
    corpus.to_parquet(corpus_tmp, index=False)
    np.save(embedding_tmp, embedding)
    assert embedding_tmp.exists(), embedding_tmp
    corpus_tmp.replace(HERE / 'data/modelseed_master.parquet')
    embedding_tmp.replace(HERE / 'embeddings/ecfp4_drfp_master.npy')

    content_hash = hashlib.sha256(
        '\n'.join(corpus.rxn_id + '\t' + corpus.rxn_smiles).encode()).hexdigest()
    # Shape alone does not pin an embedding: reordered rows pass a shape check.
    embedding_hash = hashlib.sha256(np.ascontiguousarray(embedding)).hexdigest()
    manifest = git_state() | {
        'n_compounds_with_structure': len(smiles),
        'structures_repaired': audit['repaired'],
        'structures_repaired_with_charge_change': audit['charge_changed'],
        'n_reactions': int(len(corpus)),
        'n_reactions_with_full_ec': int((corpus.ec != '').sum()),
        'n_reactions_not_heavy_balanced': int((~corpus.heavy_balanced).sum()),
        'n_reactions_not_proton_fixable': int((~corpus.proton_fixable).sum()),
        'embedding_shape': list(embedding.shape),
        'corpus_sha256': content_hash,
        'embedding_sha256': embedding_hash,
        'rdkit_version': Chem.rdBase.rdkitVersion,
        'proton_convention': 'H+ (cpd00067) stripped from every reaction',
    }
    (HERE / 'results/corpus_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
