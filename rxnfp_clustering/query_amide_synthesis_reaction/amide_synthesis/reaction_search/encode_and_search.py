#!/usr/bin/env python
"""Encode the nylon-6,6 diad-forming reaction and find its nearest ModelSEED neighbours.

Target chemistry: the amide bond between adipate (C6 diacid) and hexamethylenediamine
(HMDA, C6 diamine). Free acid + free amine condensation is not how biology makes
amides, so the target is encoded in several mechanistic framings; each is a separate
query.

Encoding is the benchmark winner, identical to pipeline/01_embed/embed_fingerprints.py:
  ECFP4-diff (Morgan r=2, 2048, count, products - reactants)  ||  DRFP (2048, folded)
Retrieval is cosine similarity on L2-normalised vectors.

Two consistency rules make the queries comparable with the corpus:

1. SPECIES COME FROM THE DATABASE. Every cofactor and monomer is read from the
   same repaired ModelSEED structures the corpus is built from, rather than being
   hardcoded. Repair shifts charges (ATP -3 -> -4, CoA -5 -> -4), so a hardcoded
   copy silently drifts out of the reference geometry.

2. PROTONS ARE BALANCED, THEN STRIPPED. The corpus strips H+ (cpd00067). A query
   that keeps explicit H+ is encoded in a different convention; the measured
   cosine between proton-bearing and proton-stripped versions of the same
   reaction is ~0.97, not 1.0. So each query is first balanced by solving for the
   H+ it needs, asserted mass- and charge-balanced WITH those protons, and only
   then encoded with the protons removed.

Outputs (results/): queries.tsv, neighbours_labelled.tsv, neighbours_full.tsv,
ec_vote.tsv, ood.json
"""
from __future__ import annotations

import csv
import glob
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator as fpg
from drfp import DrfpEncoder

from amide_synthesis.infrastructure.paths import DB, RES
from amide_synthesis.reaction_search import corpus as _corpus
from amide_synthesis.reaction_search.normalize import normalize

RDLogger.DisableLog('rdApp.*')


TOPN = 25
GATE = 0.6          # EC3 neighbour-agreement threshold used by the deployed retriever
K = 15              # neighbour count used by the deployed retriever

# ---------------------------------------------------------------- species from the database
NEEDED = {
    'ADIPATE': 'cpd03642', 'HMDA': 'cpd25003', 'PUTRES': 'cpd00118',
    'CADAV': 'cpd01155', 'WATER': 'cpd00001', 'ATP': 'cpd00002',
    'ADP': 'cpd00008', 'AMP': 'cpd00018', 'PPI': 'cpd00012', 'PI': 'cpd00009',
    'COA': 'cpd00010', 'ADIPYLCOA': 'cpd09842',
}
PROTON = '[H+]'


def load_species():
    """Repaired ModelSEED structures for every species the queries need."""
    wanted = {cpd: key for key, cpd in NEEDED.items()}
    found = {}
    for path in sorted(glob.glob(str(DB / 'compound_*.tsv'))):
        if 'provenance' in Path(path).name:
            continue
        for row in csv.DictReader(open(path), delimiter='\t'):
            key = wanted.get(row.get('id', ''))
            if not key:
                continue
            raw = row.get('smiles', '')
            assert raw and raw != 'null', f'{row["id"]} ({key}) has no structure'
            mol = Chem.MolFromSmiles(raw)
            assert mol is not None, f'{row["id"]} ({key}) is unparseable'
            found[key] = Chem.MolToSmiles(normalize(mol))
    missing = set(NEEDED) - set(found)
    assert not missing, f'species missing from the database: {sorted(missing)}'
    return found


def composition(smiles):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f'unparseable SMILES: {smiles}'
    atoms = Counter()
    for atom in mol.GetAtoms():
        atoms[atom.GetSymbol()] += 1
        atoms['H'] += atom.GetTotalNumHs()
    return atoms, Chem.GetFormalCharge(mol)


def totals(side):
    atoms, charge = Counter(), 0
    for smiles in side:
        counts, formal = composition(smiles)
        atoms += counts
        charge += formal
    return atoms, charge


def balance_with_protons(name, left, right):
    """Solve for the H+ a reaction needs, then assert it is fully balanced.

    Heavy atoms must already match; protons are the only free variable.
    """
    lhs_atoms, lhs_charge = totals(left)
    rhs_atoms, rhs_charge = totals(right)
    heavy = {el: rhs_atoms[el] - lhs_atoms[el]
             for el in set(lhs_atoms) | set(rhs_atoms)
             if el != 'H' and rhs_atoms[el] != lhs_atoms[el]}
    assert not heavy, f'{name}: heavy-atom imbalance {heavy}'

    h_delta = rhs_atoms['H'] - lhs_atoms['H']
    charge_delta = rhs_charge - lhs_charge
    assert h_delta == charge_delta, (
        f'{name}: H delta {h_delta} but charge delta {charge_delta}; '
        'not fixable by protons alone')
    left, right = list(left), list(right)
    if h_delta > 0:
        left += [PROTON] * h_delta
    elif h_delta < 0:
        right += [PROTON] * (-h_delta)

    atoms_l, charge_l = totals(left)
    atoms_r, charge_r = totals(right)
    assert atoms_l == atoms_r, f'{name}: still mass-imbalanced'
    assert charge_l == charge_r, f'{name}: still charge-imbalanced'
    return left, right, charge_l


def build_queries():
    s = load_species()
    # The products are not ModelSEED compounds, so they are written here following
    # the same pH-7 convention as the database: amide neutral, remaining free amine
    # protonated, remaining carboxylate deprotonated.
    diad = Chem.CanonSmiles('O=C([O-])CCCCC(=O)NCCCCCC[NH3+]')
    diad_put = Chem.CanonSmiles('O=C([O-])CCCCC(=O)NCCCC[NH3+]')
    diad_cad = Chem.CanonSmiles('O=C([O-])CCCCC(=O)NCCCCC[NH3+]')

    specs = [
        ('direct_condensation',
         'Adipate + hexamethylenediamine -> diad amide + H2O (the literal nylon-6,6 step)',
         [s['ADIPATE'], s['HMDA']], [diad, s['WATER']]),
        ('atp_amp_ligase',
         'ATP-dependent amide ligase via acyl-adenylate (ATP -> AMP + PPi)',
         [s['ADIPATE'], s['HMDA'], s['ATP']], [diad, s['AMP'], s['PPI']]),
        ('atp_adp_ligase',
         'ATP-dependent amide ligase via acyl-phosphate (ATP -> ADP + Pi)',
         [s['ADIPATE'], s['HMDA'], s['ATP']], [diad, s['ADP'], s['PI']]),
        ('coa_aminolysis',
         'Adipyl-CoA aminolysis by hexamethylenediamine (N-acyltransferase route)',
         [s['ADIPYLCOA'], s['HMDA']], [diad, s['COA']]),
        ('coa_aminolysis_putrescine',
         'Adipyl-CoA + putrescine (native C4 diamine analogue probe)',
         [s['ADIPYLCOA'], s['PUTRES']], [diad_put, s['COA']]),
        ('coa_aminolysis_cadaverine',
         'Adipyl-CoA + cadaverine (native C5 diamine analogue probe)',
         [s['ADIPYLCOA'], s['CADAV']], [diad_cad, s['COA']]),
    ]

    rows = []
    for name, description, left, right in specs:
        left, right, charge = balance_with_protons(name, left, right)
        protons = left.count(PROTON) + right.count(PROTON)
        enc_left = [x for x in left if x != PROTON]       # corpus convention
        enc_right = [x for x in right if x != PROTON]
        rows.append({
            'query': name, 'description': description, 'net_charge': charge,
            'n_protons_balanced': protons,
            'rxn_smiles_balanced': '.'.join(left) + '>>' + '.'.join(right),
            'rxn_smiles': '.'.join(enc_left) + '>>' + '.'.join(enc_right)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- featurisation
MORGAN = fpg.GetMorganGenerator(radius=2, fpSize=2048)


def side_vector(smiles_list):
    vec = np.zeros(2048, np.float32)
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, smiles
        vec += MORGAN.GetCountFingerprintAsNumPy(mol).astype(np.float32)
    return vec


def featurise(reaction_smiles):
    diff = np.zeros((len(reaction_smiles), 2048), np.float32)
    for i, rxn in enumerate(reaction_smiles):
        lhs, rhs = rxn.split('>>')
        diff[i] = side_vector(rhs.split('.')) - side_vector(lhs.split('.'))
    drfp = np.array(DrfpEncoder.encode(list(reaction_smiles), n_folded_length=2048),
                    dtype=np.float32)
    return np.hstack([diff, drfp])


def unit(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    assert (norms > 0).all(), 'zero-norm reaction vector'
    return matrix / norms


# ---------------------------------------------------------------- search
def neighbours(query_matrix, reference_matrix, topn):
    similarity = query_matrix @ reference_matrix.T
    order = np.argsort(-similarity, axis=1)[:, :topn]
    return order, np.take_along_axis(similarity, order, axis=1)


def self_similarity_floor(reference_matrix, percentile=5.0, chunk=512):
    """5th percentile of the reference set's own 1-NN similarity -> OOD threshold."""
    best = np.empty(len(reference_matrix), np.float32)
    for start in range(0, len(reference_matrix), chunk):
        stop = min(len(reference_matrix), start + chunk)
        block = reference_matrix[start:stop] @ reference_matrix.T
        block[np.arange(stop - start), np.arange(start, stop)] = -np.inf
        best[start:stop] = block.max(axis=1)
    return float(np.percentile(best, percentile)), best


def main():
    from amide_synthesis.infrastructure.paths import ensure_directories
    ensure_directories()
    queries = build_queries()
    queries.to_csv(RES / 'queries.tsv', sep='\t', index=False)
    print(queries[['query', 'net_charge', 'n_protons_balanced']].to_string(index=False),
          flush=True)

    xq = unit(featurise(queries.rxn_smiles.to_numpy()))

    full, x_full = _corpus.load()
    labelled, x_labelled = _corpus.labelled(full, x_full)
    print(f'corpus {len(full):,} reactions, {len(labelled):,} EC-labelled '
          f'({_corpus.manifest().get("commit", "?")[:8]})', flush=True)

    # ---- top-N against the EC-labelled reference set
    order, sims = neighbours(xq, x_labelled, TOPN)
    rows = []
    for qi, name in enumerate(queries['query']):
        for rank, (ref, sim) in enumerate(zip(order[qi], sims[qi]), start=1):
            rec = labelled.iloc[ref]
            rows.append({'query': name, 'rank': rank,
                         'cosine_similarity': round(float(sim), 4),
                         'rxn_id': rec.rxn_id, 'ec': rec.ec, 'ec_all': rec.ec_all,
                         'rxn_smiles': rec.rxn_smiles})
    pd.DataFrame(rows).to_csv(RES / 'neighbours_labelled.tsv', sep='\t', index=False)

    # ---- top-N against all scoreable ModelSEED reactions
    order_f, sims_f = neighbours(xq, x_full, TOPN)
    rows = []
    for qi, name in enumerate(queries['query']):
        for rank, (ref, sim) in enumerate(zip(order_f[qi], sims_f[qi]), start=1):
            rec = full.iloc[ref]
            rows.append({'query': name, 'rank': rank,
                         'cosine_similarity': round(float(sim), 4),
                         'rxn_id': rec.rxn_id, 'ec': rec.ec, 'status': rec.status,
                         'rxn_smiles': rec.rxn_smiles})
    pd.DataFrame(rows).to_csv(RES / 'neighbours_full.tsv', sep='\t', index=False)

    # ---- confidence-gated K-NN EC vote, as the deployed retriever scores it.
    # DIAGNOSTIC ONLY. The vote runs on unfiltered neighbours, so it reports
    # whole-reaction similarity, not mechanism: the CoA queries return EC 3.1.2
    # (thioester HYDROLASES) at 0.8 agreement without abstaining, because the
    # CoA -> CoA cofactor change dominates the fingerprint. Mechanistic calls
    # come from 03_amide_enzymes.py, which screens before ranking.
    votes = []
    for qi, name in enumerate(queries['query']):
        neigh = labelled.iloc[order[qi][:K]]
        row = {'query': name, 'diagnostic_only': True,
               'top1_similarity': round(float(sims[qi][0]), 4)}
        for level in ('ec1', 'ec2', 'ec3'):
            counts = neigh[level].value_counts()
            row[f'{level}_call'] = counts.index[0]
            row[f'{level}_agreement'] = round(float(counts.iloc[0] / K), 3)
        row['abstain'] = bool(row['ec3_agreement'] < GATE)
        votes.append(row)
    votes = pd.DataFrame(votes)
    votes.to_csv(RES / 'ec_vote.tsv', sep='\t', index=False)
    print('\n', votes.to_string(), flush=True)

    # ---- OOD: is the query inside the manifold the retriever was validated on?
    floor, best = self_similarity_floor(x_labelled)
    ood = {'reference': f'EC-labelled subset (n={len(labelled)})',
           'ood_threshold_5th_pct_of_reference_1nn': round(floor, 4),
           'reference_1nn_median': round(float(np.median(best)), 4),
           'queries': {}}
    for qi, name in enumerate(queries['query']):
        top1 = float(sims[qi][0])
        ood['queries'][name] = {
            'top1_similarity': round(top1, 4),
            'is_ood': bool(top1 < floor),
            'percentile_vs_reference_1nn': round(float((best < top1).mean() * 100), 2)}
    (RES / 'ood.json').write_text(json.dumps(ood, indent=2) + '\n')
    print('\n', json.dumps(ood, indent=2), flush=True)


if __name__ == '__main__':
    main()
