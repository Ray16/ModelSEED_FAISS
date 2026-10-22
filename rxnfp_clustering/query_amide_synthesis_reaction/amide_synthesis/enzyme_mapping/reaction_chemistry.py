"""Compare complete reaction structures to live Rhea/ChEBI evidence.

Protonation is normalized explicitly. Cofactors, stoichiometric multiplicities,
bond connectivity and all specified stereochemistry are retained. Missing stereo
is reported as compatible but underspecified, never certified as an exact match.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import json

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


@lru_cache(maxsize=1024)
def normalized_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
        raise ValueError('Unparseable/symbolic molecule in reaction audit')
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    mol = rdMolStandardize.Uncharger().uncharge(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return mol


def species(smiles):
    mol = normalized_molecule(smiles)
    # The saved corpus convention omits free H+; record the same convention here.
    if mol.GetNumAtoms() == 1 and mol.GetAtomWithIdx(0).GetAtomicNum() == 1:
        return None
    return mol


def stereo_compatible(left, right):
    """Look for a full graph isomorphism without contradictory specified stereo."""
    for match in right.GetSubstructMatches(left, useChirality=False, uniquify=False, maxMatches=1024):
        conflict = False
        for atom in left.GetAtoms():
            other = right.GetAtomWithIdx(match[atom.GetIdx()])
            if atom.HasProp('_CIPCode') and other.HasProp('_CIPCode'):
                if atom.GetProp('_CIPCode') != other.GetProp('_CIPCode'):
                    conflict = True
                    break
        if conflict:
            continue
        for bond in left.GetBonds():
            other = right.GetBondBetweenAtoms(match[bond.GetBeginAtomIdx()], match[bond.GetEndAtomIdx()])
            specified = {Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ}
            if bond.GetStereo() in specified and other.GetStereo() in specified and bond.GetStereo() != other.GetStereo():
                conflict = True
                break
        if not conflict:
            return True
    return False


def compare_side(local, reference):
    local = [m for s in local if (m := species(s)) is not None]
    reference = [m for s in reference if (m := species(s)) is not None]
    key = lambda m, stereo: Chem.MolToSmiles(m, canonical=True, isomericSmiles=stereo)
    if Counter(key(m, True) for m in local) == Counter(key(m, True) for m in reference):
        return 'exact_after_protonation_normalization'
    if Counter(key(m, False) for m in local) != Counter(key(m, False) for m in reference):
        return 'connectivity_or_stoichiometry_mismatch'
    # Match every stoichiometric copy; reuse of the same reference is forbidden.
    def assign(index, remaining):
        if index == len(local):
            return True
        for j in remaining:
            if key(local[index], False) == key(reference[j], False) and stereo_compatible(local[index], reference[j]):
                if assign(index + 1, remaining - {j}):
                    return True
        return False
    return ('compatible_stereochemistry_underspecified' if assign(0, set(range(len(reference))))
            else 'specified_stereochemistry_conflict')


def rhea_structure(client, rhea_id):
    query = f'''PREFIX rh: <http://rdf.rhea-db.org/>
SELECT ?side ?order ?contains ?participant ?chebi ?name WHERE {{
 rh:{rhea_id} rh:side ?side .
 ?side rh:curatedOrder ?order; ?contains ?participant .
 FILTER(REGEX(STR(?contains), "^http://rdf.rhea-db.org/contains[0-9]+$"))
 ?participant rh:compound ?compound .
 ?compound rh:chebi ?chebi; rh:name ?name .
}}'''
    body, _ = client.get(f'rhea_{rhea_id}_participants.json', 'https://sparql.rhea-db.org/sparql',
                         {'query': query, 'format': 'json'})
    rows = json.loads(body)['results']['bindings']
    if not rows:
        raise ValueError('No structured Rhea participants: ' + str(rhea_id))
    sides = {1: [], 2: []}
    participants = []
    for row in rows:
        coefficient = int(row['contains']['value'].split('contains')[-1])
        chebi_id = row['chebi']['value'].split('CHEBI_')[-1]
        body, _ = client.get(f'chebi_{chebi_id}.json',
                             f'https://www.ebi.ac.uk/chebi/backend/api/public/compound/{chebi_id}/')
        compound = json.loads(body)
        smiles = compound.get('default_structure', {}).get('smiles')
        if not smiles:
            raise ValueError('Missing ChEBI structure for ' + chebi_id)
        order = int(row['order']['value'])
        sides[order].extend([smiles] * coefficient)
        participants.append({'side': order, 'coefficient': coefficient, 'chebi_id': chebi_id,
                             'name': row['name']['value'], 'smiles': smiles})
    return sides, participants


def audit_equation(local_smiles, rhea_sides):
    local = [s.split('.') for s in local_smiles.split('>>')]
    if len(local) != 2:
        raise ValueError('Invalid local reaction')
    comparisons = []
    for orientation, reference in [('forward', [rhea_sides[1], rhea_sides[2]]),
                                   ('reverse', [rhea_sides[2], rhea_sides[1]])]:
        status = [compare_side(l, r) for l, r in zip(local, reference)]
        compatible = all(s.startswith(('exact_', 'compatible_')) for s in status)
        comparisons.append({'orientation': orientation, 'side_status': status, 'compatible': compatible})
    valid = [x for x in comparisons if x['compatible']]
    return {
        'status': ('exact_after_protonation_normalization' if valid and all(
            s.startswith('exact_') for s in valid[0]['side_status']) else
            'compatible_stereochemistry_underspecified' if valid else 'chemical_conflict'),
        'comparisons': comparisons,
        'normalization': 'RDKit Uncharger for protonation only; free H+ omitted; cofactors and stoichiometry retained',
    }
