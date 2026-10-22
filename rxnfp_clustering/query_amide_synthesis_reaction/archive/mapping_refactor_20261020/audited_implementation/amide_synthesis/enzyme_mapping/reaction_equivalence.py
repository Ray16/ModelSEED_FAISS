"""Conservative, auditable bridges between duplicate ModelSEED equations.

Only explicitly approved bridges are consumed. Every use rechecks both full
equations, including cofactors, stoichiometric copies, charge and stereochemistry.
The corpus convention already strips free H+; no other species is discarded.
Similar fingerprints, names and EC numbers never establish equation equivalence.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from rdkit import Chem


@lru_cache(maxsize=4096)
def molecule_key(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not mol.GetNumAtoms() or any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
        raise ValueError('Invalid or symbolic molecule in reaction bridge')
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def equation_key(smiles):
    if not isinstance(smiles, str) or smiles.count('>>') != 1:
        raise ValueError('A bridge requires a fully specified reaction SMILES')
    sides = tuple(tuple(sorted(molecule_key(m) for m in side.split('.')))
                  for side in smiles.split('>>'))
    if any(not side for side in sides):
        raise ValueError('Empty reaction side in bridge')
    return sides


def equation_hash(smiles):
    return hashlib.sha256(json.dumps(equation_key(smiles), separators=(',', ':')).encode()).hexdigest()


def validate_bridges(project, keep_reactions=None):
    """Fail closed on changed inputs, nonidentical chemistry or malformed IDs."""
    project = Path(project)
    policy_path = project / 'mapping_policy.json'
    if not policy_path.exists():
        return []
    policy = json.loads(policy_path.read_text())
    if not policy.get('reaction_bridges'):
        return []
    path = project / policy['reaction_bridges']
    expected = policy.get('reaction_bridges_sha256')
    if not expected or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('Reaction bridge registry hash mismatch')
    doc = json.loads(path.read_text())
    if doc.get('schema_version') != 1:
        raise ValueError('Unsupported reaction bridge registry')
    corpus = project / 'data/modelseed_master.parquet'
    if hashlib.sha256(corpus.read_bytes()).hexdigest() != doc['corpus_file_sha256']:
        raise ValueError('Reaction bridge corpus snapshot changed; re-audit bridges')
    frame = pd.read_parquet(corpus, columns=['rxn_id', 'rxn_smiles'])
    if frame.rxn_id.duplicated().any():
        raise ValueError('Duplicate corpus identifiers in reaction bridge validation')
    equations = frame.set_index('rxn_id').rxn_smiles.to_dict()
    keep = None if keep_reactions is None else set(keep_reactions)
    valid = []
    for bridge in doc['bridges']:
        target, source = bridge['target_reaction'], bridge['source_reaction']
        if not all(re.fullmatch(r'rxn\d+', x) for x in [target, source]):
            raise ValueError('Invalid ModelSEED bridge identifier')
        if target == source or bridge.get('approved') is not True:
            raise ValueError('Reaction bridge is not explicitly approved')
        if target not in equations or source not in equations:
            raise ValueError('Missing bridge reaction in corpus')
        if equation_key(equations[target]) != equation_key(equations[source]):
            raise ValueError('Reaction bridge chemistry differs: ' + target + ' / ' + source)
        if equation_hash(equations[target]) != bridge['equation_sha256']:
            raise ValueError('Reaction bridge equation hash mismatch')
        if not bridge.get('rhea_ids') or any(not re.fullmatch(r'\d+', str(x)) for x in bridge['rhea_ids']):
            raise ValueError('Bridge must list exact Rhea reaction identifiers')
        evidence_path = (project / bridge['audit_manifest']).resolve()
        if hashlib.sha256(evidence_path.read_bytes()).hexdigest() != bridge['audit_manifest_sha256']:
            raise ValueError('Reaction bridge audit evidence changed')
        if keep is None or target in keep:
            valid.append(bridge)
    return valid


def expand_mapping(by_reaction, evidence, bridges):
    """Preserve source annotation scope; never overwrite a direct target link."""
    added = []
    for bridge in bridges:
        target, source = bridge['target_reaction'], bridge['source_reaction']
        wanted = {'RHEA:' + str(x) for x in bridge['rhea_ids']}
        for accession, sequence in by_reaction.get(source, {}).items():
            if accession in by_reaction.get(target, {}):
                continue
            source_evidence = evidence.get((source, accession))
            if not source_evidence:
                raise ValueError('Missing source evidence for reaction bridge')
            actual = set(source_evidence['mapping_rhea_ids'].split(';'))
            if not actual.intersection(wanted):
                continue
            if hashlib.sha256(sequence.encode()).hexdigest() != source_evidence['sequence_sha256']:
                raise ValueError('Reaction bridge source sequence mismatch')
            by_reaction.setdefault(target, {})[accession] = sequence
            evidence[(target, accession)] = dict(
                source_evidence,
                modelseed_reaction=target,
                source='validated_equivalent_modelseed_reaction',
                mapping_relation='equivalent_modelseed_equation',
                mapping_source_reaction=source,
                mapping_chemical_scope=bridge['chemical_scope'],
                mapping_bridge_audit=bridge['audit_manifest'],
            )
            added.append({'rxn_id': target, 'source_reaction': source, 'uniprot': accession})
    return added
