import hashlib
import json

import pandas as pd
import pytest

from amide_synthesis.enzyme_mapping.reaction_equivalence import equation_key, equation_hash, validate_bridges, expand_mapping
from amide_synthesis.enzyme_mapping.reaction_chemistry import compare_side
from amide_synthesis.enzyme_mapping.uniprot_snapshot import reaction_annotations


def test_full_equivalence_preserves_cofactors_multiplicity_and_stereo():
    assert equation_key('CCO.N>>CCN.O') == equation_key('N.OCC>>O.NCC')
    assert equation_key('CCO.N>>CCN.O') != equation_key('CCO.N.N>>CCN.O')
    assert equation_key('N[C@H](C)C(=O)O>>N') != equation_key('N[C@@H](C)C(=O)O>>N')
    assert equation_key('CCO.N>>CCN.O') != equation_key('CCN.O>>CCO.N')


def test_chemical_check_distinguishes_missing_stereo_from_wrong_stereo():
    l = 'N[C@H](C)C(=O)O'
    d = 'N[C@@H](C)C(=O)O'
    unspecified = 'NC(C)C(=O)O'
    assert compare_side([l], [d]) == 'specified_stereochemistry_conflict'
    assert compare_side([unspecified], [l]) == 'compatible_stereochemistry_underspecified'
    assert compare_side([l], ['N[C@H](C)C(=O)[O-]']).startswith('exact_')
    assert compare_side(['CCO', 'O'], ['CCO']) == 'connectivity_or_stoichiometry_mismatch'


def test_json_evidence_does_not_cross_reactions_directions_or_component_namespaces():
    record = {'comments': [{'commentType': 'CATALYTIC ACTIVITY',
        'reaction': {'reactionCrossReferences': [{'database': 'Rhea', 'id': 'RHEA:10000'},
                                                {'database': 'Rhea', 'id': 'RHEA-COMP:10002'}],
                     'evidences': [{'evidenceCode': 'ECO:0000269'}]},
        'physiologicalDirections': [{'reactionCrossReferences': [{'database': 'Rhea', 'id': 'RHEA:10001'}],
                                    'evidences': [{'evidenceCode': 'ECO:0000250'}]}]}]}
    rows = reaction_annotations(record, ['10000', '10001', '10002'])
    assert [(r['rhea_id'], r['evidence_tier']) for r in rows] == [
        ('10000', 'experimental_direct'), ('10001', 'homology_supported_direct')]


def test_bridges_keep_source_evidence_and_do_not_replace_direct_links():
    digest = hashlib.sha256(b'MPEPTIDE').hexdigest()
    members = {'rxn2': {'P1': 'MPEPTIDE'}, 'rxn1': {'P2': 'OTHER'}}
    evidence = {('rxn2', 'P1'): {'mapping_rhea_ids': 'RHEA:10000', 'evidence_tier': 'experimental_direct',
                               'sequence_sha256': digest, 'evidence': 'source experiment'}}
    bridge = {'target_reaction': 'rxn1', 'source_reaction': 'rxn2', 'rhea_ids': ['10000'],
              'chemical_scope': 'missing stereo recorded', 'audit_manifest': 'audit.json'}
    added = expand_mapping(members, evidence, [bridge])
    assert len(added) == 1 and members['rxn1']['P2'] == 'OTHER'
    assert evidence[('rxn1', 'P1')]['mapping_source_reaction'] == 'rxn2'
    assert evidence[('rxn1', 'P1')]['evidence'] == 'source experiment'
    assert expand_mapping(members, evidence, [bridge]) == []
    evidence[('rxn2', 'P1')]['sequence_sha256'] = 'bad'
    del members['rxn1']['P1']
    with pytest.raises(ValueError, match='sequence mismatch'):
        expand_mapping(members, evidence, [bridge])


def test_bridge_refuses_changed_equation_or_unapproved_registry(tmp_path):
    (tmp_path / 'data').mkdir()
    corpus = tmp_path / 'data/modelseed_master.parquet'
    pd.DataFrame({'rxn_id': ['rxn1', 'rxn2'], 'rxn_smiles': ['CCO.N>>CCN.O']*2}).to_parquet(corpus)
    audit = tmp_path / 'audit.json'
    audit.write_text('{"status":"complete","output_sha256":{}}')
    bridge = {'target_reaction': 'rxn1', 'source_reaction': 'rxn2', 'rhea_ids': ['10000'],
              'approved': True, 'equation_sha256': equation_hash('CCO.N>>CCN.O'),
              'audit_manifest': 'audit.json', 'audit_manifest_sha256': hashlib.sha256(audit.read_bytes()).hexdigest()}
    registry = tmp_path / 'bridges.json'
    registry.write_text(json.dumps({'schema_version': 1, 'bridges': [bridge],
                                   'corpus_file_sha256': hashlib.sha256(corpus.read_bytes()).hexdigest()}))
    policy = tmp_path / 'mapping_policy.json'
    policy.write_text(json.dumps({'reaction_bridges': 'bridges.json',
                                 'reaction_bridges_sha256': hashlib.sha256(registry.read_bytes()).hexdigest()}))
    assert len(validate_bridges(tmp_path)) == 1
    audit.write_text('changed evidence')
    with pytest.raises(ValueError, match='evidence changed'):
        validate_bridges(tmp_path)
    pd.DataFrame({'rxn_id': ['rxn1', 'rxn2'], 'rxn_smiles': ['CCO.N>>CCN.O', 'CCO.N.N>>CCN.O']}).to_parquet(corpus)
    with pytest.raises(ValueError, match='corpus snapshot changed'):
        validate_bridges(tmp_path)
