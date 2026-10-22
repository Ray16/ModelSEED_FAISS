from pathlib import Path
import hashlib
import json
import pytest
from amide_synthesis.mapping_quality import reaction_statements, default_mapping

def test_component_collision_does_not_create_iaa_ligase_reaction():
    # Observed in the actual DJ-1/HchA annotations linked to rxn02221.
    text='CATALYTIC ACTIVITY: Reaction=DNA deglycation; Xref=Rhea:RHEA:57296, Rhea:RHEA-COMP:14857; Evidence={ECO:0000250|UniProtKB:Q99497};'
    assert [r['rhea_id'] for r in reaction_statements(text)]==['57296']

def test_experimental_evidence_belongs_to_its_reaction():
    text=('CATALYTIC ACTIVITY: Reaction=first; Xref=Rhea:RHEA:12345; Evidence={ECO:0000269|PubMed:123}; '
          'CATALYTIC ACTIVITY: Reaction=second; Xref=Rhea:RHEA:67890; Evidence={ECO:0000250|UniProtKB:P00001};')
    rows=reaction_statements(text)
    assert [(r['rhea_id'],r['evidence_tier']) for r in rows]==[('12345','experimental_direct'),('67890','homology_supported_direct')]

def test_component_only_annotation_is_not_a_reaction():
    assert reaction_statements('CATALYTIC ACTIVITY: Xref=Rhea:RHEA-COMP:14857, ChEBI:CHEBI:14857;')==[]

def test_reaction_and_direction_keep_their_own_evidence():
    # Observed in A0A0G2K047: experimental reaction, curator-inferred direction.
    text=('CATALYTIC ACTIVITY: Reaction=acetate activation; Xref=Rhea:RHEA:23176; '
          'Evidence={ECO:0000269|PubMed:28003429}; '
          'PhysiologicalDirection=left-to-right; Xref=Rhea:RHEA:23177; '
          'Evidence={ECO:0000305|PubMed:28003429}; '
          'PhysiologicalDirection=right-to-left; Xref=Rhea:RHEA:23178; '
          'Evidence={ECO:0000250|UniProtKB:P00001};')
    rows=reaction_statements(text)
    assert [(r['rhea_id'],r['evidence_tier']) for r in rows]==[
        ('23176','experimental_direct'),('23177','curated_direct'),
        ('23178','homology_supported_direct')]
    assert 'ECO:0000269' not in rows[1]['evidence']

def test_missing_reaction_evidence_does_not_borrow_direction_evidence():
    text=('CATALYTIC ACTIVITY: Reaction=first; Xref=Rhea:RHEA:12345; '
          'PhysiologicalDirection=left-to-right; Xref=Rhea:RHEA:12346; '
          'Evidence={ECO:0000269|PubMed:123};')
    rows=reaction_statements(text)
    assert rows[0]['evidence']==''
    assert rows[0]['evidence_tier']=='curated_direct'
    assert rows[1]['evidence_tier']=='experimental_direct'

def test_audited_links_refuse_missing_evidence():
    from amide_synthesis.precomputed_links import to_frame
    with pytest.raises(ValueError,match='Missing or invalid audited reaction evidence'):
        to_frame({'rxn00001':{'P00001':'MPEPTIDE'}},require_evidence=True)

def test_policy_rejects_changed_evidence_sidecar(tmp_path):
    p=tmp_path/'direct.tsv';p.write_text('P00001\trxn00001\tMPEPTIDE\n')
    evidence=tmp_path/'evidence.gz';evidence.write_bytes(b'frozen evidence')
    (tmp_path/'mapping_policy.json').write_text(json.dumps({
        'default_links':'direct.tsv','sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
        'evidence':'evidence.gz','evidence_sha256':hashlib.sha256(evidence.read_bytes()).hexdigest()}))
    assert default_mapping(tmp_path)==p
    evidence.write_bytes(b'changed evidence')
    with pytest.raises(ValueError,match='sidecar hash mismatch'):default_mapping(tmp_path)

def test_policy_rejects_changed_mapping_snapshot(tmp_path):
    p=tmp_path/'direct.tsv';p.write_text('P00001\trxn00001\tMPEPTIDE\n')
    (tmp_path/'mapping_policy.json').write_text(json.dumps({'default_links':'direct.tsv','sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))
    assert default_mapping(tmp_path)==p
    p.write_text('changed')
    with pytest.raises(ValueError,match='hash mismatch'):default_mapping(tmp_path)

def test_literal_nylon66_target_receives_full_architecture_credit():
    from amide_synthesis.substrate_distance import score_reaction, nylon66_architecture_score
    target='O=C(O)CCCCC(=O)O.NCCCCCCN>>O=C(O)CCCCC(=O)NCCCCCCN.O'
    old,features=score_reaction(target)
    assert features['aa_bb_pair']
    assert nylon66_architecture_score(old,features)==1.0
    # Removing the second acid should lower the targeted score, not keep a
    # polymerization reward merely because the product contains an amide.
    terminal='CCCCCC(=O)O.NCCCCCCN>>CCCCCC(=O)NCCCCCCN.O'
    value,feat=score_reaction(terminal)
    assert nylon66_architecture_score(value,feat)<1.0
