import json
import pandas as pd
import pytest
from amide_synthesis import freshness as fr
from amide_synthesis import precomputed_links as pl
from amide_synthesis import master_table as mt

def test_uncapped_mapping_preserves_more_than_two_hundred_proteins():
    frame=pl.to_frame({'rxn1':{f'P{i:05}':'M' for i in range(220)}})
    assert len(frame)==220

def test_explicit_cap_prefers_experimental_evidence_over_accession():
    members={'rxn1':{'A00001':'M','Z00001':'M'}}
    # Hash is deliberately required so metadata cannot attach to a different sequence.
    import hashlib
    digest=hashlib.sha256(b'M').hexdigest()
    ev={('rxn1','A00001'):{'evidence_tier':'computational_direct','sequence_sha256':digest},
        ('rxn1','Z00001'):{'evidence_tier':'experimental_direct','sequence_sha256':digest,'mapping_rhea_ids':'RHEA:1','evidence':'paper'}}
    row=pl.to_frame(members,cap=1,evidence=ev).iloc[0]
    assert row.uniprot=='Z00001' and row.evidence_strength=='experimental_direct'
    assert row.mapping_evidence=='paper'

def test_mapping_metadata_rejects_sequence_mismatch():
    with pytest.raises(ValueError,match='sequence mismatch'):
        pl.to_frame({'rxn1':{'P00001':'M'}},evidence={('rxn1','P00001'):{'evidence_tier':'experimental_direct','sequence_sha256':'wrong'}})

def test_stronger_mapping_can_replace_an_earlier_ec_route():
    base=pd.DataFrame([{'uniprot':'P00001','evidence_strength':'activity_ec','substrate_distance':.7,'sequence':'M','protein_name':'ligase'}])
    mapped=pd.DataFrame([{'uniprot':'P00001','evidence_strength':'experimental_direct','discovery_route':'precomputed_link','substrate_distance_measured':.7,'sequence':'M','mapping_evidence':'PubMed:1'}])
    table=mt.build(base,pd.DataFrame(),structural=pd.DataFrame(columns=['uniprot','struct_S']),extra_sources=[mapped])
    assert len(table)==1 and table.iloc[0].evidence_strength=='experimental_direct'
    assert table.iloc[0].mapping_evidence=='PubMed:1'

def test_chemistry_priority_cannot_erase_experimental_native_mapping():
    base=pd.DataFrame([{'uniprot':'P00001','evidence_strength':'activity_ec','substrate_distance':.99,'sequence':'M','protein_name':'ligase'}])
    mapped=pd.DataFrame([{'uniprot':'P00001','evidence_strength':'experimental_direct','discovery_route':'precomputed_link','substrate_distance_measured':.1,'sequence':'M','mapping_evidence':'native experiment'}])
    table=mt.build(base,pd.DataFrame(),structural=pd.DataFrame(columns=['uniprot','struct_S']),extra_sources=[mapped])
    assert table.iloc[0].evidence_strength=='experimental_direct'

def test_changed_inputs_and_missing_outputs_invalidate_record(tmp_path):
    # Native stage has small, independent inputs, so the test exercises the real
    # fingerprint contract without constructing the entire scientific pipeline.
    for name in fr.RULES['native'][0]+fr.RULES['native'][1]:(tmp_path/name).write_text(name)
    fr.record('native',tmp_path,tmp_path)
    assert fr.status('native',tmp_path,tmp_path)=='current'
    (tmp_path/'enzyme_annotations.tsv').write_text('new source')
    assert fr.status('native',tmp_path,tmp_path)=='stale_inputs'
    fr.record('native',tmp_path,tmp_path)
    (tmp_path/'native_substrates.tsv').unlink()
    assert fr.status('native',tmp_path,tmp_path)=='stale_outputs'

def test_consumer_refuses_untracked_mapping_results(tmp_path):
    with pytest.raises(fr.StaleArtifactError,match='links:untracked'):
        fr.require_inputs('master',tmp_path,tmp_path)

def test_staleness_propagates_through_unchanged_intermediate_files(tmp_path):
    stages=['table','families','links','neighbours','master','enrich']
    for stage in stages:
        for name in fr.RULES[stage][0]+fr.RULES[stage][1]:
            if not (tmp_path/name).exists():(tmp_path/name).write_text(name)
    for stage in stages:fr.record(stage,tmp_path,tmp_path)
    fr.require_inputs('enrich',tmp_path,tmp_path)
    # The master and enrichment output bytes have not changed, but a source
    # several generations upstream has. A consumer must still refuse them.
    (tmp_path/'amide_forming_neighbours.tsv').write_text('revised reaction screen')
    with pytest.raises(fr.StaleArtifactError,match='stale_inputs'):
        fr.require_inputs('enrich',tmp_path,tmp_path)

def test_changed_selection_settings_invalidate_cached_links(tmp_path,monkeypatch):
    for name in fr.RULES['links'][0]+fr.RULES['links'][1]:(tmp_path/name).write_text(name)
    fr.record('links',tmp_path,tmp_path)
    monkeypatch.setenv('MAX_LINKS_PER_REACTION','5')
    assert fr.status('links',tmp_path,tmp_path)=='stale_inputs'

def test_parallel_producers_preserve_both_registry_entries(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    for stage in ['links','table']:
        for name in fr.RULES[stage][0]+fr.RULES[stage][1]:(tmp_path/name).write_text(name)
    with ThreadPoolExecutor(2) as pool:
        list(pool.map(lambda s:fr.record(s,tmp_path,tmp_path),['links','table']))
    assert set(fr.state(tmp_path)['stages'])=={'links','table'}
