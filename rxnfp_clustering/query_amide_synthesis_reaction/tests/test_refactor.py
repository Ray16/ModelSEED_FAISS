"""Behavioral contracts for the phase layout and shared campaign implementation."""
import importlib
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from amide_synthesis import pipeline
from amide_synthesis.stages import STAGES, RULES, dependencies
from amide_synthesis.infrastructure import freshness
from amide_synthesis.structure_modeling import campaign
from amide_synthesis.structure_modeling.poses import summary


@pytest.mark.parametrize('old,new', [
    ('mapping_quality','enzyme_mapping.mapping_quality'),
    ('candidate_table','candidate_selection.candidate_table'),
    ('paths','infrastructure.paths'),
    ('freshness','infrastructure.freshness'),
    ('substrate_distance','reaction_search.substrate_distance'),
])
def test_legacy_and_canonical_imports_share_module_state(old, new):
    legacy=importlib.import_module('amide_synthesis.'+old)
    canonical=importlib.import_module('amide_synthesis.'+new)
    assert legacy is canonical
    assert canonical.__spec__.name=='amide_synthesis.'+new
    with patch.object(legacy, '_refactor_probe', 42, create=True):
        assert canonical._refactor_probe==42


def test_stages_have_one_registry_and_preserve_execution_dependencies():
    assert freshness.RULES is RULES
    positions={s.name:i for i,s in enumerate(STAGES)}
    assert len(positions)==len(STAGES)
    for stage in STAGES:
        assert stage.source.is_file()
        for upstream in dependencies(stage.name):
            assert positions[upstream]<positions[stage.name], (upstream,stage.name)


def test_phase_selection_keeps_established_stage_order():
    selected=pipeline.select_stages(phases=['reaction-search'])
    assert [s.name for s in selected]==['corpus','search','screen']
    assert all(s.phase=='structure-modeling' for s in pipeline.select_stages(phases=['structure-modeling']))


def test_description_does_not_create_a_work_tree_or_fetch_data(tmp_path):
    import os
    work=tmp_path/'not-created'
    env=dict(os.environ,AMIDE_SYNTHESIS_WORKDIR=str(work),AMIDE_SYNTHESIS_OFFLINE='1')
    result=subprocess.run([sys.executable,'-m','amide_synthesis','describe','--format','json'],
                          cwd=pipeline.PROJECT,env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert len(json.loads(result.stdout)['stages'])==len(STAGES)
    assert not work.exists()


def test_shared_geometry_matches_legacy_weighting_and_boundary_rules():
    distances=np.full(50,2.0);angles=np.full(50,160.0)
    result=summary(distances,angles)
    assert result['A_deg']==pytest.approx(160)
    assert result['Dh_A']==pytest.approx(2)
    assert result['near_attack_fraction']==1
    assert summary(np.full(50,4.0),angles)['near_attack_fraction']==0
    assert summary(distances,np.full(50,150.0))['near_attack_fraction']==0
    assert 'near_attack_fraction' not in summary(distances,angles,include_near_attack=False)


@pytest.mark.parametrize('distances,angles',[
    (np.ones(49),np.ones(49)),(np.zeros(50),np.ones(50)),
    (np.ones(50),np.full(50,np.nan)),
])
def test_shared_geometry_refuses_incomplete_or_invalid_ensembles(distances,angles):
    with pytest.raises(ValueError):summary(distances,angles)


def selection(tmp_path, query='MPEPTIDE', mg='2'):
    msa=tmp_path/'msa.csv';msa.write_text('sequence\n'+query+'\nMPEPTIDE\n')
    selected=tmp_path/'selection.tsv'
    selected.write_text('accession\tsequence\tmg_count\tphosphate\tmsa\nP00001\tMPEPTIDE\t'+mg+'\tPG\tmsa.csv\n')
    return selected


def test_preparation_records_explicit_inputs_without_launching_work(tmp_path):
    source=selection(tmp_path);run=tmp_path/'run'
    with patch('subprocess.Popen',side_effect=AssertionError('Preparation must not launch a process')):
        manifest=campaign.prepare(source,run)
    item=manifest['inputs'][0]
    assert item['phosphate']=='PG' and item['leaving_oxygen']=='O3B' and item['mg_count']==2
    assert item['input_sha256']==campaign.digest(item['input'])
    assert manifest['expected_models']==50
    with pytest.raises(ValueError,match='frozen runs cannot be overwritten'):
        campaign.prepare(source,run)


def test_preparation_rejects_inconsistent_msa_before_writing_run(tmp_path):
    source=selection(tmp_path,query='MOTHER');run=tmp_path/'run'
    with pytest.raises(ValueError,match='MSA query differs'):
        campaign.prepare(source,run)
    assert not run.exists()

@pytest.mark.parametrize('default',['',None,False])
def test_shared_optional_columns_preserve_index_and_missing_defaults(default):
    import pandas as pd
    from amide_synthesis.infrastructure.table_utils import column, nullable_column
    frame=pd.DataFrame({'present':[1,2]},index=['b','a'])
    assert column(frame,'present') is frame['present']
    result=column(frame,'absent',default)
    assert result.index.tolist()==['b','a']
    assert result.tolist()==[default,default]
    assert nullable_column(frame,'absent').isna().all()
