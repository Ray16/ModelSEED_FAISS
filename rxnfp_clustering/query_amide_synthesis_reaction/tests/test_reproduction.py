import hashlib

import pandas as pd
import pytest

import nylon_chemical_review as chemistry
from amide_synthesis.reproduce import compare_table, verify_hashes


def candidates():
    # Same qualifying architecture with independent group keys for ordering tests.
    return pd.DataFrame([{'rxn_id': 'new' + str(i), 'name': 'Unknown reaction ' + str(i),
                          'rxn_smiles': 'O=C(O)CCCCC(=O)O.NCCCCCCN>>O=C(O)CCCCC(=O)NCCCCCCN.O',
                          'organic_transformation_group': str(i)} for i in range(6)])


def test_final_selection_requires_no_curated_reaction_ids(monkeypatch):
    monkeypatch.setattr(chemistry, 'EXPLANATIONS', {})
    _, selected = chemistry.review(candidates().sample(frac=1, random_state=42), count=5)
    assert selected.rxn_id.tolist() == ['new0', 'new1', 'new2', 'new3', 'new4']
    assert set(selected.explanation_source) == {'computed_from_structure'}
    assert selected.why_similar.str.contains('net new amide').all()


def test_display_annotations_do_not_change_selection(monkeypatch):
    _, before = chemistry.review(candidates(), count=5)
    monkeypatch.setattr(chemistry, 'EXPLANATIONS', {'new5': ('title', 'why', 'difference')})
    _, after = chemistry.review(candidates().iloc[::-1], count=5)
    assert before.rxn_id.tolist() == after.rxn_id.tolist()


def test_replay_stops_on_changed_or_missing_inputs(tmp_path):
    p = tmp_path / 'input'
    p.write_bytes(b'pinned')
    expected = {'input': hashlib.sha256(p.read_bytes()).hexdigest()}
    verify_hashes(tmp_path, expected)
    p.write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        verify_hashes(tmp_path, expected)
    p.unlink()
    with pytest.raises(ValueError, match='missing'):
        verify_hashes(tmp_path, expected)


def test_scientific_regression_detects_different_protein_even_at_same_count(tmp_path):
    old, new = tmp_path / 'old.tsv', tmp_path / 'new.tsv'
    pd.DataFrame([{'rxn_id': 'rxn1', 'accession': 'A'}]).to_csv(old, sep='\t', index=False)
    pd.DataFrame([{'rxn_id': 'rxn1', 'accession': 'B'}]).to_csv(new, sep='\t', index=False)
    with pytest.raises(AssertionError):
        compare_table(new, old, keys=['rxn_id', 'accession'])
