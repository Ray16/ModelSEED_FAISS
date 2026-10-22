import pandas as pd
import pytest

from nylon_chemical_review import assess, review
from amide_synthesis.reaction_search.chemical_selection import score_features

TARGET = 'O=C(O)CCCCC(=O)O.NCCCCCCN>>O=C(O)CCCCC(=O)NCCCCCCN.O'
AB = 'NCCCCCC(=O)O.NCCCCCC(=O)O>>NCCCCCC(=O)NCCCCCC(=O)O.O'
AROMATIC = 'CC(=O)O.Nc1ccccc1>>CC(=O)Nc1ccccc1.O'


def test_exact_target_has_full_resemblance_and_components_sum():
    result = score_features(assess(TARGET))
    assert result['chemical_resemblance'] == pytest.approx(1)
    assert sum(v for k,v in result.items() if k.endswith('_contribution')) == pytest.approx(1)


@pytest.mark.parametrize('reaction', [AB, AROMATIC, 'O=C(O)C1CCCCC1.CN>>CNC(=O)C1CCCCC1.O'])
def test_monoacid_monoamine_ab_aromatic_and_cyclic_are_not_vetoed(reaction):
    features = assess(reaction)
    assert features['shortlist_eligible']
    assert 0 < score_features(features)['chemical_resemblance'] <= 1


def test_chemical_selection_is_not_the_first_drfp_hit_and_has_no_id_allowlist():
    rows = pd.DataFrame([
        dict(rxn_id='unseen_a', rxn_smiles=AROMATIC, organic_transformation_group='a',
             drfp_tanimoto=.99, hybrid_cosine=.99, name='aromatic'),
        dict(rxn_id='unseen_b', rxn_smiles=TARGET, organic_transformation_group='b',
             drfp_tanimoto=.1, hybrid_cosine=.1, name='target')])
    _, selected = review(rows, count=1)
    assert selected.rxn_id.tolist() == ['unseen_b']
