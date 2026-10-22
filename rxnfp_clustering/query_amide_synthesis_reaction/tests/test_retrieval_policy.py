import numpy as np
import pandas as pd
import pytest
from amide_synthesis.reaction_search.retrieval_policy import rank_reactions
from amide_synthesis.evaluation.compare_retrieval import bit_tanimoto


def test_primary_similarity_controls_production_order_and_hybrid_breaks_ties():
    data = pd.DataFrame({'rxn_id':['rxn3','rxn2','rxn1','rxn4'],
        'drfp_tanimoto':[.5,.5,.5,.6], 'hybrid_cosine':[.8,.9,.9,.1],
        'enzyme_count':[999,9999,0,0]})
    assert rank_reactions(data).rxn_id.tolist()==['rxn4','rxn1','rxn2','rxn3']


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_nonfinite_scores_cannot_silently_enter_production(bad):
    with pytest.raises(ValueError, match='finite'):
        rank_reactions(pd.DataFrame({'rxn_id':['rxn1'],'drfp_tanimoto':[bad],'hybrid_cosine':[.8]}))


def test_duplicate_reaction_identifiers_fail():
    with pytest.raises(ValueError, match='unique'):
        rank_reactions(pd.DataFrame({'rxn_id':['rxn1','rxn1'],
            'drfp_tanimoto':[.5,.6],'hybrid_cosine':[.7,.8]}))


def test_binary_tanimoto_counts_union_not_feature_magnitude():
    left=np.array([[1,0,1,0]],dtype=bool);right=np.array([[1,1,0,0]],dtype=bool)
    assert np.isclose(bit_tanimoto(left,right)[0],1/3)
