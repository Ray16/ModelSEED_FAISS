"""The single production retrieval policy for the adipate/HMDA target.

Diagnostic configurations belong in evaluation.compare_retrieval. A chemical
review of the retrieved reacting partners supplies the final five-reaction set.
"""
import numpy as np

PRODUCTION_POLICY = {
    'primary_representation': 'DRFP',
    'primary_similarity': 'Tanimoto',
    'tie_breakers': ['ECFP4-difference + DRFP cosine', 'ModelSEED ID'],
    'selection': 'Soft chemical resemblance after DRFP top-100 retrieval',
    'chemical_selection_policy': 'config/chemical_selection_policy.json',
    'decision_evidence': 'results/retrieval_method_comparison/report.html',
    'scope': 'Adipate/HMDA monomer coupling; no universal or activity-prediction claim',
}


def rank_reactions(reactions):
    required = {'rxn_id', 'drfp_tanimoto', 'hybrid_cosine'}
    if not required <= set(reactions.columns):
        raise ValueError('Production retrieval requires reaction ID and both declared scores')
    if reactions.rxn_id.isna().any() or reactions.rxn_id.duplicated().any():
        raise ValueError('Production retrieval requires unique reaction identifiers')
    if not np.isfinite(reactions[['drfp_tanimoto', 'hybrid_cosine']].to_numpy(float)).all():
        raise ValueError('Production retrieval scores must be finite')
    return reactions.sort_values(['drfp_tanimoto', 'hybrid_cosine', 'rxn_id'],
                                 ascending=[False, False, True], kind='stable')
