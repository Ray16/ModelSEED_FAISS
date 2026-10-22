"""Check a revised chemical selection against a historical five and weight changes.

Reference IDs are used only after selection. Agreement is a regression check,
not validation of enzyme activity or proof that the weights are optimal.
"""
import hashlib
import json

import pandas as pd

from amide_synthesis.reaction_search.chemical_selection import load_policy


def validate(reviewed, selected, previous_path, output):
    previous = pd.read_csv(previous_path, sep='\t')
    if len(previous) != 5 or previous.rxn_id.duplicated().any():
        raise ValueError('Historical validation requires five distinct reference reactions')
    references = set(previous.rxn_id)
    chosen = selected.rxn_id.tolist()
    eligible = reviewed[reviewed.shortlist_eligible].copy()
    ranked = eligible.sort_values(['chemical_resemblance','drfp_tanimoto','hybrid_cosine','rxn_id'],
                                 ascending=[False,False,False,True]).drop_duplicates('organic_transformation_group')
    ranks = {rid:i+1 for i,rid in enumerate(ranked.rxn_id)}
    policy = load_policy()
    sensitivity = []
    for component in policy['weights']:
        for factor in [.5, .75, 1.25, 1.5]:
            weights = dict(policy['weights'])
            weights[component] *= factor
            total = sum(weights.values())
            weights = {k:v/total for k,v in weights.items()}
            frame = eligible.copy()
            frame['perturbed'] = sum(frame['chem_'+k+'_credit']*weight for k,weight in weights.items())
            ordered = frame.sort_values(['perturbed','drfp_tanimoto','hybrid_cosine','rxn_id'],
                                        ascending=[False,False,False,True]).drop_duplicates('organic_transformation_group')
            members = ordered.head(len(selected)).rxn_id.tolist()
            sensitivity.append({'changed_component':component, 'factor':factor,
                                'previous_five_retained':len(references.intersection(members)),
                                'selected_reactions':members})
    result = {
        'reference_file':str(previous_path), 'reference_sha256':hashlib.sha256(previous_path.read_bytes()).hexdigest(),
        'reference_ids_used_to_select':False, 'previous_five_retained':len(references.intersection(chosen)),
        'comparison':[{'rxn_id':rid, 'selected':rid in chosen, 'new_chemical_rank':ranks.get(rid)} for rid in previous.rxn_id],
        'newly_selected_reactions':[rid for rid in chosen if rid not in references],
        'weight_sensitivity':sensitivity,
        'limits':'Reference consistency and sensitivity analysis only; weights are declared chemical judgments, not experimentally calibrated activity predictors.'}
    output.write_text(json.dumps(result, indent=2)+'\n')
    return result


def plot_components(selected, stem):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    policy=load_policy()
    fig,ax=plt.subplots(figsize=(8,max(3.5,.43*len(selected)+1.6)),layout='constrained')
    left=np.zeros(len(selected))
    colors=['#0072B2','#56B4E9','#009E73','#E69F00','#CC79A7']
    for (name,weight),color in zip(policy['weights'].items(),colors):
        values=selected['chem_'+name+'_contribution'].to_numpy(float)
        ax.barh(np.arange(len(selected)),values,left=left,label=name.replace('_',' '),color=color)
        left+=values
    ax.set_yticks(np.arange(len(selected)),selected.rxn_id)
    ax.invert_yaxis();ax.set_xlim(0,1.02)
    ax.set_xlabel('Weighted substrate resemblance · heuristic, not activity')
    ax.set_title('Chemical selection after DRFP top-100 retrieval')
    ax.spines[['top','right']].set_visible(False)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.15),ncol=3,frameon=False,fontsize=9)
    for extension in ['png','svg','pdf']:
        fig.savefig(stem.with_suffix('.'+extension),dpi=180,bbox_inches='tight')
    plt.close(fig)
