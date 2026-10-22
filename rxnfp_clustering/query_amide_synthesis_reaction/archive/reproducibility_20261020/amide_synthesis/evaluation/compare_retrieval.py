"""Compare retrieval configurations by the same explicit nylon chemistry rubric.

The binary ECFP4 variants are defined here rather than applying bit Tanimoto to
signed count differences. No experimental activity labels enter the comparison.
The production ranking remains DRFP Tanimoto, hybrid cosine and identifier.
"""
from functools import lru_cache
from pathlib import Path
import datetime
import hashlib
import json

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator

from amide_synthesis.infrastructure.paths import PROJECT
from amide_synthesis.reaction_search.encode_and_search import build_queries, featurise
from nylon_chemical_review import assess

METHODS = {
    'drfp_tanimoto': 'DRFP / Tanimoto',
    'hybrid_cosine': 'ECFP4 difference + DRFP / cosine',
    'gainloss_drfp_tanimoto': 'ECFP4 gain/loss bits + DRFP / Tanimoto',
    'sides_drfp_tanimoto': 'ECFP4 reactant/product bits + DRFP / Tanimoto',
    'signed_raw_extended_tanimoto': 'Raw signed hybrid / extended Tanimoto',
}


def bit_tanimoto(left, right):
    union = (left | right).sum(axis=1)
    if not np.all(union > 0):
        raise ValueError('Empty fingerprint union')
    return (left & right).sum(axis=1) / union


def ranking(frame, method):
    secondary = 'hybrid_cosine' if method == 'drfp_tanimoto' else 'drfp_tanimoto'
    return frame.sort_values([method, secondary, 'rxn_id'], ascending=[False, False, True])


def main():
    source = PROJECT / 'results/modelseed_nylon_top100'
    out = PROJECT / 'results/retrieval_method_comparison'
    out.mkdir(exist_ok=True)
    rows = pd.read_csv(source / 'all_amide_scores.tsv', sep='\t')
    corpus = pd.read_parquet(PROJECT / 'data/modelseed_master.parquet', columns=['rxn_id'])
    indices = corpus.reset_index().set_index('rxn_id')['index'].loc[rows.rxn_id].to_numpy()
    matrix_path = PROJECT / 'embeddings/ecfp4_drfp_master.npy'
    matrix = np.asarray(np.load(matrix_path, mmap_mode='r')[indices], dtype=np.float64)
    names = ['atp_amp_ligase', 'atp_adp_ligase', 'coa_aminolysis', 'direct_condensation']
    queries = build_queries().set_index('query')
    query_matrix = featurise(queries.loc[names].rxn_smiles.tolist()).astype(np.float64)
    query = query_matrix[[names.index(n) for n in rows.target_query]]
    dot = (matrix * query).sum(axis=1)
    cosine = dot / np.linalg.norm(matrix, axis=1) / np.linalg.norm(query, axis=1)
    if not np.allclose(cosine, rows.hybrid_cosine, atol=1e-6):
        raise ValueError('Saved hybrid scores do not match the declared embeddings')
    left = np.hstack([matrix[:, :2048] > 0, matrix[:, :2048] < 0, matrix[:, 2048:] > 0])
    right = np.hstack([query[:, :2048] > 0, query[:, :2048] < 0, query[:, 2048:] > 0])
    rows['gainloss_drfp_tanimoto'] = bit_tanimoto(left, right)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    @lru_cache(maxsize=4096)
    def molecule_bits(smiles):
        return np.asarray(generator.GetFingerprintAsNumPy(Chem.MolFromSmiles(smiles)), dtype=bool)

    def context(smiles):
        return np.hstack([np.logical_or.reduce([molecule_bits(m) for m in side.split('.')])
                          for side in smiles.split('>>')])

    left = np.hstack([np.stack([context(s) for s in rows.rxn_smiles]), matrix[:, 2048:] > 0])
    right = np.hstack([np.stack([context(queries.loc[n, 'rxn_smiles']) for n in rows.target_query]), query[:, 2048:] > 0])
    rows['sides_drfp_tanimoto'] = bit_tanimoto(left, right)
    rows['signed_raw_extended_tanimoto'] = dot / ((matrix * matrix).sum(axis=1)
                                                  + (query * query).sum(axis=1) - dot)
    with rdBase.BlockLogs():
        features = pd.DataFrame([dict(rxn_id=r.rxn_id, **assess(r.rxn_smiles)) for r in rows.itertuples()])
    reviewed = rows.merge(features, on='rxn_id', validate='one_to_one')
    core = reviewed.shortlist_eligible & reviewed.preexisting_amide_bonds.eq(0)
    expected_groups = set(reviewed.loc[core, 'organic_transformation_group'])
    summaries = []
    for method in METHODS:
        ordered = ranking(reviewed, method)
        ordered.to_csv(out / (method + '_ranking.tsv'), sep='\t', index=False)
        for k in [5, 10, 20, 50, 100]:
            subset = ordered.head(k)
            mask = subset.shortlist_eligible & subset.preexisting_amide_bonds.eq(0)
            summaries.append(dict(method=method, cutoff=k, retrieved_entries=len(subset),
                non_preamidated_analogue_groups=subset.loc[mask, 'organic_transformation_group'].nunique(),
                diacid_diamine_entries=int(subset.chemical_tier.eq('diacid–primary-diamine analogue').sum()),
                pre_amidated_entries=int(subset.chemical_tier.eq('pre-amidated intermediate').sum()),
                acyclic_partner_entries=int(subset.acyclic_aliphatic_partners.fillna(False).sum())))
    pd.DataFrame(summaries).to_csv(out / 'chemistry_coverage.tsv', sep='\t', index=False)
    reviewed.to_csv(out / 'all_480_reactions.tsv', sep='\t', index=False)
    production = ranking(reviewed, 'drfp_tanimoto')
    saved = pd.read_csv(source / 'top100_reactions.tsv', sep='\t')
    if production.head(100).rxn_id.tolist() != saved.rxn_id.tolist():
        raise ValueError('Production configuration no longer reproduces the current top 100')
    policy = {
        'status': 'selected_for_production', 'primary_representation': 'DRFP',
        'primary_metric': 'Tanimoto', 'tie_breakers': ['ECFP4-difference + DRFP cosine', 'ModelSEED ID'],
        'final_selection': 'Explicit reacting-group chemistry; no weighted activity score',
        'basis': 'Same 480 eligible reactions, target routes, structures, stoichiometry and cofactors; chemistry rubric applied without using fingerprint scores.',
        'target_features': ['separate aliphatic acid and primary-amine partners',
            'diacid/diamine functionality', 'one newly formed amide', 'retained acid and primary-amine ends',
            'prefer monomers without pre-existing amide bonds; inspect branching, rings and extra heteroatoms'],
        'core_analogue_groups_in_scope': len(expected_groups),
        'current_top100_reproduced': True,
        'decision': 'Retain DRFP primary. Adding ECFP4 to a primary Tanimoto representation did not improve recovery of the target monomer-coupling groups; side-context concatenation missed two groups within 100 entries.',
        'limits': ['Target-specific chemical judgment using the existing rubric, not an independent experimental benchmark.',
            'No universal superiority claim; some high-ranked hits from every method differ substantially from the target.',
            'Signed raw extended Tanimoto is a diagnostic, not ordinary binary Tanimoto.',
            'Extended Tanimoto on unit-normalized hybrid vectors equals cosine/(2-cosine) and therefore preserves cosine ranking.'],
        'generated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    (out / 'production_policy.json').write_text(json.dumps(policy, indent=2) + '\n')
    from amide_synthesis.evaluation.retrieval_comparison_report import render
    render(reviewed, pd.DataFrame(summaries), policy, out)
    inputs = [source / 'all_amide_scores.tsv', source / 'top100_reactions.tsv',
              PROJECT / 'data/modelseed_master.parquet', matrix_path,
              PROJECT / 'nylon_chemical_review.py', Path(__file__),
              PROJECT / 'amide_synthesis/evaluation/retrieval_comparison_report.py',
              PROJECT / 'rank_modelseed_nylon.py',
              PROJECT / 'amide_synthesis/reaction_search/encode_and_search.py']
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    (out / 'manifest.json').write_text(json.dumps({
        'input_sha256': {str(p): digest(p) for p in inputs},
        'output_sha256': {p.name: digest(p) for p in out.iterdir() if p.is_file() and p.name != 'manifest.json'},
    }, indent=2) + '\n')
    print(json.dumps(policy, indent=2))


if __name__ == '__main__':
    main()
