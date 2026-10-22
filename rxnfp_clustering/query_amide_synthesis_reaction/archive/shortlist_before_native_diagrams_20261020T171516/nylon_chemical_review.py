"""Explicit chemical review of the DRFP retrieval list; no weighted score.

Primary amines are distinguished from secondary amines, hydroxylamines,
amides and guanidines. Amide disconnections establish which nitrogen reacts.
Full equations, including cofactors, are retained in every output and score.
"""
from itertools import combinations
import hashlib
import html
import json

import pandas as pd
from rdkit import Chem

from amide_synthesis.reaction_search.acylation import acylation_cuts, neutral_key
from amide_synthesis.reaction_search.substrate_distance import CARBOXYLIC_ACID

AMIDE = Chem.MolFromSmarts('C(=O)N')
EXPLANATIONS = {
    'rxn05123': ('Glutamate–putrescine ligation',
        'A free diacid couples to a primary diamine; one amide forms and both acid and amine functionality remain.',
        'Glutamate adds an amino substituent and chirality; acid/amine spacers contain 3/4 carbons, versus 4/6 for adipate/HMDA.'),
    'rxn46045': ('Fumarate–diaminopropionate ligation (DdaG)',
        'A free diacid couples to a primary diamine, forming one amide and retaining acid and amine ends.',
        'Fumarate is unsaturated and shorter; diaminopropionate is chiral and carboxylated. Both spacers contain 2 carbons, versus 4/6 in the target.'),
    'rxn21038': ('Citrate–spermidine ligation',
        'A free polycarboxylate acylates a terminal primary amine of an aliphatic polyamine; reactive acid and amine groups remain.',
        'Citrate has three acids and a hydroxyl branch; spermidine contains an additional secondary amine. This is a polyfunctional, potentially branching analogue.'),
    'rxn42510': ('Citrate–D-ornithine ligation',
        'A free polycarboxylate couples to one of two primary amines on a small aliphatic acceptor and retains reactive groups.',
        'Citrate has an extra acid/hydroxyl branch; D-ornithine is chiral and carboxylated, with a shorter 4-carbon amine spacer.'),
    'rxn47618': ('Citrate–diaminopropionate ligation',
        'A free polycarboxylate couples to a primary diamine in one amide-forming step, retaining acid and amine groups.',
        'Citrate is branched and tricarboxylic; the acceptor is a short, chiral amino acid with a 2-carbon amine spacer. rxn54616 encodes the same transformation.'),
}


def primary_amines(mol):
    """Carbon-bound NH2/NH3+; exclude N–O, amidine, amide and aromatic N."""
    result = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 7 or atom.GetIsAromatic() or atom.GetTotalNumHs() < 2:
            continue
        neighbors = [a for a in atom.GetNeighbors() if a.GetAtomicNum() != 1]
        if len(neighbors) != 1 or neighbors[0].GetAtomicNum() != 6:
            continue
        carbon = neighbors[0]
        if carbon.GetIsAromatic() or any(b.GetBondType() != Chem.BondType.SINGLE for b in carbon.GetBonds()):
            continue
        result.append(atom.GetIdx())
    return result


def acid_carbons(mol):
    return sorted({m[0] for m in mol.GetSubstructMatches(CARBOXYLIC_ACID)})


def span(mol, anchors):
    paths = [Chem.GetShortestPath(mol, a, b)[1:-1] for a, b in combinations(anchors, 2)]
    if not paths: return None, None
    path = max(paths, key=lambda p: sum(mol.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in p))
    carbon = sum(mol.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in path)
    return carbon, len(path) - carbon


def compact_partner(mol):
    return not any(a.IsInRing() or a.GetIsAromatic() for a in mol.GetAtoms())


def assess(smiles):
    reactants, products = [[Chem.MolFromSmiles(s) for s in side.split('.')] for side in smiles.split('>>')]
    reactant_keys = {neutral_key(m): m for m in reactants}
    product_keys = {neutral_key(m) for m in products}
    new_amides = sum(len(m.GetSubstructMatches(AMIDE)) for m in products) - sum(len(m.GetSubstructMatches(AMIDE)) for m in reactants)
    events = []
    for product in products:
        for piece, nitrogen, acyl in acylation_cuts(product):
            key = neutral_key(piece)
            if key not in reactant_keys or key in product_keys or acyl is None:
                continue
            acceptor = reactant_keys[key]
            reacts_primary = nitrogen in primary_amines(piece)
            donor_key = neutral_key(acyl)
            free_donor = donor_key in reactant_keys
            donor = reactant_keys[donor_key] if free_donor else acyl
            n_primary = len(primary_amines(acceptor))
            n_acids = len(acid_carbons(acyl))
            retains_amine = bool(primary_amines(product))
            retains_acid = bool(acid_carbons(product))
            separate = key != donor_key
            small = compact_partner(acceptor) and compact_partner(donor)
            preexisting_amides = len(acceptor.GetSubstructMatches(AMIDE)) + len(donor.GetSubstructMatches(AMIDE))
            qualifies = (reacts_primary and free_donor and separate and n_primary >= 2 and n_acids >= 2
                         and retains_amine and retains_acid and new_amides == 1 and small)
            tier = ('pre-amidated intermediate' if qualifies and preexisting_amides else
                    'diacid–primary-diamine analogue' if qualifies and n_acids == 2 and n_primary == 2 else
                    'polyfunctional analogue' if qualifies else 'other amidation')
            amine_span, amine_hetero = span(acceptor, primary_amines(acceptor))
            acid_span, acid_hetero = span(acyl, acid_carbons(acyl))
            events.append(dict(chemical_tier=tier, shortlist_eligible=qualifies,
                reacts_at_primary_amine=reacts_primary, free_acid_donor=free_donor,
                separate_donor_acceptor=separate, primary_amines_on_acceptor=n_primary,
                carboxyl_groups_on_donor=n_acids, acyclic_aliphatic_partners=small,
                preexisting_amide_bonds=preexisting_amides,
                product_retains_primary_amine=retains_amine, product_retains_carboxyl=retains_acid,
                new_amide_bonds=new_amides, acid_spacer_carbons=acid_span,
                amine_spacer_carbons=amine_span, amine_spacer_noncarbon_atoms=amine_hetero,
                acceptor_heavy_atoms=acceptor.GetNumHeavyAtoms(), donor_heavy_atoms=donor.GetNumHeavyAtoms(),
                reacting_acceptor_smiles=Chem.MolToSmiles(acceptor),
                reacting_acid_smiles=Chem.MolToSmiles(acyl)))
    if not events:
        return dict(chemical_tier='unresolved amide disconnection', shortlist_eligible=False,
                    new_amide_bonds=new_amides)
    tier_order = {'diacid–primary-diamine analogue': 0, 'polyfunctional analogue': 1,
                  'pre-amidated intermediate': 2, 'other amidation': 3}
    return min(events, key=lambda e: (tier_order[e['chemical_tier']], not e['reacts_at_primary_amine'],
                                     not e['free_acid_donor'], e['acceptor_heavy_atoms'] + e['donor_heavy_atoms']))


def review(top100):
    features = pd.DataFrame([dict(rxn_id=r.rxn_id, **assess(r.rxn_smiles)) for r in top100.itertuples()])
    reviewed = top100.merge(features, on='rxn_id', validate='one_to_one')
    pool = reviewed[reviewed.shortlist_eligible].copy()
    pool['tier_order'] = pool.chemical_tier.map({'diacid–primary-diamine analogue': 0,
                                               'polyfunctional analogue': 1, 'pre-amidated intermediate': 2})
    pool = pool.sort_values(['tier_order', 'rxn_id']).drop_duplicates('organic_transformation_group')
    if len(pool) < 5:
        raise RuntimeError(f'Chemical rubric produced only {len(pool)} distinct candidates; review before publishing a five-entry shortlist')
    shortlist = pool.head(5).drop(columns='tier_order').copy()
    for name in ['display_title', 'why_similar', 'key_difference']: shortlist[name] = ''
    for i, row in shortlist.iterrows():
        if row.rxn_id not in EXPLANATIONS: raise RuntimeError('Unreviewed chemical candidate: ' + row.rxn_id)
        for field, value in zip(['display_title', 'why_similar', 'key_difference'], EXPLANATIONS[row.rxn_id]):
            shortlist.loc[i, field] = value
    # This is a priority group selection, not a new numerical ranking.
    return reviewed, shortlist.reset_index(drop=True)


def render(out, reviewed, shortlist):
    esc = lambda x: html.escape(str(x))
    rows = '<tr><th>ModelSEED</th><th>Reaction</th><th>Chemical fit</th><th>Why similar</th><th>Main difference</th></tr>'
    for r in shortlist.itertuples():
        values = [r.rxn_id, r.display_title, r.chemical_tier, r.why_similar, r.key_difference]
        rows += '<tr>' + ''.join('<td>' + esc(x) + '</td>' for x in values) + '</tr>'
    table = '<tr><th>Retrieval rank</th><th>ModelSEED</th><th>DRFP</th><th>Chemical category</th><th>Native reaction</th></tr>'
    for r in reviewed.itertuples():
        table += '<tr>' + ''.join('<td>' + esc(x) + '</td>' for x in [r.amide_rank, r.rxn_id,
            f'{r.drfp_tanimoto:.3f}', r.chemical_tier, r.definition]) + '</tr>'
    rubric = {
        'retrieval': 'The DRFP-ranked top 100 remains unchanged; 100 of 480 balanced amide-forming candidates.',
        'selection': ['New amide forms at a carbon-bound primary NH2/NH3+, not hydroxylamine or guanidine.',
                      'A separate free acid donor has at least two carboxyl groups; acceptor has at least two primary amines.',
                      'One new amide; product retains primary-amine and carboxyl functionality.',
                      'Both organic partners are acyclic and nonaromatic; their sizes and carbon spacers are reported explicitly.',
                      'Exact diacid/diamine architecture is a closer analogue than higher polyfunctionality.',
                      'Simple monomers are prioritized over substrates already containing amide-linked intermediates; no weighted score or size cutoff is used.',
                      'ATP-bookkeeping duplicate transformations are grouped for selection only.'],
        'interpretation': 'An explicit, target-specific qualitative review, not a calibrated activity predictor, weighted similarity score or unique universal ordering. Five candidates are not ranked within their chemical categories.',
        'scope_limit': 'Limited to the DRFP top 100; useful reactions outside that retrieval cut can still be missed. Category priorities are chemical judgments for this target, not experimentally calibrated activity thresholds.',
        'cofactors': 'Retained in every full reaction and embedding score. Only reacting partners are examined separately for the chemical review.',
        'feature_convention': 'Spacer lengths count carbon atoms between anchor atoms; any intervening noncarbon atoms are recorded separately. Nylon target: acid spacer 4, amine spacer 6.',
    }
    (out / 'chemical_review_policy.json').write_text(json.dumps(rubric, indent=2) + '\n')
    policy = ''.join('<li>' + esc(x) + '</li>' for x in rubric['selection'])
    page = ('<!doctype html><meta charset="utf-8"><title>Nylon-6,6 chemical shortlist</title>'
        '<style>body{font:16px/1.5 system-ui;max-width:1300px;margin:25px auto;padding:0 15px}img{width:100%;max-width:850px}'
        'table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #ddd;vertical-align:top;text-align:left}th{background:#eee}</style>'
        '<h1>Five chemistry-led candidates for nylon-6,6 diad formation</h1>'
        '<p>The DRFP top 100 is the retrieval list. These five are selected by explicit reacting-group chemistry; cofactors remain included.</p>'
        '<img src="top5_similarity_heatmap.svg"><table>' + rows + '</table>'
        '<p><a href="top5_reactions.tsv">Chemical shortlist TSV</a> · <a href="top100_reactions.tsv">DRFP top 100 TSV</a> · '
        '<a href="chemical_review.tsv">Chemical review of all 100</a> · <a href="drfp_retrieval_report.html">Original retrieval report</a></p>'
        '<details><summary>Selection criteria and limits</summary><ul>' + policy + '</ul><p>' + esc(rubric['interpretation']) +
        '</p><p>' + esc(rubric['scope_limit']) + '</p><p>Embedding scores are supporting evidence; compare rankings within a method, not numerical scales across methods.</p></details>'
        '<details><summary>Review all 100 retrieved reactions</summary><table>' + table + '</table></details>')
    (out / 'report.html').write_text(page)
