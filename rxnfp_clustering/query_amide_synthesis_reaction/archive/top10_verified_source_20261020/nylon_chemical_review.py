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
    'rxn22343': ('UDP-peptide–D-lysine ligation (MurE)',
        'A free carboxyl on a D-glutamyl-containing donor couples to a primary amine of D-lysine, leaving acid and amine functionality.',
        'The donor carries a large UDP-linked peptide scaffold. Rhea contains a symbolic participant, so full structural equivalence remains qualified.'),
    'rxn42395': ('Staphyloferrin B formation (SbnC)',
        'A free diacid, 2-oxoglutarate, couples to a primary amine on a citrate-derived intermediate, retaining reactive groups.',
        'The amine partner is a large, branched, pre-amidated intermediate rather than a simple diamine; citrate stereochemistry is underspecified locally.'),
    'rxn09897': ('Glycylglycine formation',
        'Two AB amino-acid monomers form one amide while retaining a free primary amine and carboxyl group.',
        'Glycine has a much shorter carbon spacer than adipate/HMDA. BacD/YwfE is a related-reaction candidate, not an established exact Gly+Gly assignment.'),
    'rxn45007': ('Citryl-intermediate–diaminopropionate ligation (SbnF)',
        'A free carboxyl on a citrate-derived donor couples to a primary amine of diaminopropionate and retains acid/amine functionality.',
        'The donor already contains an amide; both partners are branched or chiral, and the local citrate stereochemistry is underspecified.'),
    'rxn21036': ('Petrobactin-intermediate ligation (AsbB)',
        'A citrate-derived free acid couples to a primary amine on spermidine, retaining acid and amine groups.',
        'The donor carries a dihydroxybenzoyl aromatic group and existing amides. These reduce literal substrate resemblance without excluding the enzyme.'),
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
        # Anilines are primary amines too. Exclude amide/amidine/cyanamide
        # nitrogens through C=heteroatom bonds, not aromatic carbon attachment.
        if any(b.GetBondType() in {Chem.BondType.DOUBLE, Chem.BondType.TRIPLE}
               and b.GetOtherAtom(carbon).GetAtomicNum() != 6 for b in carbon.GetBonds()):
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


def assess(smiles, strict=False):
    reactants, products = [[Chem.MolFromSmiles(s) for s in side.split('.')] for side in smiles.split('>>')]
    reactant_keys = {neutral_key(m): m for m in reactants}
    from collections import Counter
    multiplicity = Counter(neutral_key(m) for m in reactants)
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
            legacy_qualifies = (reacts_primary and free_donor and separate and n_primary >= 2 and n_acids >= 2
                         and retains_amine and retains_acid and new_amides == 1 and small)
            qualifies = legacy_qualifies if strict else (reacts_primary and free_donor and n_primary >= 1 and n_acids >= 1
                and new_amides >= 1 and (separate or multiplicity[key] >= 2))
            tier = ('pre-amidated intermediate' if qualifies and preexisting_amides else
                    'diacid–primary-diamine analogue' if qualifies and n_acids == 2 and n_primary == 2 else
                    'polyfunctional analogue' if qualifies and n_acids >= 2 and n_primary >= 2 else
                    'AB self-condensation' if qualifies and not separate else
                    'free-acid primary-amine amidation' if qualifies else 'other amidation')
            amine_span, amine_hetero = span(acceptor, primary_amines(acceptor))
            acid_span, acid_hetero = span(acyl, acid_carbons(acyl))
            events.append(dict(chemical_tier=tier, shortlist_eligible=qualifies,
                reacts_at_primary_amine=reacts_primary, free_acid_donor=free_donor,
                separate_donor_acceptor=separate, self_condensation=not separate, legacy_monomer_eligible=legacy_qualifies,
                primary_amines_on_acceptor=n_primary,
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
    return min(events, key=lambda e: (not e['shortlist_eligible'], tier_order.get(e['chemical_tier'], 1), not e['reacts_at_primary_amine'],
                                     not e['free_acid_donor'], e['acceptor_heavy_atoms'] + e['donor_heavy_atoms']))


def review(top100, count=10, policy='chemical_resemblance'):
    features = pd.DataFrame([dict(rxn_id=r.rxn_id, **assess(r.rxn_smiles, strict=policy == 'legacy_monomer')) for r in top100.itertuples()])
    reviewed = top100.merge(features, on='rxn_id', validate='one_to_one')
    pool = reviewed[reviewed.shortlist_eligible].copy()
    if policy == 'legacy_monomer':
        pool['tier_order'] = pool.chemical_tier.map({'diacid–primary-diamine analogue': 0,
                                                   'polyfunctional analogue': 1, 'pre-amidated intermediate': 2})
        pool = pool.sort_values(['tier_order', 'rxn_id']).drop(columns='tier_order')
    elif policy == 'chemical_resemblance':
        from amide_synthesis.reaction_search.chemical_selection import score_features
        scored = pd.DataFrame([dict(rxn_id=row.rxn_id, **score_features(row)) for _,row in pool.iterrows()])
        reviewed = reviewed.merge(scored, on='rxn_id', how='left', validate='one_to_one')
        pool = reviewed[reviewed.shortlist_eligible].copy()
        keys = ['chemical_resemblance'] + [key for key in ['drfp_tanimoto','hybrid_cosine'] if key in pool] + ['rxn_id']
        pool = pool.sort_values(keys, ascending=[False]*(len(keys)-1)+[True])
    else:
        from amide_synthesis.reaction_search.retrieval_policy import rank_reactions
        pool = rank_reactions(pool) if 'drfp_tanimoto' in pool else pool.sort_values('rxn_id')
    pool = pool.drop_duplicates('organic_transformation_group')
    if len(pool) < count:
        raise RuntimeError(f'Chemical eligibility produced only {len(pool)} distinct candidates; requested {count}')
    shortlist = pool.head(count).copy()
    reviewed.attrs['selection_policy'] = policy
    for name in ['display_title', 'why_similar', 'key_difference', 'explanation_source']: shortlist[name] = ''
    for i, row in shortlist.iterrows():
        annotation = EXPLANATIONS.get(row.rxn_id)
        for field, value in zip(['display_title', 'why_similar', 'key_difference'],
                                annotation or explain(row)):
            shortlist.loc[i, field] = value
        shortlist.loc[i, 'explanation_source'] = 'curated_display_only' if annotation else 'computed_from_structure'
    # This is a priority group selection, not a new numerical ranking.
    return reviewed, shortlist.reset_index(drop=True)


def explain(row):
    """Generate an explanation for any qualifying reaction, without an ID allowlist.

    Curated text above only improves display names. It cannot admit, exclude or
    reorder a reaction. Spacer lengths are descriptors, not selection thresholds.
    """
    return (
        row.get('name') if isinstance(row.get('name'),str) and row.get('name') else row.rxn_id,
        f"A free donor with {int(row.carboxyl_groups_on_donor)} carboxyl groups couples "
        f"to an acceptor with {int(row.primary_amines_on_acceptor)} primary amines; "
        f"{int(row.new_amide_bonds)} net new amide(s).",
        f"Acid/amine carbon spacers: {row.acid_spacer_carbons}/{row.amine_spacer_carbons} "
        f"versus target 4/6; pre-existing amides: {int(row.preexisting_amide_bonds)}; "
        f"retained acid/primary amine: {bool(row.product_retains_carboxyl)}/{bool(row.product_retains_primary_amine)}. "
        'Branching, unsaturation and stereochemistry require inspection of the saved SMILES.',
    )


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
    count = len(shortlist)
    rubric = {
        'version': '20261020-user-corrected',
        'retrieval': 'DRFP-ranked top 100 from 480 balanced amide-forming reactions; cofactors retained.',
        'selection': ['A carbon-bound primary amine forms an amide with a free carboxylic-acid donor.',
                      'At least one acid and one reacting primary amine suffice; AB self-condensation is allowed with stoichiometric copies.',
                      'At least one net new amide; multiple amidations are allowed.',
                      'Aromaticity, rings, functional-group counts, existing amides, spacers and retained ends are descriptors, not exclusion rules.',
                      'Chemically rank eligible reactions by the explicit soft resemblance policy, use DRFP/hybrid scores only for ties, group duplicate transformations, then select the requested count.'],
        'interpretation': 'Native reaction similarity does not establish target-substrate specificity or activity. Curated display text never controls selection.',
        'scope_limit': 'The top 100 retrieval cut can still miss useful reactions. Secondary-amine acylation and activated-donor-only reactions are outside this free-acid/primary-amine branch.',
        'determinism': 'Selection uses code and versioned chemical preferences; original reaction IDs never affect selection.',
        'cofactors': 'Retained in every full equation and score.',
    }
    if reviewed.attrs.get('selection_policy') == 'chemical_resemblance':
        from amide_synthesis.reaction_search.chemical_selection import load_policy
        rubric['soft_chemical_policy'] = load_policy()
    if reviewed.attrs.get('selection_policy') == 'legacy_monomer':
        rubric['version'] = 'legacy_monomer_replay'
        rubric['selection'] = ['Historical separate diacid/diamine, one-amide, retained-end and acyclic/nonaromatic filters; category then ID order. Superseded for current production.']
    (out / 'chemical_review_policy.json').write_text(json.dumps(rubric, indent=2) + '\n')
    policy = ''.join('<li>' + esc(x) + '</li>' for x in rubric['selection'])
    page = ('<!doctype html><meta charset="utf-8"><title>Nylon-6,6 chemical shortlist</title>'
        '<style>body{font:16px/1.5 system-ui;max-width:1300px;margin:25px auto;padding:0 15px}img{width:100%;max-width:850px}'
        'table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #ddd;vertical-align:top;text-align:left}th{background:#eee}</style>'
        f'<h1>{count} candidates for nylon-6,6 amide formation</h1>'
        '<p>DRFP retrieves the top 100. Explicit soft chemical criteria select the final panel; aromatic, cyclic, monoacid, monoamine and AB substrates remain eligible.</p>'
        f'<img src="top{count}_native_reactions.svg">'
        f'<details><summary>Embedding similarity scores</summary><img src="top{count}_similarity_heatmap.svg"></details><table>' + rows + '</table>'
        f'<p><a href="top{count}_reactions.tsv">Chemical shortlist TSV</a> · <a href="top100_reactions.tsv">DRFP top 100 TSV</a> · '
        f'<a href="chemical_review.tsv">Chemical review of all 100</a> · <a href="top{count}_native_reactions.pdf">Native reaction diagrams PDF</a> · '
        '<a href="drfp_retrieval_report.html">Original retrieval report</a></p>'
        '<details><summary>Selection criteria and limits</summary><ul>' + policy + '</ul><p>' + esc(rubric['interpretation']) +
        '</p><p>' + esc(rubric['scope_limit']) + '</p><p>Embedding scores are supporting evidence; compare rankings within a method, not numerical scales across methods.</p></details>'
        '<details><summary>Review all 100 retrieved reactions</summary><table>' + table + '</table></details>')
    (out / 'report.html').write_text(page)
