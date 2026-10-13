#!/usr/bin/env python
"""Find the ModelSEED reactions that actually make the nylon-6,6 bond.

A whole-reaction difference fingerprint is dominated by the largest compositional
change, which for the activated framings is the cofactor (ATP -> AMP + PPi, or
acyl-CoA -> CoA), not the amide bond. Cosine alone therefore ranks cofactor
sharing as highly as chemistry: the CoA query's nearest EC-labelled neighbours
are thioester HYDROLASES (EC 3.1.2), which make no amide at all.

So candidates are screened by mechanism, independently of the embedding, and the
screens are applied BEFORE any top-N cut so ineligible reactions cannot occupy
ranked slots.

Screen 1 - a reactant amine is reconstructed from a product amide
    Cut product carboxamide C-N bonds and compare the reconstructed parent amine
    with the reactants. Validate the nitrogen that actually reacted and try cut
    sets of up to two bonds, including simultaneous acylation of a diamine.
    Net amide counts and local environments alone cannot establish this linkage.

Screen 2 - the acylated nitrogen is a free aliphatic amine
    "Two nitrogens" is far too loose: it accepts adenine, imidazole, nitriles and
    guanidino groups, and would call serotonin a diamine. Eligible nitrogens are
    explicitly sp3, hydrogen-bearing, non-aromatic, and not part of an amide,
    nitrile, nitro or guanidino group. A reaction counts as acylating a free
    diamine only when such a donor carries at least two of them.

Screen 3 - the substrate is a small molecule, not a protein
    ModelSEED represents protein substrates (biotin/lipoyl carriers, protein
    lysines) with a small surrogate structure, so structure alone cannot tell them
    apart. Evidence is taken from the compound table: R-group formulas and name
    stems. Stems are matched truncation-tolerantly because ModelSEED names are cut
    mid-word ("...-E2-protei"), which a naive "protein" test misses - that is how
    rxn40983 previously reached rank 2 of the CoA query.

Outputs (results/):
  amide_forming_neighbours.tsv   eligible reactions, ranked per query
  enzyme_candidates.tsv          per-EC shortlist
  screen_summary.json            how many reactions each screen admits
"""
from __future__ import annotations

import csv
import glob
from concurrent.futures import ProcessPoolExecutor
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')
from .paths import DB, RES


from . import encode_and_search as search
from . import corpus as _corpus
from .acylation import acylated_acceptors
from .substrate_distance import score_reaction

TOPN = 30
# Cosine is a RECALL ROUTE, not a gate. A whole-reaction difference fingerprint
# is dominated by the cofactor, so the per-query top-N silently discarded most
# of the chemistry the screen had already proven: of the 62 eligible reactions
# scoring >= 0.7 on substrate distance, 41 never reached enzyme lookup -
# including every folylpolyglutamate synthetase (a processive amide polymerase)
# and the meso-diaminopimelate Mur ligases. Reactions are now admitted if they
# are near in embedding space OR nylon-like in substrate chemistry; the two
# routes are unioned and labelled, so this can only add candidates.
# DEFAULT 0.0: admit every reaction the screen has PROVEN forms an amide on a
# free small-molecule amine, and let substrate distance RANK them rather than
# gate them.
#
# This project removed four separate places where the embedding silently gated
# retrieval instead of ranking it (the per-query top-N, the candidate-table
# priority, the OOD floor, and stage 04 re-litigating stage 03's admission).
# Then it left a gate of its own: a 0.40 substrate-distance floor, which
# excluded 214 of the 473 screen-eligible reactions - every one of them a
# reaction whose amide formation on a free amine had been PROVEN by
# reconstruction, merely not nylon-like enough for the floor.
#
# Gating on a score discards evidence; ranking on it does not. The score still
# does all the prioritisation work downstream - `master_priority` weights it
# 3x, and the shortlist allocates slots by evidence tier - so low-scoring
# chemistry cannot crowd out the nylon-like candidates. Set the environment
# variable to restore a floor if a run needs one.
SD_FLOOR = float(os.environ.get('SUBSTRATE_DISTANCE_FLOOR', 0.0))
from .workers import worker_count, describe
WORKERS = worker_count()

MACROMOLECULE_STEMS = (
    'protei', 'carrier', 'bccp', 'monomer', 'apo-', 'holo-', 'lysw', 'pup',
    'acyl-carrier', 'trna', 'ribosom', 'peptidoglyc', 'polymer', 'enzyme',
    'subunit', 'domain', 'oligo', 'glycogen', 'starch', 'cellulos')

# A POLYMER acceptor is not the same thing as a PROTEIN acceptor, and
# collapsing them cost the most relevant enzyme class in the database.
#
# The macromolecular screen exists to refuse reactions that acylate a folded
# protein or a carrier: you cannot offer HMDA to an enzyme whose acceptor is a
# protein lysine. That is right for LysW carrier protein and protein-L-lysine.
# But it also refused every enzyme whose acceptor is its own GROWING POLYAMIDE,
# which is precisely what a nylon polymerase is:
#
#   cyanophycin synthetase (rxn19007, rxn46344)  poly(Asp-Arg) biopolymer,
#       iteratively extended by ATP-dependent amide bonds - the closest
#       biological analogue to nylon polycondensation in the database
#   poly-gamma-glutamyl elongation (rxn58922, rxn58972)
#   coenzyme F420 polyglutamate ligases CofE/CofF (rxn16237, rxn24598,
#       rxn24604 and the F420-4..7 series) - iterative glutamylation
#
# 36 of the 231 macromolecular-flagged reactions score >= 0.6 on substrate
# distance, and the ones above are the top of that list. They are now admitted
# as a distinct, labelled class rather than discarded: the acceptor is a
# polyamide being extended, not a protein being decorated.
POLYMER_ACCEPTOR_TERMS = (
    'cyanophycin', 'polyglutam', 'poly-gamma-glutam', 'polygamma-glutam',
    'poly(gamma-glutam', 'polyamide', 'poly-l-glutam', 'glutamyl-l-glutamate',
    'f420-', 'polyaspart', 'poly-beta-', 'multi-l-glutam')
# If any of these appear, the acceptor really is a protein or an assembly and
# the reaction stays excluded whatever else it matches.
TRUE_MACROMOLECULE_TERMS = (
    'protei', 'carrier', 'bccp', 'lysw', 'pup', 'trna', 'ribosom',
    'peptidoglyc', 'enzyme', 'subunit', 'domain', 'apo-', 'holo-')


def _mask_cofactors(text):
    """Stop "coenzyme" matching the "enzyme" stem.

    A pre-existing false positive in the original screen: the stem `enzyme`
    is a substring of `coenzyme`, so every coenzyme-bearing reaction looked
    macromolecular - which silently excluded the coenzyme F420 polyglutamate
    ligases, among the most nylon-relevant chemistry in the database.
    """
    return (text.replace('coenzyme', 'cofactorterm')
                .replace('co-enzyme', 'cofactorterm'))


def _screen_block(reactions):
    """Worker: (free amines on the ACTUAL acceptor, substrate distance).

    Both come from the same reconstruction, so scoring costs no extra passes.
    """
    out = []
    for rxn in reactions:
        try:
            acceptors = acylated_acceptors(rxn)
        except Exception:
            acceptors = []
        amines = max((n for _, n in acceptors), default=0)
        distance, features = score_reaction(rxn) if amines else (0.0, None)
        iterative = bool(features and features['self_condensation'])
        out.append((amines, distance, iterative))
    return out


def load_compound_metadata():
    names, formulas = {}, {}
    for path in sorted(glob.glob(str(DB / 'compound_*.tsv'))):
        if 'provenance' in Path(path).name:
            continue
        for row in csv.DictReader(open(path), delimiter='\t'):
            names[row['id']] = row.get('name') or ''
            formulas[row['id']] = row.get('formula') or ''
    return names, formulas


def load_reaction_metadata():
    names, ecs, definitions, compounds = {}, defaultdict(set), {}, {}
    for path in sorted(glob.glob(str(DB / 'reaction_*.tsv'))):
        for row in csv.DictReader(open(path), delimiter='\t'):
            names[row['id']] = row.get('name', '') or ''
            definitions[row['id']] = row.get('definition', '') or ''
            compounds[row['id']] = [c for c in (row.get('compound_ids') or '').split(';') if c]
            if row.get('ec_numbers') and row['ec_numbers'] != 'null':
                for e in row['ec_numbers'].replace('|', ';').split(';'):
                    if e.strip():
                        ecs[row['id']].add(e.strip())
    aliases = defaultdict(set)
    alias_file = DB / 'Aliases/Unique_ModelSEED_Reaction_Aliases.txt'
    wanted = {'KEGG', 'MetaCyc', 'BiGG', 'Rhea'}
    if alias_file.exists():
        for row in csv.DictReader(open(alias_file), delimiter='\t'):
            if row.get('Source', '') in wanted:
                aliases[row['ModelSEED ID']].add(f"{row['Source']}:{row['External ID']}")
    return names, ecs, definitions, compounds, aliases


def is_polymer_elongation(definition, compound_ids, cpd_names):
    """True when the macromolecular partner is a POLYAMIDE being extended.

    Distinguishes a nylon-relevant polymerase (cyanophycin synthetase, a
    polyglutamate ligase) from protein acylation (a carrier protein, a
    protein lysine). Protein evidence always wins: a reaction naming both a
    polymer and a carrier protein stays excluded.
    """
    text = (definition or '').lower()
    for cpd in compound_ids:
        text += ' ' + cpd_names.get(cpd, '').lower()
    text = _mask_cofactors(text)
    if any(term in text for term in TRUE_MACROMOLECULE_TERMS):
        return False
    return any(term in text for term in POLYMER_ACCEPTOR_TERMS)


def is_macromolecular(definition, compound_ids, cpd_names, cpd_formulas):
    """Compound-level evidence first, then truncation-tolerant name stems."""
    for cpd in compound_ids:
        formula = cpd_formulas.get(cpd, '')
        if 'R' in formula and not formula.startswith(('Ru', 'Rb', 'Rh', 'Re')):
            return True
        name = _mask_cofactors(cpd_names.get(cpd, '').lower())
        if any(stem in name for stem in MACROMOLECULE_STEMS):
            return True
    return any(stem in _mask_cofactors((definition or '').lower())
               for stem in MACROMOLECULE_STEMS)


def main():
    from .paths import ensure_directories
    ensure_directories()
    queries = search.build_queries()
    xq = search.unit(search.featurise(queries.rxn_smiles.to_numpy()))
    full, x_full = _corpus.load()

    cpd_names, cpd_formulas = load_compound_metadata()
    names, ecs, definitions, compounds, aliases = load_reaction_metadata()

    print(f'Screening {len(full):,} reactions on {describe(WORKERS)}', flush=True)
    reactions = list(full.rxn_smiles)
    block = 500
    chunks = [reactions[i:i + block] for i in range(0, len(reactions), block)]
    screened = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for result in pool.map(_screen_block, chunks):
            screened.extend(result)
    assert len(screened) == len(full)

    acylates = np.array([n > 0 for n, _, _ in screened], bool)
    amines = np.array([n for n, _, _ in screened], int)   # on the ACTUAL acceptor
    distance = np.array([d for _, d, _ in screened], float)
    iterative = np.array([i for _, _, i in screened], bool)
    macro = np.zeros(len(full), bool)
    polymer = np.zeros(len(full), bool)
    for i, rxn_id in enumerate(full.rxn_id):
        if acylates[i]:
            macro[i] = is_macromolecular(definitions.get(rxn_id, ''),
                                         compounds.get(rxn_id, []),
                                         cpd_names, cpd_formulas)
            if macro[i]:
                # A polyamide being extended is not a protein being decorated.
                polymer[i] = is_polymer_elongation(definitions.get(rxn_id, ''),
                                                   compounds.get(rxn_id, []),
                                                   cpd_names)

    # Documented balance policy: a reaction whose heavy atoms do not balance is
    # not a trustworthy basis for an experimental candidate, so it is excluded
    # from ranking rather than merely flagged.
    balanced = full.heavy_balanced.to_numpy(dtype=bool)
    if 'proton_fixable' in full.columns:
        balanced &= full.proton_fixable.to_numpy(dtype=bool)
    # Polymer-elongating reactions are macromolecular AND admitted: their
    # acceptor is a growing polyamide, which is the target chemistry.
    eligible = acylates & (~macro | polymer) & (amines >= 1) & balanced
    summary = {
        'n_reactions': int(len(full)),
        'acylates_a_free_amine': int(acylates.sum()),
        'of_those_macromolecular_substrate': int((acylates & macro).sum()),
        'of_those_polymer_elongation_readmitted': int(
            (acylates & macro & polymer).sum()),
        'of_those_on_a_free_diamine_donor': int((acylates & ~macro & (amines >= 2)).sum()),
        'of_those_excluded_as_unbalanced': int((acylates & ~macro & (amines >= 1)
                                                & ~balanced).sum()),
        'eligible_for_ranking': int(eligible.sum()),
        'eligible_above_substrate_distance_floor': int((eligible
                                                        & (distance >= SD_FLOOR)).sum()),
        'substrate_distance_floor': SD_FLOOR,
        'self_condensing_iterative': int((eligible & iterative).sum()),
    }
    print(json.dumps(summary, indent=2), flush=True)
    (RES / 'screen_summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    rows = []
    candidates = np.flatnonzero(eligible)          # screens applied BEFORE top-N

    # Route A: nearest in embedding space, per query framing (the original gate).
    # Route B: nylon-like substrate chemistry, regardless of embedding distance.
    # Unioned, never intersected, so route B can only ADD candidates.
    all_sims = np.vstack([x_full @ xq[qi] for qi in range(len(queries))])
    best_query = all_sims[:, :].argmax(axis=0)
    by_cosine = set()
    for qi in range(len(queries)):
        by_cosine.update(candidates[np.argsort(-all_sims[qi][candidates])][:TOPN].tolist())
    by_distance = set(np.flatnonzero(eligible & (distance >= SD_FLOOR)).tolist())
    print(f'admitted: {len(by_cosine)} by cosine, {len(by_distance)} by substrate '
          f'distance >= {SD_FLOOR}, {len(by_cosine | by_distance)} unioned '
          f'(+{len(by_distance - by_cosine)} the cosine gate would have missed)',
          flush=True)

    for qi, name in enumerate(queries['query']):
        sims = all_sims[qi]
        # Each query reports its own top-N, plus every distance-admitted reaction
        # for which this framing is the closest one - so each row still states
        # which framing found it and nothing is attributed twice.
        selected = candidates[np.argsort(-sims[candidates])][:TOPN].tolist()
        order = list(dict.fromkeys(
            selected + sorted(r for r in by_distance - by_cosine
                              if best_query[r] == qi)))
        for rank, ref in enumerate(order, start=1):
            rec = full.iloc[ref]
            rows.append({
                'query': name, 'rank': rank,
                'found_via': ('cosine_topn' if ref in by_cosine
                              else 'substrate_distance'),
                'substrate_distance': round(float(distance[ref]), 4),
                'self_condensation': bool(iterative[ref]),
                'polymer_elongation': bool(polymer[ref]),
                'cosine_similarity': round(float(sims[ref]), 4),
                'rxn_id': rec.rxn_id,
                'reaction_name': names.get(rec.rxn_id, ''),
                'ec': ';'.join(sorted(ecs.get(rec.rxn_id, []))) or (rec.ec or ''),
                'n_free_amines_on_donor': int(amines[ref]),
                'acylates_free_diamine': bool(amines[ref] >= 2),
                'heavy_balanced': bool(rec.heavy_balanced),
                'external_ids': ';'.join(sorted(aliases.get(rec.rxn_id, []))[:6]),
                'definition': definitions.get(rec.rxn_id, ''),
                'rxn_smiles': rec.rxn_smiles})
    hits = pd.DataFrame(rows)
    hits.to_csv(RES / 'amide_forming_neighbours.tsv', sep='\t', index=False)

    # ---- per-EC shortlist
    tally = defaultdict(lambda: {'best': -1.0, 'reactions': set(), 'diamine': False,
                                 'queries': set(), 'name': ''})
    for row in hits.itertuples():
        for ec in filter(None, str(row.ec).split(';')):
            entry = tally[ec]
            if row.cosine_similarity > entry['best']:
                entry['best'] = row.cosine_similarity
                entry['name'] = row.reaction_name
            entry['reactions'].add(row.rxn_id)
            entry['diamine'] |= bool(row.acylates_free_diamine)
            entry['queries'].add(row.query)
    shortlist = pd.DataFrame([
        {'ec': ec, 'best_similarity': v['best'], 'acylates_free_diamine': v['diamine'],
         'n_reactions': len(v['reactions']), 'example_reaction_name': v['name'],
         'reactions': ';'.join(sorted(v['reactions'])[:8]),
         'matched_queries': ';'.join(sorted(v['queries']))}
        for ec, v in tally.items()])
    if not shortlist.empty:
        shortlist = shortlist.sort_values(['acylates_free_diamine', 'best_similarity'],
                                          ascending=[False, False]).reset_index(drop=True)
    shortlist.to_csv(RES / 'enzyme_candidates.tsv', sep='\t', index=False)

    pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 52)
    print('\nTop candidates (free-diamine acylations first):')
    print(shortlist.head(18).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
