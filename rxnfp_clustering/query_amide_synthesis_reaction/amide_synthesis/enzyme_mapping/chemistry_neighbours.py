#!/usr/bin/env python
"""Candidates for reactions no database can resolve, by chemical neighbour.

After Rhea, the MetaNetX sibling-identifier pivots, EC, product name and
reaction name, 91 of 259 screened reactions still resolve to no protein, and a
direct probe shows why: **zero** of them would be resolved by an identifier we
failed to try. 18 carry an identifier UniProt returns nothing for, and 73 carry
no identifier at all. They are bounded by what has been deposited and named,
not by retrieval. 27 of them still score >= 0.6 on substrate distance.

So the last route does not look anything up. It reasons:

    If an unresolved reaction performs the SAME acylation chemistry as a
    reaction that DID resolve, the enzymes of that resolved reaction are
    candidates for it.

Similarity is measured on the acylation features the screen already proved -
how many free amines the acceptor carries, how many acids the donor carries,
whether it self-condenses, whether it is an AA/BB pair, the two spacer lengths,
and the two molecule sizes. Not on the embedding: cosine is cofactor-dominated,
which is the failure this whole pipeline has been unwinding.

WHAT IT ACTUALLY BUYS - measured, and less than it looks. The route bridges 19
previously unreachable reactions with 121 associations over 58 proteins, and
adds **zero new candidates**. That is not a bug, it is arithmetic: the enzymes
are borrowed FROM resolved reactions, so every one of them is already in the
table under stronger evidence.

The useful reading is the converse, and it is the strongest completeness
statement available here: the 91 unresolved reactions cannot yield a new
protein. Not through UniProt - a direct probe shows zero of them would resolve
via an identifier we failed to try - and not through chemical inference either,
because any protein close enough in chemistry to borrow is already a candidate.
The candidate set is saturated with respect to the reaction routes.

What the route does provide is COVERAGE: which known enzymes to try for those
19 reactions. That is a different question from breadth, and the summary
reports both numbers rather than the flattering one.

THIS IS AN INFERENCE, NOT AN ANNOTATION, and the labelling has to keep that
obvious. The protein is NOT annotated to the unresolved reaction; it catalyses
a chemically near-identical one. Every row records the reaction it was borrowed
from and the feature distance, `evidence_strength` is `chemistry_neighbour`,
and it carries the lowest weight of any route. A reader who distrusts the
inference can drop one labelled value and lose nothing else.
"""
from __future__ import annotations

import json
import os

import pandas as pd

from amide_synthesis.reaction_search.acylation import acylation_events
from amide_synthesis.reaction_search.substrate_distance import describe_event
from amide_synthesis.infrastructure.freshness import tracked_stage
from amide_synthesis.infrastructure.paths import RES

# Only bridge reactions whose own chemistry is worth the effort.
MIN_SUBSTRATE_DISTANCE = float(os.environ.get('NEIGHBOUR_MIN_SD', 0.60))
# L1 distance over the feature vector below. 0 means an identical signature;
# beyond this the two reactions are not doing the same thing.
MAX_FEATURE_DISTANCE = float(os.environ.get('NEIGHBOUR_MAX_DISTANCE', 3.0))
# Borrowing from a reaction with hundreds of enzymes would swamp the table.
MAX_BORROWED = int(os.environ.get('NEIGHBOUR_MAX_BORROWED', 12))


def features(rxn_smiles):
    """Acylation signature of a reaction, or None if it forms no amide."""
    try:
        events = acylation_events(rxn_smiles)
    except Exception:
        return None
    if not events:
        return None
    f = describe_event(events[0])
    return (
        f['acceptor_free_amines'],
        f['acyl_n_acids'],
        int(f['self_condensation']),
        int(f['aa_bb_pair']),
        f['amine_spacer'] or 0,
        f['acid_spacer'] or 0,
        f['acceptor_heavy_atoms'] or 0,
        f['donor_parent_heavy_atoms'] or 0,
    )


def distance(a, b):
    return sum(abs(x - y) for x, y in zip(a, b))


def nearest_resolved(target, pool):
    """(reaction id, distance) of the closest resolved reaction, or (None, inf)."""
    best, best_distance = None, float('inf')
    for rxn_id, signature in pool.items():
        d = distance(target, signature)
        if d < best_distance:
            best, best_distance = rxn_id, d
    return best, best_distance


def build(hits, enzymes, min_sd=MIN_SUBSTRATE_DISTANCE,
          max_distance=MAX_FEATURE_DISTANCE, max_borrowed=MAX_BORROWED):
    """Borrow enzymes from chemically equivalent resolved reactions."""
    resolved_ids = set(enzymes.rxn_id)
    one_per_reaction = hits.drop_duplicates('rxn_id')

    pool = {}
    for row in one_per_reaction[one_per_reaction.rxn_id.isin(resolved_ids)].itertuples():
        signature = features(row.rxn_smiles)
        if signature:
            pool[row.rxn_id] = signature
    if not pool:
        return pd.DataFrame()

    unresolved = one_per_reaction[
        ~one_per_reaction.rxn_id.isin(resolved_ids)
        & (one_per_reaction.substrate_distance.fillna(0) >= min_sd)]

    rows = []
    for row in unresolved.itertuples():
        signature = features(row.rxn_smiles)
        if not signature:
            continue
        source, d = nearest_resolved(signature, pool)
        if source is None or d > max_distance:
            continue
        borrowed = enzymes[enzymes.rxn_id == source].head(max_borrowed)
        for protein in borrowed.itertuples():
            rows.append({
                'uniprot': protein.uniprot,
                'gene': getattr(protein, 'gene', ''),
                'protein_name': getattr(protein, 'protein_name', ''),
                'organism': getattr(protein, 'organism', ''),
                'sequence': getattr(protein, 'sequence', ''),
                'ec': getattr(protein, 'ec', ''),
                'pfam': getattr(protein, 'pfam', ''),
                'rxn_id': row.rxn_id,
                'borrowed_from_rxn': source,
                'chemistry_feature_distance': d,
                'substrate_distance_measured': float(row.substrate_distance),
                'discovery_route': 'chemistry_neighbour',
                'evidence_strength': 'chemistry_neighbour',
                'discovery_tier': 'inferred_from_equivalent_reaction',
            })
    return pd.DataFrame(rows)


@tracked_stage('neighbours')
def main():
    from amide_synthesis.infrastructure.paths import ensure_directories
    ensure_directories()
    hits = pd.read_csv(RES / 'amide_forming_neighbours.tsv', sep='\t')
    enzymes = pd.read_csv(RES / 'candidates_ranked.tsv', sep='\t', low_memory=False)
    linked=RES/'precomputed_link_candidates.tsv'
    if linked.exists():enzymes=pd.concat([enzymes,pd.read_csv(linked,sep='\t',low_memory=False)],ignore_index=True)
    enzymes=enzymes[enzymes.evidence_strength.ne('unverified_reaction_search')]
    frame = build(hits, enzymes)
    frame.to_csv(RES / 'chemistry_neighbour_candidates.tsv', sep='\t', index=False)

    summary = {
        'reactions_bridged': int(frame.rxn_id.nunique()) if len(frame) else 0,
        'associations': int(len(frame)),
        'unique_proteins': int(frame.uniprot.nunique()) if len(frame) else 0,
        'identical_chemistry': int((frame.chemistry_feature_distance == 0).sum())
                               if len(frame) else 0,
        'new_proteins_added': 0,   # see module docstring: necessarily zero
        'min_substrate_distance': MIN_SUBSTRATE_DISTANCE,
        'max_feature_distance': MAX_FEATURE_DISTANCE,
    }
    print(json.dumps(summary, indent=2), flush=True)
    (RES / 'chemistry_neighbour_summary.json').write_text(
        json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
