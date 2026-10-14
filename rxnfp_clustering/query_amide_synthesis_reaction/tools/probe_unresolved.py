#!/usr/bin/env python
"""Is an unresolved reaction OUR failure or an empty database? Re-query and see.

The unresolved count is not a quality measure on its own, and it moves every
time a route changes — it rose 91 -> 168 when the substrate-distance gate was
removed, purely because 214 previously EXCLUDED reactions became admitted
ones. Reading that as a regression would be wrong, and reading a stale
breakdown would be worse.

So this re-queries UniProt for every unresolved reaction, through every pivot
the pipeline knows (ModelSEED Rhea aliases, the MetaNetX seed pivot, the
MetaNetX MetaCyc/KEGG/BiGG sibling pivots, EC), and sorts the failures:

    identifier present, UniProt returns hits  -> OUR BUG. Fix it.
    identifier present, UniProt returns zero  -> undeposited enzymology
    no identifier, a name was tried           -> nothing named in UniProt
    no identifier and no usable name          -> nothing left to query

The first line is the only one that is actionable, and it is the number to
watch. It has been nonzero once: rxn14123 and rxn19001 sat at cosine 0.499
against an OOD floor of 0.5041 and were refused by a stage that re-litigated
an admission already decided upstream.

Measured 2026-10-13, current pipeline (480 screened, 308 directly resolved,
19 holding a borrowed association, 7 resolved through KEGG orthology, 146
with no enzyme at all):

      0  identifier present, UniProt returns hits   <- nothing is being lost
     15  identifier present, UniProt returns zero
    109  no identifier, a name was tried, nothing found
     29  no identifier and no usable name

Usage:
    python tools/probe_unresolved.py
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from amide_synthesis.enzyme_resolution import (enzyme_like_name,   # noqa: E402
                                               product_names)
from amide_synthesis.fetch_enzymes import load_aliases              # noqa: E402
from amide_synthesis.metanetx import (foreign_to_rhea,              # noqa: E402
                                      rhea_for_reaction, seed_to_rhea)
from amide_synthesis.paths import RES                               # noqa: E402
from amide_synthesis.uniprot import search                           # noqa: E402

THREADS = 6


def clean_ec(value):
    return [e for e in str(value).split(';')
            if e and e != 'nan' and '-' not in e
            and not e.split('.')[-1].isalpha()]


# Direct annotation: UniProt says THIS protein catalyses THIS reaction.
DIRECT = ('enzymes_uniprot.tsv', 'precomputed_link_candidates.tsv')
# Borrowed: the chemistry-neighbour route attached an enzyme from a reaction
# proven to do the same acylation. A real association, but a weaker claim, so
# it is counted separately rather than folded into either total.
BORROWED = ('chemistry_neighbour_candidates.tsv',)
# Resolved through KEGG orthology rather than any UniProt pivot. KEGG assigns
# the ortholog group to the REACTION, so this is a genuine reaction->enzyme
# link and a stronger claim than a borrowed one - but it reaches the enzymes
# through an EC the ortholog declares rather than through an annotation on
# the protein, so it is counted separately rather than folded into DIRECT.
ORTHOLOG = ('kegg_ortholog_links.tsv',)


def unresolved(hits=None):
    """Reactions with no enzyme at all, plus the two weaker categories.

    Returns (frame, screened, directly_resolved, borrowed_only,
    ortholog_only).

    The borrowed category was previously ignored, so 19 reactions that DO
    carry an enzyme - attached by the neighbour route from a chemically
    equivalent reaction - were reported as unresolved. "172 unresolved" was
    therefore an overstatement of "no enzyme anywhere": the honest figure is
    153, with 19 more holding a borrowed association. The KEGG orthology
    route later resolved 7 of those 153, giving the current 146.
    """
    hits = (pd.read_csv(RES / 'amide_forming_neighbours.tsv', sep='\t')
            if hits is None else hits).drop_duplicates('rxn_id')

    def gather(names):
        found = set()
        for name in names:
            path = RES / name
            if path.exists():
                frame = pd.read_csv(path, sep='\t', low_memory=False)
                if 'rxn_id' in frame.columns:
                    found |= set(frame.rxn_id.dropna())
        return found

    direct = gather(DIRECT)
    borrowed = gather(BORROWED) - direct
    ortholog = gather(ORTHOLOG) - direct - borrowed
    pending = hits[~hits.rxn_id.isin(direct | borrowed | ortholog)]
    return (pending, len(hits), len(direct),
            len(hits[hits.rxn_id.isin(borrowed)]),
            len(hits[hits.rxn_id.isin(ortholog)]))


def classify(row, aliases, seed_map, foreign_map, client=search):
    rhea = rhea_for_reaction(row.rxn_id, aliases, seed_map, foreign_map)[:6]
    ec = clean_ec(getattr(row, 'ec', ''))
    queries = []
    if rhea:
        queries.append(' OR '.join(f'rhea:{x}' for x in rhea))
    if ec:
        queries.append(' OR '.join(f'ec:{x}' for x in ec))
    hits = 0
    for query in queries:
        rows, _, _ = client(query, max_pages=1, fields='accession')
        hits += len(rows)
    named = bool(enzyme_like_name(getattr(row, 'reaction_name', ''))
                 or product_names(getattr(row, 'definition', '') or ''))
    if queries:
        verdict = 'identifier_but_uniprot_has_hits' if hits else \
                  'identifier_but_uniprot_empty'
    else:
        verdict = 'no_identifier_name_tried' if named else \
                  'no_identifier_no_name'
    return {'rxn_id': row.rxn_id,
            'substrate_distance': getattr(row, 'substrate_distance', None),
            'verdict': verdict, 'uniprot_hits': hits}


def main():
    pending, screened, resolved, borrowed, ortholog = unresolved()
    print(f'screened {screened}  directly resolved {resolved}  '
          f'borrowed association {borrowed}  '
          f'KEGG ortholog {ortholog}  '
          f'no enzyme at all {len(pending)}\n')
    if pending.empty:
        print('nothing unresolved')
        return

    aliases, seed_map, foreign_map = load_aliases(), seed_to_rhea(), foreign_to_rhea()
    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        frame = pd.DataFrame(list(pool.map(
            lambda r: classify(r, aliases, seed_map, foreign_map),
            [r for r in pending.itertuples()])))

    order = ['identifier_but_uniprot_has_hits', 'identifier_but_uniprot_empty',
             'no_identifier_name_tried', 'no_identifier_no_name']
    labels = {
        'identifier_but_uniprot_has_hits': 'OUR BUG - identifier works, we missed it',
        'identifier_but_uniprot_empty': 'undeposited - identifier returns zero',
        'no_identifier_name_tried': 'no identifier; a name was tried, nothing found',
        'no_identifier_no_name': 'no identifier and no usable name',
    }
    counts = frame.verdict.value_counts()
    for verdict in order:
        print(f'  {counts.get(verdict, 0):>5}  {labels[verdict]}')

    bug = frame[frame.verdict == 'identifier_but_uniprot_has_hits']
    if len(bug):
        print('\nACTIONABLE - these should have resolved:')
        print(bug.sort_values('uniprot_hits', ascending=False)
              .head(20).to_string(index=False))
    else:
        print('\nno reaction remains where an identifier would have worked.')

    high = frame[pd.to_numeric(frame.substrate_distance,
                               errors='coerce').fillna(0) >= 0.6]
    print(f'\nunresolved scoring >= 0.6 on substrate distance: {len(high)}')
    frame.to_csv(RES / 'unresolved_probe.tsv', sep='\t', index=False)
    print(f'written: {RES / "unresolved_probe.tsv"}')


if __name__ == '__main__':
    main()
