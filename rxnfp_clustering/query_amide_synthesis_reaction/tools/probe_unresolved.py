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

Measured 2026-10-13, after the gate removal (473 admitted, 305 resolved):

      0  identifier present, UniProt returns hits   <- nothing is being lost
     21  identifier present, UniProt returns zero
    117  no identifier, a name was tried, nothing found
     30  no identifier and no usable name

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


def unresolved(hits=None):
    hits = (pd.read_csv(RES / 'amide_forming_neighbours.tsv', sep='\t')
            if hits is None else hits).drop_duplicates('rxn_id')
    resolved = set()
    for name, column in (('enzymes_uniprot.tsv', 'rxn_id'),
                         ('precomputed_link_candidates.tsv', 'rxn_id')):
        path = RES / name
        if path.exists():
            resolved |= set(pd.read_csv(path, sep='\t',
                                        low_memory=False)[column])
    return hits[~hits.rxn_id.isin(resolved)], len(hits), len(resolved)


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
    pending, screened, resolved = unresolved()
    print(f'screened {screened}  resolved {resolved}  '
          f'unresolved {len(pending)}\n')
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
