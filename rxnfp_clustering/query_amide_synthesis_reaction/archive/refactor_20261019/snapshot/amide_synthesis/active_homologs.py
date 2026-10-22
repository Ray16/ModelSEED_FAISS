#!/usr/bin/env python
"""Candidates reached from the enzymes that MEASURABLY work, not from a reaction.

WHY THIS ROUTE EXISTS, and why it is the uncomfortable one. Every other route
here starts from chemistry: a ModelSEED reaction the screen proved forms an
amide on a free amine, or a family enriched in such proteins. That is a
principled search and it produced 28 converged candidates.

It also cannot reach the only proteins known to do this reaction. The assayed
panel's three producers - AS9 at 9.28 ug/ml, AS17 at 1.91, AS23 at 1.45 - are
annotated as long-chain-fatty-acid--CoA ligases and acyl-CoA synthetases,
EC 6.2, which is THIOESTER chemistry. Annotation-based retrieval finds an
enzyme through the reaction it is annotated to catalyse, so it structurally
cannot find one whose annotation names a different reaction. Measured: of
2,952 proteins BLAST finds related to those three, **zero** are among this
project's 54,199 candidates, and zero of the closest 259 carry any
amide-forming evidence.

So this route asks the other question: not "what chemistry resembles the
target" but "what resembles the proteins that already work". It is the only
route anchored in measured activity rather than annotation.

WHAT A HIT IS AND IS NOT. Sequence identity to a measured producer is
evidence about FUNCTION by descent, and at 50-65% it is strong evidence of
shared function. It is NOT evidence that the shared function is nylon amide
synthesis - the producers' own annotations say otherwise, and three of 49
panel members produce at all. So the weight sits at 2.0, level with
`activity_ec`: better than a domain match, well below a protein measured on
the diad itself, and far below a protein whose own reaction was screened.

The identity is carried per row so a reader can apply their own threshold
rather than inheriting mine.
"""
from __future__ import annotations

import json
import os

import pandas as pd

from .paths import RES

SOURCE = 'blast_actives_deep.tsv'
# Below this, "related" stops meaning much for function.
MIN_IDENTITY = float(os.environ.get('ACTIVE_HOMOLOG_MIN_IDENTITY', 40.0))
# A fragment matching one domain is not a comparable enzyme; identity is
# reported against the shorter sequence, so length has to be checked too.
LENGTH_TOLERANCE = float(os.environ.get('ACTIVE_HOMOLOG_LENGTH_TOLERANCE', 0.25))


def panel_median_length(table):
    panel = table[table.evidence_strength.eq('measured_activity')]
    lengths = pd.to_numeric(panel.get('length'), errors='coerce')
    return float(lengths.median()) if lengths.notna().any() else 0.0


def select(hits, median_length, min_identity=MIN_IDENTITY,
           tolerance=LENGTH_TOLERANCE):
    """Comparable-length homologs above the identity floor."""
    column = next((c for c in hits.columns if 'Identities' in c), None)
    if column is None:
        return hits.head(0)
    out = hits[pd.to_numeric(hits[column], errors='coerce') >= min_identity]
    if median_length and 'Length' in out.columns:
        length = pd.to_numeric(out.Length, errors='coerce')
        out = out[(length >= median_length * (1 - tolerance))
                  & (length <= median_length * (1 + tolerance))]
    return out.rename(columns={column: 'pct_identity_to_active'})


def main():
    from .paths import ensure_directories
    from .enrich import fetch
    from .master_table import RENAME
    ensure_directories()

    path = RES / SOURCE
    if not path.exists():
        raise SystemExit(f'{SOURCE} not found; run the BLAST search first')
    hits = pd.read_csv(path, sep='\t', low_memory=False)

    master = RES / 'master_candidates_enriched.tsv'
    if not master.exists():
        master = RES / 'master_candidates.tsv'
    table = pd.read_csv(master, sep='\t', low_memory=False,
                        usecols=lambda c: c in ('uniprot', 'evidence_strength',
                                                'length'))
    median = panel_median_length(table)
    chosen = select(hits, median)
    print(f'{len(hits):,} BLAST hits -> {len(chosen):,} at >= {MIN_IDENTITY}% '
          f'identity and within {int(LENGTH_TOLERANCE * 100)}% of the panel '
          f'median ({median:.0f} aa)', flush=True)

    accessions = sorted(chosen.Accession.dropna().astype(str).unique())
    rows, failed = fetch(accessions)
    frame = pd.DataFrame(rows).rename(columns=RENAME)
    identity = dict(zip(chosen.Accession, chosen.pct_identity_to_active))
    closest = dict(zip(chosen.Accession, chosen.get('query', '')))
    frame['pct_identity_to_active'] = frame.uniprot.map(identity)
    frame['closest_measured_active'] = frame.uniprot.map(closest)
    frame['discovery_route'] = 'active_homolog'
    frame['evidence_strength'] = 'homolog_of_measured_active'
    frame['discovery_tier'] = 'blast_vs_measured_active'
    frame.to_csv(RES / 'active_homolog_candidates.tsv', sep='\t', index=False)

    summary = {
        'blast_hits': int(len(hits)),
        'passing_identity_and_length': int(len(chosen)),
        'records_retrieved': int(len(frame)),
        'failed_batches': failed,
        'min_identity': MIN_IDENTITY,
        'panel_median_length': round(median, 1),
        'note': 'Anchored in MEASURED activity rather than annotation - the '
                'only route here that is. Identity to a producer is evidence '
                'of shared function, not evidence that the shared function '
                'is nylon amide synthesis.',
    }
    print(json.dumps(summary, indent=2), flush=True)
    (RES / 'active_homolog_summary.json').write_text(
        json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
