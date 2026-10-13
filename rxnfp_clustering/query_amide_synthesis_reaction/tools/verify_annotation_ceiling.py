#!/usr/bin/env python
"""Test whether a blank annotation field is OUR gap or UniProt's.

A coverage percentage is not self-explanatory. `catalytic_activity` sits at
76.5%, and that is either a retrieval failure we should fix or the limit of
what UniProt curates for unreviewed entries - opposite conclusions, and
asserting one without checking has already been wrong twice in this project.

So this re-queries UniProt for a random sample of accessions that are BLANK in
our table and asks whether UniProt has the field at all. If it does, we have a
bug. If it does not, the percentage is a property of the source.

Usage:
    python tools/verify_annotation_ceiling.py [field] [sample size]

Measured 2026-10-13, catalytic_activity, n=120, seed 0:
    UniProt HAS the field        0      <- would be our bug
    UniProt genuinely has none 120
    failed batches               0

8,008 of 34,134 rows lack the field; 5,756 are clade homologs and 2,076 family
homologs, i.e. unreviewed TrEMBL entries, which systematically carry no CC
lines. 47 are the assayed panel, which has no UniProtKB entry at all.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from amide_synthesis.enrich import FILLABLE, blank, fetch   # noqa: E402
from amide_synthesis.paths import RES                        # noqa: E402

UNIPROT_ACCESSION = r'^[A-NR-Z0-9][0-9][A-Z0-9]{3}[0-9]'


def verify(field='catalytic_activity', size=120, seed=0):
    source = RES / 'master_candidates_enriched.tsv'
    if not source.exists():
        source = RES / 'master_candidates.tsv'
    table = pd.read_csv(source, sep='\t', low_memory=False)
    if field not in table.columns:
        raise SystemExit(f'no column {field!r} in {source.name}')

    uniprot_name = next((k for k, v in FILLABLE.items() if v == field), None)
    if uniprot_name is None:
        raise SystemExit(f'{field!r} is not fetched from UniProt')

    missing = table[blank(table[field])]
    candidates = missing[missing.uniprot.astype(str).str.match(UNIPROT_ACCESSION)]
    pool = sorted(candidates.uniprot.dropna().unique())
    if not pool:
        print(f'no blank {field} rows carry a UniProt accession')
        return
    random.seed(seed)
    sample = random.sample(pool, min(size, len(pool)))

    rows, failures = fetch(sample)
    present = [r['Entry'] for r in rows if (r.get(uniprot_name) or '').strip()]

    print(f'source                     {source.name}')
    print(f'field                      {field}  ({uniprot_name})')
    print(f'blank rows                 {len(missing)} of {len(table)}')
    print(f'blank with a UniProt id    {len(pool)}')
    print(f'probed                     {len(sample)}  (seed {seed})')
    print(f'records returned           {len(rows)}  ({failures} failed batches)')
    print(f'UniProt HAS the field      {len(present)}   <- our bug if nonzero')
    print(f'UniProt genuinely has none {len(rows) - len(present)}')
    if present:
        print(f'examples we are missing:   {present[:8]}')
    print('\nby evidence route:')
    print(missing.evidence_strength.value_counts().to_string())


if __name__ == '__main__':
    name = sys.argv[1] if len(sys.argv) > 1 else 'catalytic_activity'
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    verify(name, count)
