#!/usr/bin/env python
"""Is any column computed upstream and then silently dropped at the merge?

THE DEFECT THIS CATCHES, which occurred twice. `master_table.build`
concatenates every route and then de-duplicates on accession, keeping the
strongest record per protein. When a weaker route is the only one carrying a
column, every row holding it loses the contest and the column survives in the
schema while holding nothing:

  acylates_free_diamine     the screen PROVES the acceptor is a free diamine -
                            exactly what HMDA is - for 89 of 480 reactions.
                            Dropped at the merge; it is now the most
                            discriminating axis in the convergence count.
  borrowed_from_rxn         the chemistry-neighbour route's entire answer.
                            0 of 54,199 until it was re-attached after the
                            de-duplication rather than before it.

Neither failed loudly. The column existed, every value was blank, and nothing
reported it - which is why this is a check rather than a lesson.

A column at zero is a bug. A column that is merely THIN is usually correct:
struct_S covers 36 because a separate project scored 36 proteins, and the
assay columns cover 49 because the panel has 49 members. The tool therefore
separates the two and only treats zero as a failure.

Usage:
    python tools/audit_columns.py          # exits non-zero if any column is empty
"""
from __future__ import annotations
from amide_synthesis.infrastructure.paths import PROJECT

import sys
from pathlib import Path

import pandas as pd


from amide_synthesis.candidate_selection.master_table import COLUMNS
from amide_synthesis.infrastructure.paths import RES

THIN_FRACTION = 0.01


def populated(frame, column):
    series = frame[column]
    if series.dtype == object:
        return int(series.fillna('').astype(str).str.strip().ne('').sum())
    return int(series.notna().sum())


def audit(frame, declared=COLUMNS):
    empty, thin = [], []
    for column in declared:
        if column not in frame.columns:
            empty.append((column, 'MISSING'))
            continue
        count = populated(frame, column)
        if count == 0:
            empty.append((column, 0))
        elif count < len(frame) * THIN_FRACTION:
            thin.append((column, count))
    return empty, thin


def main():
    source = RES / 'master_candidates.tsv'
    if not source.exists():
        raise SystemExit(f'{source.name} not found; run the master stage')
    frame = pd.read_csv(source, sep='\t', low_memory=False)
    empty, thin = audit(frame)

    print(f'{len(frame):,} rows, {len(COLUMNS)} declared columns\n')
    if thin:
        print('THIN (<1%) - expected where the evidence itself is rare:')
        for column, count in sorted(thin, key=lambda x: x[1]):
            print(f'   {column:38s} {count:>6,}')
    if empty:
        print('\nEMPTY - declared but never populated. Each is a signal '
              'computed upstream and lost at the merge:')
        for column, count in empty:
            print(f'   {column:38s} {count}')
        return 1
    print('\nno column is empty: nothing is being computed and discarded.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
