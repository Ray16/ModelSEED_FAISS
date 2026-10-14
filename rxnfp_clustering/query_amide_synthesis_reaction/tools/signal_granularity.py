#!/usr/bin/env python
"""Is a signal a property of the PROTEIN, or of its family? Measured.

THE RECURRING ERROR THIS CATCHES. Four times now a signal has been treated
as per-protein evidence and turned out to be inherited:

  bifunctional_ligand_in_the_pocket  restated the protein's own curated
                                     reaction for 1,753 of 1,754 candidates
                                     carrying it; headline 28 -> 17
  room_for_adipate                   all 166 passing members of EC 6.3.1.8
                                     carry identical spans, inherited from
                                     a ligand bound in the family
  sabio_* kinetics                   measured on a relative, attached by EC
  substrate_distance_inherited       says so in the column name, and is
                                     still the value behind many rows

Each was found by hand, after the claim had been made. This measures it for
every signal at once, so the next one is found before.

HOW. For each family with enough members, count distinct values of a column.
A column constant within nearly every family is describing the family. One
that varies is describing the protein. `length` is the control at 0%
constant; anything near it is genuinely per-protein.

THE ALL-ABSENT TRAP, which this tool would otherwise fall into itself. A
column that is False or null for every member of a family is trivially
"constant", so a rare signal scores as family-level purely by being rare.
Constancy is therefore measured ONLY over families where at least one member
carries the signal, and the count of such families is reported alongside -
a percentage over three families means nothing.

WHAT A HIGH SCORE DOES NOT MEAN. It is not an accusation that the signal is
wrong. Family members genuinely do the same chemistry, so a real chemical
property SHOULD be family-constant. What it means is narrower and more
useful: that signal cannot discriminate within a family, so several
same-family candidates agreeing on it are not independent evidence, and a
convergence count built on it is a statement about the family rather than
about any one protein.

Usage:
    python tools/signal_granularity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from amide_synthesis.paths import RES   # noqa: E402

MIN_MEMBERS = 5
# Below this, a percentage is noise rather than a measurement.
MIN_INFORMATIVE_FAMILIES = 10
SIDECARS = ('pocket_capacity.tsv', 'ligand_evidence.tsv',
            'evidence_lines.tsv', 'kinetics_per_candidate.tsv')
# `length` is the control: genuinely per-protein, expected near 0%.
COLUMNS = (
    'length', 'master_priority', 'substrate_distance_measured',
    'substrate_distance_inherited', 'substrate_distance_own_rhea',
    'acylates_free_diamine', 'self_condensation', 'struct_S',
    'room_for_hmda', 'room_for_adipate', 'amine_span', 'acid_span',
    'any_bifunctional_ligand', 'curated_bifunctional_ligand',
    'n_discriminating_lines', 'sabio_has_kinetics',
)


def _informative(series):
    """Does this row carry the signal at all?"""
    numeric = pd.to_numeric(series, errors='coerce')
    if numeric.notna().any():
        return numeric.fillna(0) != 0
    return series.notna() & series.astype(str).str.strip().ne('')


def granularity(table, group='pfam', columns=COLUMNS):
    """Per column: how often it is constant inside a family that carries it."""
    keys = table[group].fillna('(none)').astype(str)
    sizes = keys.value_counts()
    big = set(sizes[sizes >= MIN_MEMBERS].index) - {'(none)'}
    frame = table[keys.isin(big)].copy()
    frame['_group'] = keys[keys.isin(big)]
    rows = []
    for column in columns:
        if column not in frame.columns:
            continue
        carries = _informative(frame[column])
        # Only families where the signal is present somewhere.
        live = frame[['_group', column]][carries.groupby(
            frame._group).transform('any')]
        groups = live.groupby('_group')[column].nunique(dropna=True)
        if len(groups) < MIN_INFORMATIVE_FAMILIES:
            rows.append({'signal': column, 'families': len(groups),
                         'constant_pct': None})
            continue
        rows.append({'signal': column, 'families': int(len(groups)),
                     'constant_pct': round(100 * float((groups <= 1).mean()),
                                           1)})
    out = pd.DataFrame(rows, columns=['signal', 'families', 'constant_pct'])
    if out.empty:
        return out
    return out.sort_values('constant_pct', ascending=False,
                           na_position='last')


def main():
    table = pd.read_csv(RES / 'master_candidates_enriched.tsv', sep='\t',
                        low_memory=False)
    for name in SIDECARS:
        path = RES / name
        if not path.exists():
            continue
        extra = pd.read_csv(path, sep='\t', low_memory=False)
        keep = [c for c in extra.columns
                if c != 'uniprot' and c not in table.columns]
        if keep:
            table = table.merge(extra[['uniprot'] + keep].drop_duplicates(
                'uniprot'), on='uniprot', how='left')

    out = granularity(table)
    print(f'constancy within Pfam families of >= {MIN_MEMBERS} members, '
          'measured only over families that carry the signal\n')
    print(f'{"signal":34s} {"families":>9s} {"constant":>9s}')
    for _, row in out.iterrows():
        shown = ('too few' if pd.isna(row.constant_pct)
                 else f'{row.constant_pct:.1f}%')
        print(f'  {row.signal:32s} {row.families:9d} {shown:>9s}')
    print('\n`length` is the control: genuinely per-protein. A signal near '
          'it discriminates between\nfamily members; one near 100% describes '
          'the family, so several same-family candidates\nagreeing on it are '
          'not independent evidence.')
    out.to_csv(RES / 'signal_granularity.tsv', sep='\t', index=False)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
