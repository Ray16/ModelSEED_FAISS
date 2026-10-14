"""Controls for the signal-granularity audit.

The tool exists because four signals were treated as per-protein evidence
and turned out to be inherited. Its own load-bearing decision is the
all-absent trap: a signal that is False everywhere in a family is
trivially constant, so a rare signal scores as family-level purely by
being rare. Measuring naively said `acylates_free_diamine` was 96.8%
family-constant; measuring only over families that carry it says 53.3%.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import signal_granularity as sg   # noqa: E402


def _table(groups, values, column='signal'):
    rows = []
    for name, vals in zip(groups, values):
        for v in vals:
            rows.append({'pfam': name, column: v})
    return pd.DataFrame(rows)


class TestAllAbsentTrap(unittest.TestCase):

    def test_families_without_the_signal_are_excluded(self):
        """Ten all-False families must not make a signal look constant."""
        groups = [f'F{i}' for i in range(11)]
        values = [[0] * 5 for _ in range(10)] + [[1, 0, 1, 0, 1]]
        out = sg.granularity(_table(groups, values), columns=('signal',))
        # Only the one carrying family is informative, which is below the
        # reporting floor, so no percentage is claimed.
        self.assertEqual(int(out.iloc[0].families), 1)
        self.assertTrue(pd.isna(out.iloc[0].constant_pct))

    def test_constant_where_present_scores_high(self):
        groups = [f'F{i}' for i in range(12)]
        values = [[1] * 5 for _ in range(12)]
        out = sg.granularity(_table(groups, values), columns=('signal',))
        self.assertEqual(out.iloc[0].constant_pct, 100.0)

    def test_varying_where_present_scores_low(self):
        groups = [f'F{i}' for i in range(12)]
        values = [[1, 2, 3, 4, 5] for _ in range(12)]
        out = sg.granularity(_table(groups, values), columns=('signal',))
        self.assertEqual(out.iloc[0].constant_pct, 0.0)


class TestGrouping(unittest.TestCase):

    def test_small_families_are_ignored(self):
        groups = ['Big'] * 1 + ['Tiny'] * 1
        table = _table(['Big', 'Tiny'], [[1] * 6, [1, 2]])
        out = sg.granularity(table, columns=('signal',))
        # Tiny has 2 members, below MIN_MEMBERS, so only Big counts.
        self.assertEqual(int(out.iloc[0].families), 1)

    def test_missing_pfam_is_not_a_family(self):
        table = pd.DataFrame({'pfam': [None] * 8, 'signal': [1] * 8})
        out = sg.granularity(table, columns=('signal',))
        self.assertEqual(int(out.iloc[0].families), 0)

    def test_absent_column_is_skipped(self):
        out = sg.granularity(_table(['F'], [[1] * 6]),
                             columns=('nope',))
        self.assertTrue(out.empty)


class TestInformative(unittest.TestCase):

    def test_zero_and_false_do_not_carry_a_signal(self):
        self.assertFalse(sg._informative(pd.Series([0, 0.0])).any())

    def test_nonzero_carries(self):
        self.assertTrue(sg._informative(pd.Series([0, 3])).iloc[1])

    def test_blank_strings_do_not_carry(self):
        out = sg._informative(pd.Series(['', '  ', 'CHEBI:1']))
        self.assertEqual(list(out), [False, False, True])


if __name__ == '__main__':
    unittest.main()
