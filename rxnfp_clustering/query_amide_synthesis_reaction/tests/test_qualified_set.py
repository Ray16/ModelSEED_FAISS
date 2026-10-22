"""Controls for the uncapped qualified set.

This file answers a different question from `shortlist` - what the evidence
supports, rather than which 250 to put on a bench - so the controls are about
keeping those two apart: the bar must be applied, the caps must not be, and
bench eligibility must still exclude a protein nobody can express.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from amide_synthesis.evaluation import qualified_set as qs   # noqa: E402


def _table(n, **over):
    """A frame that clears bench eligibility by default."""
    base = {
        'uniprot': [f'P{i}' for i in range(n)],
        'gene': ['murE'] * n,
        'organism': ['E. coli'] * n,
        'sequence': ['M' * 500] * n,
        'length': [500] * n,
        'keywords': [''] * n,
        'master_priority': list(range(n, 0, -1)),
        'evidence_strength': ['exact_reaction'] * n,
        'discovery_route': ['reaction'] * n,
        'pfam': ['PF01225'] * n,
        'substrate_distance_measured': [0.8] * n,
    }
    base.update(over)
    return pd.DataFrame(base)


def _evidence(n, lines):
    return pd.DataFrame({
        'uniprot': [f'P{i}' for i in range(n)],
        'n_discriminating_lines': lines,
        'evidence_lines': ['own_reaction_chemistry+x'] * n,
    })


class TestBarIsApplied(unittest.TestCase):

    def test_below_the_bar_is_excluded(self):
        out = qs.qualified(_table(3), _evidence(3, [0, 1, 2]), set())
        self.assertEqual(list(out.uniprot), ['P2'])

    def test_missing_line_count_is_not_admitted(self):
        evidence = _evidence(2, [2, 2])
        evidence.loc[1, 'n_discriminating_lines'] = None
        out = qs.qualified(_table(2), evidence, set())
        self.assertEqual(list(out.uniprot), ['P0'])


class TestCapsAreNotApplied(unittest.TestCase):
    """The whole point: 40 proteins were invisible because of these caps."""

    def test_many_from_one_family_all_survive(self):
        n = 30
        out = qs.qualified(_table(n), _evidence(n, [2] * n), set())
        self.assertEqual(len(out), n,
                         'a per-family cap has leaked in from shortlist')

    def test_many_from_one_genus_all_survive(self):
        n = 20
        table = _table(n, organism=['Escherichia coli'] * n)
        out = qs.qualified(table, _evidence(n, [2] * n), set())
        self.assertEqual(len(out), n)


class TestBenchEligibilityStillApplies(unittest.TestCase):

    def test_transmembrane_is_still_excluded(self):
        table = _table(2, keywords=['', 'Transmembrane'])
        out = qs.qualified(table, _evidence(2, [2, 2]), set())
        self.assertEqual(list(out.uniprot), ['P0'])

    def test_no_sequence_is_still_excluded(self):
        table = _table(2, sequence=['M' * 500, ''])
        out = qs.qualified(table, _evidence(2, [2, 2]), set())
        self.assertEqual(list(out.uniprot), ['P0'])


class TestOrdering(unittest.TestCase):

    def test_evidence_outranks_priority(self):
        """A 4-line candidate must precede a 2-line one with higher weight."""
        table = _table(2, master_priority=[1.0, 99.0])
        out = qs.qualified(table, _evidence(2, [4, 2]), set())
        self.assertEqual(list(out.uniprot), ['P0', 'P1'])


class TestShortlistFlag(unittest.TestCase):

    def test_membership_is_recorded_both_ways(self):
        out = qs.qualified(_table(2), _evidence(2, [2, 2]), {'P0'})
        flags = dict(zip(out.uniprot, out.on_bench_shortlist))
        self.assertTrue(flags['P0'])
        self.assertFalse(flags['P1'])


if __name__ == '__main__':
    unittest.main()
