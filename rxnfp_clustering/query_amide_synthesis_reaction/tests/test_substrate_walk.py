"""Controls for the substrate-walk detector.

Both regex failure modes were observed on this exact data, which is why the
series are matched on exact names: `adipate` matched 2-oxoadipate and
L-2-aminoadipate (lysine-pathway metabolites, not adipate), while
`5-?aminovalerate` matched nothing because SABIO spells it "5-Aminovaleric
acid". One invents a hit, the other hides one.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import substrate_walk as sw   # noqa: E402


def _kin(rows):
    return pd.DataFrame(rows, columns=['ec', 'enzyme_name', 'species',
                                       'parameter', 'value', 'unit',
                                       'uniprot_ids'])


class TestWalks(unittest.TestCase):

    def test_rungs_are_collected_per_ec(self):
        out = sw.walks(_kin([
            ('6.3.2.1', 'panC', 'beta-Alanine', 'Km', 0.8, 'mM', 'P31663'),
            ('6.3.2.1', 'panC', '5-Aminovaleric acid', 'Km', 72, 'mM', None),
        ]), sw.OMEGA_AMINO)
        self.assertEqual(out.iloc[0].rungs, 'C3,C5')
        self.assertEqual(int(out.iloc[0].longest), 5)
        self.assertFalse(bool(out.iloc[0].reaches_target))

    def test_reaching_c6_is_flagged(self):
        out = sw.walks(_kin([
            ('x', 'e', 'beta-Alanine', 'Km', 1, 'mM', None),
            ('x', 'e', '6-Aminohexanoate', 'Km', 1, 'mM', None),
        ]), sw.OMEGA_AMINO)
        self.assertTrue(bool(out.iloc[0].reaches_target))

    def test_unknown_substrates_are_ignored(self):
        out = sw.walks(_kin([
            ('x', 'e', 'ATP', 'Km', 1, 'mM', None),
        ]), sw.OMEGA_AMINO)
        self.assertTrue(out.empty)

    def test_duplicate_rung_counts_once(self):
        out = sw.walks(_kin([
            ('x', 'e', '4-Aminobutanoate', 'Km', 1, 'mM', None),
            ('x', 'e', 'gamma-Aminobutyric acid', 'Km', 2, 'mM', None),
        ]), sw.OMEGA_AMINO)
        self.assertEqual(int(out.iloc[0].n_rungs), 1)
        self.assertEqual(out.iloc[0].rungs, 'C4')


class TestExactNamesNotRegexes(unittest.TestCase):
    """The two observed failure modes, pinned."""

    def test_lysine_pathway_metabolites_are_not_adipate(self):
        out = sw.walks(_kin([
            ('x', 'e', '2-Oxoadipate', 'Km', 1, 'mM', None),
            ('x', 'e', 'L-2-Aminoadipate', 'Km', 1, 'mM', None),
        ]), sw.DIACID)
        self.assertTrue(out.empty, 'a substring match on "adipate" is wrong')

    def test_sabio_spelling_of_c5_is_recognised(self):
        self.assertIn('5-Aminovaleric acid', sw.OMEGA_AMINO[5])

    def test_glycated_cadaverines_are_not_in_any_series(self):
        names = {n for v in sw.OMEGA_AMINO.values() for n in v}
        names |= {n for v in sw.DIACID.values() for n in v}
        self.assertNotIn('Ribulosecadaverine', names)


if __name__ == '__main__':
    unittest.main()
