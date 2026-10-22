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


class TestRepresentatives(unittest.TestCase):
    """A handful a bench can order, not the whole EC class."""

    def _master(self, n=20, **over):
        base = {
            'uniprot': [f'P{i}' for i in range(n)],
            'gene': ['panC'] * n,
            'organism': ['E. coli'] * n,
            'lineage': ['Bacteria; Proteobacteria'] * n,
            'ec': ['6.3.2.1'] * n,
            'length': [283] * n,
            'reviewed': ['reviewed'] * n,
            'pdb': [None] * n,
            'alphafold': [None] * n,
            'substrate_distance_measured': [0.4] * n,
        }
        base.update(over)
        return pd.DataFrame(base)

    def test_capped_per_ec(self):
        out = sw.representatives(self._master(20), '6.3.2.1')
        self.assertEqual(len(out), sw.PER_EC)

    def test_unexpressible_lengths_excluded(self):
        master = self._master(3, length=[283, 40, 3000])
        out = sw.representatives(master, '6.3.2.1')
        self.assertEqual(list(out.uniprot), ['P0'])

    def test_reviewed_and_structure_rank_first(self):
        master = self._master(3, reviewed=['', 'reviewed', 'reviewed'],
                              pdb=[None, None, '1ABC'])
        out = sw.representatives(master, '6.3.2.1', limit=3)
        self.assertEqual(list(out.uniprot)[:2], ['P2', 'P1'])

    def test_domain_of_life_is_recorded(self):
        """All 326 carnosine synthases are eukaryotic; that must be visible."""
        master = self._master(2, lineage=['Bacteria; X', 'Eukaryota; Metazoa'])
        out = sw.representatives(master, '6.3.2.1', limit=2)
        flags = dict(zip(out.uniprot, out.prokaryote))
        self.assertTrue(flags['P0'])
        self.assertFalse(flags['P1'])

    def test_other_ecs_are_not_pulled_in(self):
        master = self._master(2, ec=['6.3.2.1', '6.3.2.11'])
        out = sw.representatives(master, '6.3.2.1', limit=5)
        self.assertEqual(list(out.uniprot), ['P0'])


class TestAcceptorSeries(unittest.TestCase):
    """HMDA is an acceptor, so the diamine series is the relevant half."""

    def test_hmda_spellings_are_all_recognised(self):
        for name in ('Hexamethylenediamine', '1,6-Diaminohexane',
                     'Hexane-1,6-diamine'):
            self.assertIn(name, sw.FREE_DIAMINE[6])

    def test_diamine_series_is_ordered_by_carbons(self):
        self.assertEqual(sorted(sw.FREE_DIAMINE), [2, 3, 4, 5, 6])

    def test_putrescine_and_cadaverine_are_the_c4_c5_rungs(self):
        self.assertIn('Putrescine', sw.FREE_DIAMINE[4])
        self.assertIn('Cadaverine', sw.FREE_DIAMINE[5])

    def test_glycated_cadaverine_is_not_the_c5_rung(self):
        """`Ribulosecadaverine` is an adduct on a kinase, not cadaverine."""
        self.assertNotIn('Ribulosecadaverine', sw.FREE_DIAMINE[5])
        out = sw.walks(_kin([
            ('2.7.1.172', 'kinase', 'Ribulosecadaverine', 'Km', 340, 'uM',
             None)]), sw.FREE_DIAMINE)
        self.assertTrue(out.empty)

    def test_acceptor_series_are_registered(self):
        self.assertIn('free diamine acceptor', sw.SERIES)
        self.assertIn('polyamine acceptor', sw.SERIES)
