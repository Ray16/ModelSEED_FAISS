"""Controls for the acceptor-chemistry cohort.

The cohort is defined by MEASURED kinetics on a free polyamine acceptor, so
the load-bearing decisions are which parameters count as evidence of
turnover and which candidates are bench-eligible.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from amide_synthesis import shortlist as sl   # noqa: E402
from tools import acceptor_cohort as ac      # noqa: E402


def _kin(rows):
    return pd.DataFrame(rows, columns=['ec', 'enzyme_name', 'species',
                                       'parameter', 'value'])


class TestAcceptorECs(unittest.TestCase):

    def test_km_on_a_free_polyamine_qualifies(self):
        out = ac.acceptor_ecs(_kin([
            ('6.3.1.8', 'gss', 'Spermidine', 'Km', 7424.0)]))
        self.assertEqual(list(out.ec), ['6.3.1.8'])

    def test_ki_alone_does_not_qualify(self):
        """A Ki says a molecule binds, not that it is turned over.

        EC 2.3.1.87 shows an apparent C2-C6 acyl series that is entirely
        bisubstrate-analogue inhibition constants.
        """
        out = ac.acceptor_ecs(_kin([
            ('2.3.1.87', 'aanat', 'Spermidine', 'Ki', 6.5)]))
        self.assertTrue(out.empty)

    def test_non_acceptor_substrates_are_ignored(self):
        out = ac.acceptor_ecs(_kin([
            ('1.1.1.1', 'x', 'Acetyl-CoA', 'Km', 1.0),
            ('1.1.1.1', 'x', 'ATP', 'Km', 1.0)]))
        self.assertTrue(out.empty)

    def test_hmda_counts_as_an_acceptor_if_it_ever_appears(self):
        for name in ('Hexamethylenediamine', '1,6-Diaminohexane',
                     'Hexane-1,6-diamine'):
            self.assertIn(name, ac.ACCEPTORS)


class TestCohort(unittest.TestCase):

    def _master(self, **over):
        base = {
            'uniprot': ['P1', 'P2', 'P3'],
            'ec': ['6.3.1.8', '2.3.1.57', '9.9.9.9'],
            'gene': ['gss', 'speG', 'other'],
            'organism': ['E. coli'] * 3,
            'lineage': ['Bacteria; X'] * 3,
            'sequence': ['M' * 400] * 3,
            # Below the floor, whatever the floor currently is.
            'length': [619.0, float(sl.MIN_LENGTH - 1), 400.0],
            'keywords': [''] * 3,
            'pdb': ['2IO7;', '1ABC;', None],
            'alphafold': [None] * 3,
            'reviewed': ['reviewed'] * 3,
            'evidence_strength': ['exact_reaction'] * 3,
            'master_priority': [7.0, 6.0, 5.0],
            'substrate_distance_measured': [0.73, 0.55, 0.9],
        }
        base.update(over)
        return pd.DataFrame(base)

    def test_only_the_named_ecs_are_included(self):
        out = ac.cohort(self._master(), ['6.3.1.8', '2.3.1.57'])
        self.assertEqual(sorted(out.uniprot), ['P1', 'P2'])

    def test_short_member_with_a_structure_survives(self):
        """Under the floor; the solved structure is the exemption."""
        out = ac.cohort(self._master(), ['2.3.1.57'])
        self.assertEqual(list(out.uniprot), ['P2'])

    def test_short_member_without_a_structure_is_dropped(self):
        out = ac.cohort(self._master(pdb=[None, None, None]), ['2.3.1.57'])
        self.assertTrue(out.empty)

    def test_matched_ec_is_recorded(self):
        out = ac.cohort(self._master(), ['6.3.1.8'])
        self.assertEqual(out.matched_ec.iloc[0], '6.3.1.8')


if __name__ == '__main__':
    unittest.main()
