"""Controls for the EC-class substrate ceiling.

The risks here are an EC absent-vs-zero confusion (which would turn "no amide
chemistry in this class" into "not looked up") and the ceiling leaking into
the ranking it must not touch.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from amide_synthesis import ec_substrate as es   # noqa: E402


def mapping(rows):
    return pd.DataFrame(rows, columns=['RHEA_ID', 'DIRECTION',
                                       'MASTER_ID', 'ID'])


def scores(rows):
    return pd.DataFrame(rows, columns=['rhea', 'substrate_distance'])


class CeilingTests(unittest.TestCase):
    def test_the_best_reaction_in_the_class_wins(self):
        out = es.ceilings(scores([(100, 0.2), (200, 0.8)]),
                          mapping([(100, 'UN', 100, '6.3.1.1'),
                                   (200, 'UN', 200, '6.3.1.1')]))
        self.assertAlmostEqual(out['6.3.1.1'][0], 0.8)
        self.assertEqual(out['6.3.1.1'][1], 200)

    def test_the_whole_direction_group_is_scanned(self):
        """rhea2ec keys on the master; the score may sit on a direction."""
        out = es.ceilings(scores([(103, 0.7)]),
                          mapping([(100, 'LR', 100, '6.3.2.1')]))
        self.assertAlmostEqual(out['6.3.2.1'][0], 0.7)

    def test_an_ec_with_no_amide_chemistry_is_absent_not_zero(self):
        """Absent means 'this class does none'; 0.0 would read as unlooked-up."""
        out = es.ceilings(scores([(100, 0.0)]),
                          mapping([(100, 'UN', 100, '1.1.1.1')]))
        self.assertNotIn('1.1.1.1', out)

    def test_partial_ec_numbers_are_ignored(self):
        out = es.ceilings(scores([(100, 0.9)]),
                          mapping([(100, 'UN', 100, '6.3.1.-')]))
        self.assertEqual(out, {})

    def test_a_malformed_row_does_not_abort_the_rest(self):
        out = es.ceilings(scores([(100, 0.9)]),
                          mapping([(None, 'UN', None, '6.3.1.1'),
                                   (100, 'UN', 100, '6.3.2.2')]))
        self.assertIn('6.3.2.2', out)


class AttachTests(unittest.TestCase):
    def _table(self):
        return pd.DataFrame([
            {'uniprot': 'P1', 'ec': '6.3.1.1',
             'substrate_distance_measured': 0.2,
             'substrate_distance_inherited': 0.0},
            {'uniprot': 'P2', 'ec': '6.3.1.1; 1.1.1.1',
             'substrate_distance_measured': 0.9,
             'substrate_distance_inherited': 0.0},
            {'uniprot': 'P3', 'ec': '',
             'substrate_distance_measured': 0.0,
             'substrate_distance_inherited': 0.5},
        ])

    def _ceiling(self):
        return {'6.3.1.1': (0.8, 200)}

    def test_the_ceiling_is_attached_where_the_ec_matches(self):
        out = es.attach(self._table(), self._ceiling()).set_index('uniprot')
        self.assertAlmostEqual(out.loc['P1', 'ec_substrate_ceiling'], 0.8)
        self.assertEqual(out.loc['P1', 'ec_substrate_source_ec'], '6.3.1.1')

    def test_a_protein_with_no_ec_gets_none_not_zero(self):
        out = es.attach(self._table(), self._ceiling()).set_index('uniprot')
        self.assertTrue(pd.isna(out.loc['P3', 'ec_substrate_ceiling']))

    def test_the_best_of_several_ecs_is_used(self):
        out = es.attach(self._table(), {'6.3.1.1': (0.4, 1),
                                        '1.1.1.1': (0.7, 2)})
        self.assertAlmostEqual(out.set_index('uniprot').loc[
            'P2', 'ec_substrate_ceiling'], 0.7)

    def test_the_discovery_flag_fires_only_when_the_class_does_better(self):
        out = es.attach(self._table(), self._ceiling()).set_index('uniprot')
        self.assertTrue(bool(out.loc['P1', 'ec_beats_own_reaction']))
        self.assertFalse(bool(out.loc['P2', 'ec_beats_own_reaction']))

    def test_inherited_distance_also_counts_as_the_protein_s_own(self):
        table = self._table()
        table.loc[0, 'substrate_distance_inherited'] = 0.95
        out = es.attach(table, self._ceiling()).set_index('uniprot')
        self.assertFalse(bool(out.loc['P1', 'ec_beats_own_reaction']))


class RankingIsolationTests(unittest.TestCase):
    def test_the_ceiling_is_absent_from_the_priority_formula(self):
        from amide_synthesis import master_table
        source = Path(master_table.__file__).read_text().lower()
        self.assertNotIn('ec_substrate', source)


if __name__ == '__main__':
    unittest.main()
