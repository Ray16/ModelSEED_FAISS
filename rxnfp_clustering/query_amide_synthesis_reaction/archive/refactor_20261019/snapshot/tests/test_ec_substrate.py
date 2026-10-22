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

from amide_synthesis import ec_substrate as es          # noqa: E402
from amide_synthesis import ec_substrate, master_table  # noqa: E402


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



class RankingIsolationTests(unittest.TestCase):
    """The CEILING stays out of the ranking; the protein's OWN reaction goes in.

    These are different columns from the same sidecar and the distinction is
    the whole point. `ec_substrate_ceiling` is a maximum over an EC class, so
    many unrelated proteins share one value and it must not score anything.
    `substrate_distance_own_rhea` is measured on the reaction UniProt
    attributes to that protein, which is the same kind of evidence as
    `substrate_distance_measured` and does feed the ranking.
    """

    def _build(self, **overrides):
        from amide_synthesis.master_table import build
        row = {'uniprot': 'P1', 'substrate_distance': 0.3,
               'evidence_strength': 'exact_reaction',
               'protein_name': 'ligase'}
        row.update(overrides)
        return build(pd.DataFrame([row]),
                     pd.DataFrame(columns=['uniprot']),
                     structural=pd.DataFrame(columns=['uniprot', 'struct_S']))

    def test_the_ceiling_does_not_change_the_priority(self):
        """Behavioural, not textual: master MAY read it, but not score on it."""
        low = self._build(ec_substrate_ceiling=0.05).master_priority.iloc[0]
        high = self._build(ec_substrate_ceiling=0.99).master_priority.iloc[0]
        self.assertAlmostEqual(low, high)

    def test_the_own_reaction_distance_DOES_change_the_priority(self):
        low = self._build(substrate_distance_own_rhea=0.0)
        high = self._build(substrate_distance_own_rhea=0.9)
        self.assertGreater(high.master_priority.iloc[0],
                           low.master_priority.iloc[0])

    def test_the_own_reaction_distance_supersedes_an_inherited_value(self):
        out = self._build(substrate_distance_own_rhea=0.9)
        self.assertEqual(out.substrate_distance_source.iloc[0],
                         'own_curated_reaction')

    def test_a_weaker_own_reaction_does_not_displace_a_measured_one(self):
        out = self._build(substrate_distance=0.8,
                          substrate_distance_own_rhea=0.2)
        self.assertEqual(out.substrate_distance_source.iloc[0],
                         'own_modelseed_reaction')

    def test_the_discovery_flag_fires_when_the_class_does_better(self):
        out = self._build(substrate_distance=0.2, ec_substrate_ceiling=0.9)
        self.assertTrue(bool(out.ec_beats_own_reaction.iloc[0]))

    def test_the_discovery_flag_is_quiet_when_it_does_not(self):
        out = self._build(substrate_distance=0.9, ec_substrate_ceiling=0.2)
        self.assertFalse(bool(out.ec_beats_own_reaction.iloc[0]))


if __name__ == '__main__':
    unittest.main()


class OwnReactionTests(unittest.TestCase):
    """The protein's own curated reaction, from its catalytic-activity line."""

    def _scores(self, rows):
        return pd.DataFrame(rows, columns=['rhea', 'substrate_distance'])

    def test_the_cited_rhea_id_is_scored(self):
        statement = ('CATALYTIC ACTIVITY: Reaction=2 D-alanine + ATP = '
                     'D-alanyl-D-alanine; Xref=Rhea:RHEA:11224, '
                     'ChEBI:CHEBI:15378; EC=6.3.2.4;')
        best, cited = es.own_reaction_distance(
            [statement], self._scores([(11224, 0.87)]))
        self.assertAlmostEqual(best[0], 0.87)
        self.assertEqual(cited[0], 11224)

    def test_the_direction_group_is_scanned_in_both_directions(self):
        """The statement may cite one member while the score sits on another."""
        best, _ = es.own_reaction_distance(
            ['Xref=Rhea:RHEA:11225'], self._scores([(11224, 0.8)]))
        self.assertAlmostEqual(best[0], 0.8)

    def test_a_statement_with_no_rhea_id_scores_none(self):
        best, cited = es.own_reaction_distance(
            ['CATALYTIC ACTIVITY: Reaction=something;'],
            self._scores([(11224, 0.8)]))
        self.assertIsNone(best[0])
        self.assertIsNone(cited[0])

    def test_a_cited_reaction_that_scores_zero_gives_none_not_zero(self):
        """None means 'no amide chemistry'; 0.0 would read as a real measure."""
        best, _ = es.own_reaction_distance(
            ['Xref=Rhea:RHEA:11224'], self._scores([(11224, 0.0)]))
        self.assertIsNone(best[0])

    def test_the_best_of_several_cited_reactions_wins(self):
        best, cited = es.own_reaction_distance(
            ['Xref=Rhea:RHEA:10000; Xref=Rhea:RHEA:20000'],
            self._scores([(10000, 0.3), (20000, 0.9)]))
        self.assertAlmostEqual(best[0], 0.9)
        self.assertEqual(cited[0], 20000)

    def test_missing_statements_do_not_raise(self):
        best, _ = es.own_reaction_distance(
            [None, '', float('nan')], self._scores([(1, 0.5)]))
        self.assertEqual(best, [None, None, None])


class TestEveryRouteIsAnnotated(unittest.TestCase):
    """`ec_substrate` must assemble annotations from EVERY protein route.

    It deliberately reads the route files rather than `master_candidates.tsv`,
    to avoid a circular dependency with `master_table`. The cost of that is a
    second hardcoded list of routes, which fell one behind: the `actives`
    route was added and its 822 proteins silently received no EC ceiling and
    no own-reaction distance, while the stage reported a healthy 54,199 and
    every file involved looked complete.

    The two lists are independent by design, so this checks them against each
    other rather than merging them.
    """

    def test_master_route_files_all_have_an_annotation_source(self):
        annotated = {name for name, _ in ec_substrate.SOURCES}
        missing = set(master_table.ROUTE_FILES) - annotated
        self.assertEqual(
            missing, set(),
            'routes carried by master_table but never annotated by '
            'ec_substrate, so their proteins get no EC ceiling: '
            f'{sorted(missing)}')

    def test_sources_are_unique(self):
        names = [name for name, _ in ec_substrate.SOURCES]
        self.assertEqual(len(names), len(set(names)))
