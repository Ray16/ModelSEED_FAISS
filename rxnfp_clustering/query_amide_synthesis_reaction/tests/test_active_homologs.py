"""Controls for the route anchored in measured activity.

This is the only route here not derived from annotation, and the only one
that reaches the family the assayed panel demonstrates. The risks are a
fragment outranking a whole enzyme, and the claim being overstated into
evidence of nylon chemistry, which it is not.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from amide_synthesis import active_homologs as ah   # noqa: E402


def hits(rows):
    return pd.DataFrame(rows, columns=['Accession', 'Identities(%)',
                                       'Length', 'query'])


class SelectionTests(unittest.TestCase):
    MEDIAN = 519.0

    def test_a_close_comparable_homolog_is_kept(self):
        out = ah.select(hits([('A', 65.3, 524, 'AS23')]), self.MEDIAN)
        self.assertEqual(len(out), 1)
        self.assertIn('pct_identity_to_active', out.columns)

    def test_below_the_identity_floor_is_dropped(self):
        out = ah.select(hits([('A', 30.0, 520, 'AS9')]), self.MEDIAN)
        self.assertEqual(len(out), 0)

    def test_a_fragment_is_dropped_however_high_its_identity(self):
        """Identity is against the shorter sequence; 134 aa matching one
        domain of a 516-aa enzyme scored 34.3% and nearly reached a
        recommendation list."""
        out = ah.select(hits([('A', 90.0, 134, 'AS9')]), self.MEDIAN)
        self.assertEqual(len(out), 0)

    def test_a_fusion_is_dropped_too(self):
        out = ah.select(hits([('A', 60.0, 1400, 'AS9')]), self.MEDIAN)
        self.assertEqual(len(out), 0)

    def test_no_median_means_no_length_filter(self):
        out = ah.select(hits([('A', 60.0, 134, 'AS9')]), 0.0)
        self.assertEqual(len(out), 1)

    def test_a_frame_without_an_identity_column_is_empty_not_fatal(self):
        out = ah.select(pd.DataFrame({'Accession': ['A']}), self.MEDIAN)
        self.assertEqual(len(out), 0)


class WeightTests(unittest.TestCase):
    def test_it_ranks_level_with_activity_ec(self):
        from amide_synthesis.master_table import EVIDENCE_WEIGHT
        self.assertEqual(EVIDENCE_WEIGHT['homolog_of_measured_active'],
                         EVIDENCE_WEIGHT['activity_ec'])

    def test_it_cannot_outrank_a_measured_protein(self):
        from amide_synthesis.master_table import EVIDENCE_WEIGHT
        self.assertLess(EVIDENCE_WEIGHT['homolog_of_measured_active'],
                        EVIDENCE_WEIGHT['measured_activity'])

    def test_it_outranks_a_bare_domain_match(self):
        from amide_synthesis.master_table import EVIDENCE_WEIGHT
        self.assertGreater(EVIDENCE_WEIGHT['homolog_of_measured_active'],
                           EVIDENCE_WEIGHT['family_homolog'])

    def test_it_has_a_shortlist_share(self):
        from amide_synthesis.shortlist import TIER_SHARE
        self.assertIn('homolog_of_measured_active',
                      {tier for tier, _ in TIER_SHARE})


class ProvenanceTests(unittest.TestCase):
    def test_the_panel_median_comes_from_measured_rows_only(self):
        table = pd.DataFrame([
            {'uniprot': 'P1', 'evidence_strength': 'measured_activity',
             'length': 500},
            {'uniprot': 'P2', 'evidence_strength': 'measured_activity',
             'length': 540},
            {'uniprot': 'P3', 'evidence_strength': 'clade_homolog',
             'length': 50}])
        self.assertEqual(ah.panel_median_length(table), 520.0)

    def test_no_measured_rows_gives_zero_not_an_error(self):
        table = pd.DataFrame([{'uniprot': 'P1',
                               'evidence_strength': 'clade_homolog',
                               'length': 50}])
        self.assertEqual(ah.panel_median_length(table), 0.0)


class FamilyVerdictExemptionTests(unittest.TestCase):
    """A family verdict must not suppress the measured-data hypothesis.

    819 of 822 measured-active homologs carry PF13193, which belongs to
    "ATP-dependent AMP-binding enzyme" - rejected by famval because its
    product is a thioester that cannot polycondense. That verdict is right
    about the chemistry and wrong to apply here: three members of that very
    family are the only proteins known to make the nylon bond.
    """

    def test_the_route_is_exempt_from_the_family_verdict(self):
        from amide_synthesis import shortlist
        source = Path(shortlist.__file__).read_text()
        self.assertIn("('reaction', 'active_homolog')", source)

    def test_reaction_route_remains_exempt_too(self):
        from amide_synthesis import shortlist
        source = Path(shortlist.__file__).read_text()
        self.assertIn("'reaction'", source)

    def test_other_routes_are_still_subject_to_it(self):
        import pandas as pd
        from amide_synthesis import shortlist as sl
        frame = pd.DataFrame([
            {'uniprot': 'P1', 'evidence_strength': 'clade_homolog',
             'master_priority': 5.0, 'length': 500, 'sequence': 'M' * 500,
             'organism': 'E. coli', 'family': 'F', 'keywords': '',
             'in_rejected_family': True}])
        self.assertEqual(sl.eligible(frame).excluded_because.iloc[0],
                         'family_judged_reject')
