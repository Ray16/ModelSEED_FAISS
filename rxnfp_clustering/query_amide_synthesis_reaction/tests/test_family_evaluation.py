"""Controls for the family-level verdict.

This is the decision that governs tens of thousands of candidates - a protein
admitted by family membership inherits its family's score - so the logic is
pinned rather than left to inspection.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from amide_synthesis.candidate_selection import family_evaluation as ev


def family(**kw):
    base = {
        'best_substrate_distance': 0.8, 'retains_acid': True,
        'retains_amine': True, 'aa_bb_pair': False, 'self_condensing': False,
        'donor_acids': 1, 'acceptor_amines': 1, 'donor_heavy_atoms': 10,
        'members_in_table': 100, 'members_contradicted': 0,
    }
    base.update(kw)
    return base


class VerdictTests(unittest.TestCase):
    def test_no_acylation_is_rejected(self):
        call, why = ev.verdict(family(best_substrate_distance=0.0))
        self.assertEqual(call, 'reject')
        self.assertIn('acylation', why)

    def test_a_terminal_amide_is_rejected(self):
        """A product that cannot react again cannot polycondense."""
        call, why = ev.verdict(family(retains_amine=False))
        self.assertEqual(call, 'reject')
        self.assertIn('terminal', why)

    def test_an_aa_bb_pair_is_a_lead(self):
        call, why = ev.verdict(family(aa_bb_pair=True))
        self.assertEqual(call, 'lead')
        self.assertIn('nylon-6,6', why)

    def test_a_self_condensing_monomer_is_a_lead(self):
        call, why = ev.verdict(family(self_condensing=True))
        self.assertEqual(call, 'lead')

    def test_bifunctional_on_one_side_is_a_lead(self):
        call, _ = ev.verdict(family(donor_acids=2))
        self.assertEqual(call, 'lead')
        call, _ = ev.verdict(family(acceptor_amines=2))
        self.assertEqual(call, 'lead')

    def test_monofunctional_both_sides_is_only_watched(self):
        call, why = ev.verdict(family())
        self.assertEqual(call, 'watch')
        self.assertIn('monofunctional', why)

    def test_a_bulky_donor_scaffold_is_downgraded(self):
        """FPGS has the bond chemistry but drags a pterin into the pocket."""
        call, why = ev.verdict(family(donor_acids=2, donor_heavy_atoms=32))
        self.assertEqual(call, 'watch')
        self.assertIn('scaffold', why)

    def test_an_unknown_scaffold_size_does_not_downgrade(self):
        call, _ = ev.verdict(family(donor_acids=2, donor_heavy_atoms=None))
        self.assertEqual(call, 'lead')


class MemberContradictionTests(unittest.TestCase):
    """Gamma-glutamyltranspeptidase: good seeds, hydrolase members."""

    def test_a_contradicted_majority_is_rejected(self):
        call, why = ev.verdict(family(
            aa_bb_pair=True, members_in_table=1594,
            members_contradicted=1550))
        self.assertEqual(call, 'reject')
        self.assertIn('1550 of 1594', why)

    def test_a_contradicted_minority_does_not_reject(self):
        call, _ = ev.verdict(family(
            aa_bb_pair=True, members_in_table=1000,
            members_contradicted=100))
        self.assertEqual(call, 'lead')

    def test_seed_chemistry_alone_would_have_passed_it(self):
        """Guards against the check being quietly removed."""
        call, _ = ev.verdict(family(aa_bb_pair=True, members_in_table=1594,
                                    members_contradicted=0))
        self.assertEqual(call, 'lead')

    def test_zero_members_does_not_divide_by_zero(self):
        call, _ = ev.verdict(family(aa_bb_pair=True, members_in_table=0,
                                    members_contradicted=0))
        self.assertEqual(call, 'lead')


class ChemistryAggregationTests(unittest.TestCase):
    def test_the_best_scoring_reaction_supplies_the_donor_scaffold(self):
        nylon = ('OC(=O)CCCCC(=O)O.NCCCCCCN'
                 '>>OC(=O)CCCCC(=O)NCCCCCCN')
        out = ev.family_chemistry([('nylon diad', nylon, 0.965)])
        self.assertTrue(out['aa_bb_pair'])
        self.assertEqual(out['donor_acids'], 2)
        self.assertEqual(out['acceptor_amines'], 2)
        self.assertEqual(out['donor_heavy_atoms'], 10)
        self.assertEqual(out['example_reaction'], 'nylon diad')

    def test_an_unparseable_reaction_is_skipped_not_fatal(self):
        out = ev.family_chemistry([('broken', 'not a reaction', 0.5)])
        self.assertEqual(out['best_substrate_distance'], 0.0)

    def test_features_are_maxed_across_several_reactions(self):
        mono = 'CC(=O)O.NCCCCCCN>>CC(=O)NCCCCCCN'
        nylon = 'OC(=O)CCCCC(=O)O.NCCCCCCN>>OC(=O)CCCCC(=O)NCCCCCCN'
        out = ev.family_chemistry([('mono', mono, 0.4),
                                   ('nylon', nylon, 0.965)])
        self.assertTrue(out['aa_bb_pair'])
        self.assertEqual(out['example_reaction'], 'nylon')


if __name__ == '__main__':
    unittest.main()
