"""Controls for substrate-level scoring, recall routing, and family expansion.

Same discipline as `scripts/test_screens.py`: every case below is a claim that
was asserted somewhere in prose or a comment, and several are bugs that were
actually shipped. Run after touching `substrate_distance.py`,
`expand_families.py`, or the selection logic in `amide_enzymes.py`.

No network. Family expansion is exercised through injected fakes.
"""
from __future__ import annotations

import unittest

import pandas as pd

from amide_synthesis import substrate_distance as sd
from amide_synthesis.acylation import acylation_events
from amide_synthesis.substrate_distance import (describe_event, score_event,
                                                score_reaction)
from amide_synthesis import expand_families as ef

# Reaction SMILES, pH-7 convention, matching the corpus.
NYLON_DIAD = ('O=C([O-])CCCCC(=O)[O-].[NH3+]CCCCCC[NH3+]'
              '>>[NH3+]CCCCCCNC(=O)CCCCC(=O)[O-].O')
SELF_CONDENSE_C6 = ('[NH3+]CCCCCCC(=O)[O-].[NH3+]CCCCCCC(=O)[O-]'
                    '>>[NH3+]CCCCCCC(=O)NCCCCCCC(=O)[O-].O')
ACETYL_PUTRESCINE = 'CC(=O)[O-].[NH3+]CCCC[NH3+]>>CC(=O)NCCCC[NH3+].O'
TERMINAL_ACETAMIDE = 'CC(=O)[O-].[NH3+]CC>>CC(=O)NCC.O'
AMIDE_HYDROLYSIS = 'CC(=O)NCC.O>>CC(=O)[O-].[NH3+]CC'


class TestAcylDonorReconstruction(unittest.TestCase):
    """The donor half must be recovered as its parent ACID, not the acyl group.

    Without the added hydroxyl an adipoyl donor looks like a monoacid and the
    AA/BB pairing that defines nylon-6,6 is never detected.
    """

    def test_adipate_donor_recovers_two_carboxyls(self):
        events = acylation_events(NYLON_DIAD)
        self.assertTrue(events)
        features = describe_event(events[0])
        self.assertEqual(features['acyl_n_acids'], 2,
                         'adipoyl must reconstitute as a DIacid')

    def test_acetyl_donor_is_a_monoacid(self):
        features = describe_event(acylation_events(ACETYL_PUTRESCINE)[0])
        self.assertEqual(features['acyl_n_acids'], 1)


class TestPolycondensationLogic(unittest.TestCase):
    """The three amide classes the boolean screen cannot separate."""

    def test_nylon_diad_is_an_aa_bb_pair(self):
        features = describe_event(acylation_events(NYLON_DIAD)[0])
        self.assertTrue(features['aa_bb_pair'])
        self.assertFalse(features['self_condensation'])

    def test_bifunctional_monomer_is_self_condensing(self):
        features = describe_event(acylation_events(SELF_CONDENSE_C6)[0])
        self.assertTrue(features['self_condensation'],
                        'donor and acceptor are the same species')

    def test_acetyl_on_a_diamine_cannot_polymerise(self):
        """A C2 monoacid donor can never extend a chain.

        This is the GNAT case. The acceptor IS a free diamine, so the boolean
        screen flags it exactly like a nylon-relevant reaction; only the donor
        distinguishes them.
        """
        features = describe_event(acylation_events(ACETYL_PUTRESCINE)[0])
        self.assertFalse(features['aa_bb_pair'])
        self.assertFalse(features['self_condensation'])
        self.assertLess(score_event(features),
                        score_reaction(NYLON_DIAD)[0] - 0.3)

    def test_terminal_amide_scores_zero(self):
        self.assertEqual(score_reaction(TERMINAL_ACETAMIDE)[0], 0.0)

    def test_hydrolysis_is_not_formation(self):
        self.assertEqual(score_reaction(AMIDE_HYDROLYSIS)[0], 0.0)

    def test_nylon_outranks_every_other_control(self):
        nylon = score_reaction(NYLON_DIAD)[0]
        for rxn in (ACETYL_PUTRESCINE, TERMINAL_ACETAMIDE, AMIDE_HYDROLYSIS):
            self.assertGreater(nylon, score_reaction(rxn)[0])

    def test_score_is_bounded(self):
        for rxn in (NYLON_DIAD, SELF_CONDENSE_C6, ACETYL_PUTRESCINE,
                    TERMINAL_ACETAMIDE, AMIDE_HYDROLYSIS):
            self.assertGreaterEqual(score_reaction(rxn)[0], 0.0)
            self.assertLessEqual(score_reaction(rxn)[0], 1.0)


FPGS_LIKE = ('[NH3+]C(CCC(=O)[O-])C(=O)[O-].'
             'O=C([O-])C(CCC(=O)[O-])NC(=O)c1ccc(NCC2CNc3nc(N)[nH]c(=O)c3N2)cc1'
             '>>[O-]C(=O)C(CCC(=O)NC(CCC(=O)[O-])C(=O)[O-])'
             'NC(=O)c1ccc(NCC2CNc3nc(N)[nH]c(=O)c3N2)cc1.O')


class TestScaffoldParsimony(unittest.TestCase):
    """Forming the right BOND is not accepting the right MOLECULE.

    Folylpolyglutamate synthetase genuinely runs iterative polyamide
    extension onto a free amine and scored 0.883 on bond chemistry alone -
    above every NIS synthetase. Its substrate is tetrahydrofolyl-
    polyglutamate and its pocket grips a pterin, so it will never see a C6
    diacid. The scaffold has to be counted.
    """

    def test_parent_donor_is_measured_not_the_cut_fragment(self):
        """The acyl half is small by construction; the parent carries the pterin."""
        features = describe_event(acylation_events(FPGS_LIKE)[0])
        self.assertGreater(features['donor_parent_heavy_atoms'], 25,
                           'the pterin-bearing parent must be the thing measured')

    def test_bulky_scaffold_scores_below_the_bare_monomer(self):
        self.assertGreater(score_reaction(NYLON_DIAD)[0],
                           score_reaction(FPGS_LIKE)[0])

    def test_parsimony_is_perfect_for_the_nylon_monomers(self):
        features = describe_event(acylation_events(NYLON_DIAD)[0])
        self.assertAlmostEqual(features['parsimony'], 1.0, places=2)

    def test_parsimony_cannot_rescue_a_terminal_amide(self):
        """Small and clean is not a virtue if the chemistry cannot polymerise."""
        self.assertEqual(score_reaction(TERMINAL_ACETAMIDE)[0], 0.0)


class TestSpacerGeometry(unittest.TestCase):
    """Chain length must be measured, since it is the stated gap to the NIS
    enzymes (native monomers are two carbons shorter per arm)."""

    def test_hmda_spacer_is_six_carbons(self):
        features = describe_event(acylation_events(NYLON_DIAD)[0])
        self.assertEqual(features['amine_spacer'], 6)

    def test_adipate_spacer_is_four_carbons(self):
        features = describe_event(acylation_events(NYLON_DIAD)[0])
        self.assertEqual(features['acid_spacer'], 4)

    def test_shorter_arms_score_lower_than_exact_match(self):
        short = ('O=C([O-])CCC(=O)[O-].[NH3+]CCCC[NH3+]'
                 '>>[NH3+]CCCCNC(=O)CCC(=O)[O-].O')   # succinate + putrescine
        self.assertGreater(score_reaction(NYLON_DIAD)[0],
                           score_reaction(short)[0])


class TestFamilyClassification(unittest.TestCase):
    """The co-occurrence bug: a passenger domain inheriting catalysis.

    Shipped once. A multi-domain protein credited ALL of its Pfam domains with
    its substrate distance, so a peptidoglycan-amidase domain (CHAP) and a
    calcium-channel domain (RyR/IP3R homology) were expanded as validated
    amide-forming families.
    """

    # Seed counts mirror the real run (PF04183: 35 seeds / 10116 members;
    # PF01225: 4 / 57656), because the enrichment verdict depends on them.
    def _families(self, nis_seeds=35):
        rows = [
            # A real NIS synthetase: catalytic domain plus a passenger domain.
            {'uniprot': 'P0DXD6', 'pfam': 'PF04183;PF05257',
             'family': 'IucA / IucC family', 'substrate_distance': 0.83,
             'self_condensation': True},
        ]
        rows += [{'uniprot': f'NIS{i}', 'pfam': 'PF04183',
                  'family': 'IucA / IucC family', 'substrate_distance': 0.81,
                  'self_condensation': True}
                 for i in range(nis_seeds - 1)]
        rows += [{'uniprot': f'Q9WY{79 + i}', 'pfam': 'PF01225',
                  'family': 'Mur ligase', 'substrate_distance': 0.88,
                  'self_condensation': False} for i in range(4)]
        return ef.seed_families(pd.DataFrame(rows))

    def test_single_seed_domain_is_a_passenger(self):
        sizes = {'PF04183': 10116, 'PF05257': 10093, 'PF01225': 57656}
        verdict = ef.classify_families(self._families(), sizes)
        self.assertEqual(verdict['PF05257']['route'], 'passenger',
                         'CHAP rode along on one protein; it is not catalytic')

    def test_enriched_family_expands_by_domain(self):
        sizes = {'PF04183': 10116, 'PF05257': 10093, 'PF01225': 57656}
        verdict = ef.classify_families(self._families(), sizes)
        self.assertEqual(verdict['PF04183']['route'], 'domain')

    def test_huge_ancient_family_goes_to_the_clade_route(self):
        """Mur ligase is real but universal; only a sub-clade is relevant."""
        sizes = {'PF04183': 10116, 'PF05257': 10093, 'PF01225': 57656}
        verdict = ef.classify_families(self._families(), sizes)
        self.assertEqual(verdict['PF01225']['route'], 'clade')
        self.assertIn('Q9WY79', ef.clade_seeds(verdict))

    def test_passenger_domains_are_never_expanded(self):
        sizes = {'PF04183': 10116, 'PF05257': 10093, 'PF01225': 57656}
        verdict = ef.classify_families(self._families(), sizes)
        self.assertNotIn('PF05257', ef.clade_seeds(verdict))

    def test_sparse_seeding_degrades_to_clade_not_to_domain(self):
        """Enrichment depends on how many seeds curation happened to supply.

        A family whose chemistry is genuinely characteristic but which has only
        a couple of annotated members must fall back to the (conservative)
        clade route rather than expanding the whole domain on thin evidence.
        """
        sizes = {'PF04183': 10116, 'PF05257': 10093, 'PF01225': 57656}
        verdict = ef.classify_families(self._families(nis_seeds=2), sizes)
        self.assertEqual(verdict['PF04183']['route'], 'clade')

    def test_missing_family_size_does_not_promote(self):
        """An unknown denominator must not be read as infinite enrichment."""
        verdict = ef.classify_families(self._families(), {})
        self.assertNotIn('domain', {v['route'] for v in verdict.values()})


class TestExpansionPlumbing(unittest.TestCase):
    """Offline: the tier/dedup contract, with a fake UniProt."""

    def test_first_tier_keeps_the_accession(self):
        """The SAMPLING path's dedup contract.

        Validated families are now enumerated in full instead of sampled, so
        there is one query per family and the tier is read from the member's
        own record - see EnumerationTests / MemberTierTests in
        test_expand_families.py. This contract still governs the sampling
        path, which is kept for comparison, so the test names it explicitly
        rather than relying on the default.
        """
        def fake_search(query, max_pages=1, **kw):
            if 'reviewed' in query:
                return [{'Entry': 'P1'}], False, False
            return [{'Entry': 'P1'}, {'Entry': 'P2'}], False, False

        families = {'PF04183': {'pfam': 'PF04183', 'family': 'NIS',
                                'family_substrate_distance': 0.83,
                                'n_seeds': 2, 'seed_accessions': ['A'],
                                'self_condensing': True}}
        frame, report = ef.expand(families, search=fake_search, threads=1,
                                  enumerate_all=False)
        self.assertEqual(len(frame), 2)
        tier = dict(zip(frame.Entry, frame.discovery_tier))
        self.assertEqual(tier['P1'], 'reviewed',
                         'the most informative tier must win a tie')

    def test_inherited_distance_is_labelled_as_inherited(self):
        def fake_search(query, max_pages=1, **kw):
            return [{'Entry': 'P1'}], False, False

        families = {'PF04183': {'pfam': 'PF04183', 'family': 'NIS',
                                'family_substrate_distance': 0.83,
                                'n_seeds': 2, 'seed_accessions': ['A'],
                                'self_condensing': True}}
        frame, _ = ef.expand(families, search=fake_search, threads=1)
        self.assertIn('family_substrate_distance', frame.columns)
        self.assertNotIn('substrate_distance', frame.columns,
                         'an inherited score must not pose as a measured one')
        self.assertTrue((frame.evidence_strength == 'family_homolog').all())


if __name__ == '__main__':
    unittest.main(verbosity=2)


class TargetParameterTests(unittest.TestCase):
    """The target is four numbers; nothing else in the module is nylon-6,6.

    They are overridable so the pipeline can be aimed at another polyamide
    without editing code. The defaults must not drift: every reference score
    quoted throughout this project assumes them.
    """

    def test_the_defaults_are_nylon_6_6(self):
        self.assertEqual(sd.ADIPATE_SPACER, 4)
        self.assertEqual(sd.HMDA_SPACER, 6)
        self.assertEqual(sd.ADIPATE_HEAVY, 10)
        self.assertEqual(sd.HMDA_HEAVY, 8)

    def test_the_reference_score_is_unchanged(self):
        """0.965 is quoted in the README, the deck and several commits."""
        score, _ = sd.score_reaction(
            'OC(=O)CCCCC(=O)O.NCCCCCCN>>OC(=O)CCCCC(=O)NCCCCCCN')
        self.assertAlmostEqual(score, 0.965, places=3)

    def test_retargeting_changes_which_spacer_scores_best(self):
        """Nylon-6,10 wants a C10 diacid, so sebacate should beat adipate."""
        import importlib
        import os
        sebacate = ('OC(=O)CCCCCCCCC(=O)O.NCCCCCCN'
                    '>>OC(=O)CCCCCCCCC(=O)NCCCCCCN')
        adipate = 'OC(=O)CCCCC(=O)O.NCCCCCCN>>OC(=O)CCCCC(=O)NCCCCCCN'
        before = (sd.score_reaction(adipate)[0],
                  sd.score_reaction(sebacate)[0])
        self.assertGreater(before[0], before[1])      # nylon-6,6 default
        os.environ['TARGET_ACID_SPACER'] = '8'
        os.environ['TARGET_ACID_HEAVY_ATOMS'] = '14'
        try:
            retargeted = importlib.reload(sd)
            after = (retargeted.score_reaction(adipate)[0],
                     retargeted.score_reaction(sebacate)[0])
            self.assertGreater(after[1], after[0])    # nylon-6,10 prefers C10
        finally:
            os.environ.pop('TARGET_ACID_SPACER', None)
            os.environ.pop('TARGET_ACID_HEAVY_ATOMS', None)
            importlib.reload(sd)
        self.assertEqual(sd.ADIPATE_SPACER, 4)
