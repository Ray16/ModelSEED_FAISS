"""Controls for the family-expansion routes, which had none.

This module produces 32,482 of 40,725 candidates - four fifths of the table,
on the weakest evidence in it - and had no direct tests at all. These cover
the enrichment test, the passenger removal that enrichment exists for, and the
second signature axis.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from amide_synthesis import expand_families as ef   # noqa: E402


def candidate(accession, pfam='PF00501', interpro='IPR000873',
              distance=0.85, family='AMP-binding', self_condensing=False):
    return {'uniprot': accession, 'pfam': pfam, 'interpro': interpro,
            'substrate_distance': distance, 'family': family,
            'self_condensation': self_condensing}


class SeedTests(unittest.TestCase):
    def test_pfam_is_the_default_axis(self):
        seeded = ef.seed_families(pd.DataFrame([candidate('P1')]))
        self.assertIn('PF00501', seeded)

    def test_interpro_axis_reads_the_interpro_column(self):
        seeded = ef.seed_families(pd.DataFrame([candidate('P1')]),
                                  signature='interpro')
        self.assertIn('IPR000873', seeded)
        self.assertNotIn('PF00501', seeded)

    def test_accessions_of_the_other_axis_are_ignored(self):
        """An IPR in the pfam column must not be seeded as a Pfam."""
        seeded = ef.seed_families(
            pd.DataFrame([candidate('P1', pfam='IPR000873')]))
        self.assertEqual(seeded, {})

    def test_every_domain_on_a_protein_is_credited(self):
        """Any domain could be the catalytic one; passengers go later."""
        seeded = ef.seed_families(
            pd.DataFrame([candidate('P1', pfam='PF00501;PF13193')]))
        self.assertEqual(set(seeded), {'PF00501', 'PF13193'})

    def test_a_comma_separated_list_is_split(self):
        seeded = ef.seed_families(
            pd.DataFrame([candidate('P1', interpro='IPR000873,IPR020845')]),
            signature='interpro')
        self.assertEqual(set(seeded), {'IPR000873', 'IPR020845'})

    def test_a_family_below_the_distance_floor_is_dropped(self):
        seeded = ef.seed_families(
            pd.DataFrame([candidate('P1', distance=0.01)]), floor=0.40)
        self.assertEqual(seeded, {})

    def test_the_family_keeps_the_BEST_distance_of_its_seeds(self):
        seeded = ef.seed_families(pd.DataFrame([
            candidate('P1', distance=0.5), candidate('P2', distance=0.9)]))
        self.assertAlmostEqual(
            seeded['PF00501']['family_substrate_distance'], 0.9)

    def test_seeds_are_counted(self):
        seeded = ef.seed_families(pd.DataFrame([
            candidate('P1'), candidate('P2')]))
        self.assertEqual(seeded['PF00501']['n_seeds'], 2)

    def test_self_condensation_propagates_if_any_seed_has_it(self):
        seeded = ef.seed_families(pd.DataFrame([
            candidate('P1'), candidate('P2', self_condensing=True)]))
        self.assertTrue(seeded['PF00501']['self_condensing'])

    def test_a_missing_column_does_not_raise(self):
        frame = pd.DataFrame([{'uniprot': 'P1', 'substrate_distance': 0.9}])
        self.assertEqual(ef.seed_families(frame), {})


class EnrichmentTests(unittest.TestCase):
    """Enrichment exists to separate a catalytic domain from a passenger."""

    def _families(self):
        return {
            'PF_SPECIFIC': {'pfam': 'PF_SPECIFIC', 'n_seeds': 20,
                            'family_substrate_distance': 0.9, 'family': 'x',
                            'seed_accessions': ['P1'], 'self_condensing': False},
            'PF_UNIVERSAL': {'pfam': 'PF_UNIVERSAL', 'n_seeds': 5,
                             'family_substrate_distance': 0.9, 'family': 'y',
                             'seed_accessions': ['P1'], 'self_condensing': False},
            'PF_PASSENGER': {'pfam': 'PF_PASSENGER', 'n_seeds': 1,
                             'family_substrate_distance': 0.9, 'family': 'z',
                             'seed_accessions': ['P1'], 'self_condensing': False},
        }

    def _classify(self):
        sizes = {'PF_SPECIFIC': 400, 'PF_UNIVERSAL': 450000,
                 'PF_PASSENGER': 50}
        return ef.classify_families(self._families(), sizes)

    def test_a_specific_family_is_expanded_by_domain(self):
        self.assertEqual(self._classify()['PF_SPECIFIC']['route'], 'domain')

    def test_a_universal_family_goes_to_the_clade_route(self):
        """450k members at 5 seeds means the domain says almost nothing."""
        self.assertEqual(self._classify()['PF_UNIVERSAL']['route'], 'clade')

    def test_a_single_seed_family_is_a_passenger(self):
        """One seed cannot distinguish catalysis from co-occurrence."""
        self.assertEqual(self._classify()['PF_PASSENGER']['route'], 'passenger')

    def test_enrichment_is_per_thousand_members(self):
        got = self._classify()['PF_SPECIFIC']['enrichment_per_1k']
        self.assertAlmostEqual(got, 20 / 400 * 1000)


class SizeQueryTests(unittest.TestCase):
    def test_pfam_uses_the_pfam_xref(self):
        seen = []
        ef.family_size('PF00501', count=lambda q: seen.append(q) or 7)
        self.assertEqual(seen, ['xref:pfam-PF00501'])

    def test_interpro_uses_the_interpro_xref(self):
        seen = []
        ef.family_size('IPR000873', count=lambda q: seen.append(q) or 7,
                       signature='interpro')
        self.assertEqual(seen, ['xref:interpro-IPR000873'])


class ExpandTests(unittest.TestCase):
    def _families(self):
        return {'PF00501': {'pfam': 'PF00501', 'family': 'AMP-binding',
                            'family_substrate_distance': 0.85,
                            'self_condensing': True, 'n_seeds': 3,
                            'seed_accessions': ['P1']}}

    def _search(self, record):
        def search(query, size=100, max_pages=5, fields=''):
            record.append(query)
            return [{'Entry': f'Q{len(record):05d}'}], False, False
        return search

    def test_the_query_uses_the_right_xref_per_axis(self):
        seen = []
        ef.expand(self._families(), search=self._search(seen), threads=1)
        self.assertTrue(all(q.startswith('xref:pfam-PF00501') for q in seen))

        seen = []
        families = {'IPR000873': dict(self._families()['PF00501'],
                                      pfam='IPR000873')}
        ef.expand(families, search=self._search(seen), threads=1,
                  signature='interpro')
        self.assertTrue(all(q.startswith('xref:interpro-IPR000873')
                            for q in seen), seen)

    def test_every_tier_is_queried_on_the_sampling_path(self):
        """Only the legacy sampling path queries per tier; see EnumerationTests."""
        seen = []
        ef.expand(self._families(), search=self._search(seen), threads=1,
                  enumerate_all=False)
        self.assertEqual(len(seen), len(ef.TIERS))

    def test_members_inherit_the_family_evidence(self):
        frame, _ = ef.expand(self._families(), search=self._search([]),
                             threads=1)
        self.assertTrue((frame.evidence_strength == 'family_homolog').all())
        self.assertAlmostEqual(frame.family_substrate_distance.iloc[0], 0.85)
        self.assertTrue(bool(frame.family_self_condensing.iloc[0]))

    def test_saturation_is_reported(self):
        def search(query, size=100, max_pages=5, fields=''):
            return [{'Entry': 'Q1'}], False, True
        _, report = ef.expand(self._families(), search=search, threads=1)
        self.assertTrue(report.saturated.all())

class SilentEmptyAxisTests(unittest.TestCase):
    """An axis whose column is absent seeds NOTHING, and says so.

    This shipped once: `candidates_ranked.tsv` keeps only a display subset and
    carries no InterPro column, so the second axis seeded 0 signatures while
    the stage ran clean and the summary looked normal. `main` now joins the
    column from the stage-04 record and warns if it still cannot find it.
    """

    def test_an_all_nan_column_seeds_nothing(self):
        frame = pd.DataFrame([candidate('P1', interpro=float('nan'))])
        self.assertEqual(ef.seed_families(frame, signature='interpro'), {})

    def test_an_empty_string_column_seeds_nothing(self):
        frame = pd.DataFrame([candidate('P1', interpro='')])
        self.assertEqual(ef.seed_families(frame, signature='interpro'), {})

    def test_the_other_axis_is_unaffected_by_an_empty_one(self):
        frame = pd.DataFrame([candidate('P1', interpro='')])
        self.assertIn('PF00501', ef.seed_families(frame))

class EnumerationTests(unittest.TestCase):
    """Validated families are evaluated in full, not sampled.

    The tiered path retrieved 26,074 of 257,297 members (10.1%), and which
    members you got was decided by UniProt's relevance order inside a
    truncation.
    """

    def _families(self, size=1200):
        return {'PF04183': {'pfam': 'PF04183', 'family': 'IucA/IucC',
                            'family_substrate_distance': 0.83,
                            'self_condensing': False, 'n_seeds': 41,
                            'family_size': size,
                            'seed_accessions': ['P1']}}

    def _search(self, record, rows=None):
        def search(query, size=100, max_pages=5, fields=''):
            record.append({'query': query, 'size': size,
                           'max_pages': max_pages})
            return (rows if rows is not None
                    else [{'Entry': 'Q00001'}]), False, False
        return search

    def test_one_query_per_family_not_one_per_tier(self):
        seen = []
        ef.expand(self._families(), search=self._search(seen), threads=1,
                  enumerate_all=True)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0]['query'], 'xref:pfam-PF04183')

    def test_no_tier_filter_is_appended(self):
        """A filter would exclude the members it does not match."""
        seen = []
        ef.expand(self._families(), search=self._search(seen), threads=1,
                  enumerate_all=True)
        self.assertNotIn(' AND ', seen[0]['query'])

    def test_the_page_budget_covers_the_whole_family(self):
        seen = []
        ef.expand(self._families(size=1200), search=self._search(seen),
                  threads=1, enumerate_all=True)
        pages, size = seen[0]['max_pages'], seen[0]['size']
        self.assertGreaterEqual(pages * size, 1200)

    def test_a_family_that_grew_still_finishes(self):
        """Slack pages exist so a stale size does not truncate silently."""
        seen = []
        ef.expand(self._families(size=1), search=self._search(seen),
                  threads=1, enumerate_all=True)
        self.assertGreater(seen[0]['max_pages'], 1)

    def test_the_tiered_path_is_still_available(self):
        seen = []
        ef.expand(self._families(), search=self._search(seen), threads=1,
                  enumerate_all=False)
        self.assertEqual(len(seen), len(ef.TIERS))

    def test_members_inherit_family_evidence_when_enumerated(self):
        frame, _ = ef.expand(self._families(), search=self._search([]),
                             threads=1, enumerate_all=True)
        self.assertAlmostEqual(frame.family_substrate_distance.iloc[0], 0.83)
        self.assertTrue((frame.evidence_strength == 'family_homolog').all())


class MemberTierTests(unittest.TestCase):
    """The tier describes the MEMBER, not the query that found it."""

    def test_reviewed_wins(self):
        self.assertEqual(ef.member_tier({'Reviewed': 'reviewed'}), 'reviewed')

    def test_a_structure_is_recognised(self):
        self.assertEqual(
            ef.member_tier({'Reviewed': 'unreviewed', 'PDB': '4C5C;'}),
            'structure')

    def test_protein_level_evidence_is_recognised(self):
        self.assertEqual(
            ef.member_tier({'Protein existence': 'Evidence at protein level'}),
            'protein_evidence')

    def test_everything_else_is_diverse(self):
        self.assertEqual(ef.member_tier({}), 'diverse')

    def test_a_blank_pdb_field_is_not_a_structure(self):
        self.assertEqual(ef.member_tier({'PDB': '   '}), 'diverse')


if __name__ == '__main__':
    unittest.main()


class FamilyLabelTests(unittest.TestCase):
    """A member is labelled by the signature that seeded it.

    Taking the SEED PROTEIN's label was wrong in a way that looked fine: a
    seed is labelled by the first name in its own signature list, so
    P0ADT5/P33222 - acid--amine ligases seeded from PF03738 - were presented
    as "CHAP domain", a peptidoglycan hydrolase.
    """

    def _expand(self, resolved):
        import amide_synthesis.families as fam
        original = fam.pfam_names
        fam.pfam_names = lambda accessions: resolved
        try:
            families = {'PF03738': {
                'pfam': 'PF03738', 'family': 'CHAP domain',
                'family_substrate_distance': 0.74, 'self_condensing': False,
                'n_seeds': 4, 'family_size': 900, 'seed_accessions': ['P1']}}

            def search(query, size=100, max_pages=5, fields=''):
                return [{'Entry': 'P0ADT5'}], False, False
            return ef.expand(families, search=search, threads=1,
                             enumerate_all=True)[0]
        finally:
            fam.pfam_names = original

    def test_the_signature_name_wins_over_the_seeds_label(self):
        frame = self._expand(
            {'PF03738': 'Glutathionylspermidine synthase preATP-grasp'})
        self.assertEqual(frame.family.iloc[0],
                         'Glutathionylspermidine synthase preATP-grasp')

    def test_the_seed_label_is_the_fallback_when_unresolved(self):
        """A lookup outage must not blank the column."""
        frame = self._expand({})
        self.assertEqual(frame.family.iloc[0], 'CHAP domain')

    def test_an_empty_resolved_name_falls_back(self):
        frame = self._expand({'PF03738': ''})
        self.assertEqual(frame.family.iloc[0], 'CHAP domain')
