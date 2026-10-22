"""Controls for per-member discrimination within a family.

Every member of a family inherits one score, so these three components are
the only thing distinguishing 46k homologues from each other. The failure
modes that matter: a component that silently reads 0 for everything, and the
score leaking into the ranking it is forbidden from touching.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from amide_synthesis.candidate_selection import member_similarity as ms


class SignatureTests(unittest.TestCase):
    def test_both_databases_are_parsed(self):
        got = ms.signature_set('PF00501; IPR000873;')
        self.assertEqual(got, frozenset({'PF00501', 'IPR000873'}))

    def test_missing_values_are_empty_not_crashing(self):
        for value in (None, '', float('nan'), 'nan'):
            self.assertEqual(ms.signature_set(value), frozenset())


class DomainJaccardTests(unittest.TestCase):
    def test_identical_architecture_scores_one(self):
        both = frozenset({'PF00501', 'PF13193'})
        self.assertAlmostEqual(ms.domain_jaccard(both, [both]), 1.0)

    def test_sharing_one_domain_of_two_scores_a_third(self):
        """A passenger-domain match must not look like a full match."""
        member = frozenset({'PF00501'})
        seed = frozenset({'PF00501', 'PF13193'})
        self.assertAlmostEqual(ms.domain_jaccard(member, [seed]), 0.5)

    def test_the_best_seed_wins(self):
        member = frozenset({'PF00501', 'PF13193'})
        seeds = [frozenset({'PF99999'}), member]
        self.assertAlmostEqual(ms.domain_jaccard(member, seeds), 1.0)

    def test_no_overlap_scores_zero(self):
        self.assertEqual(
            ms.domain_jaccard(frozenset({'PF1'}), [frozenset({'PF2'})]), 0.0)

    def test_empty_inputs_score_zero_without_dividing_by_zero(self):
        self.assertEqual(ms.domain_jaccard(frozenset(), [frozenset()]), 0.0)
        self.assertEqual(ms.domain_jaccard(frozenset({'PF1'}), []), 0.0)


class LengthRatioTests(unittest.TestCase):
    def test_equal_lengths_score_one(self):
        self.assertAlmostEqual(ms.length_ratio(500, [500]), 1.0)

    def test_a_fragment_scores_low(self):
        """200 residues against a 900-residue synthetase is a fragment."""
        self.assertAlmostEqual(ms.length_ratio(200, [900]), 200 / 900)

    def test_it_is_symmetric(self):
        self.assertAlmostEqual(ms.length_ratio(200, [900]),
                               ms.length_ratio(900, [200]))

    def test_the_closest_seed_wins(self):
        self.assertAlmostEqual(ms.length_ratio(500, [900, 480]), 480 / 500)

    def test_bad_values_score_zero(self):
        for length in (None, 0, -5, 'abc', float('nan')):
            self.assertEqual(ms.length_ratio(length, [500]), 0.0)
        self.assertEqual(ms.length_ratio(500, []), 0.0)


class KmerTests(unittest.TestCase):
    def test_an_identical_sequence_scores_one(self):
        sequence = 'MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ'
        got = ms.best_kmer_cosine([sequence], [sequence])
        self.assertAlmostEqual(got[0], 1.0, places=5)

    def test_an_unrelated_sequence_scores_low(self):
        got = ms.best_kmer_cosine(['WWWWWWWWWWWWWWWWWWWW'],
                                  ['MKTAYIAKQRQISFVKSHFS'])
        self.assertLess(got[0], 0.2)

    def test_the_best_seed_wins(self):
        sequence = 'MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ'
        got = ms.best_kmer_cosine([sequence],
                                  ['WWWWWWWWWWWWWW', sequence])
        self.assertAlmostEqual(got[0], 1.0, places=5)

    def test_no_seeds_scores_zero(self):
        self.assertEqual(list(ms.best_kmer_cosine(['MKTA'], [])), [0.0])

    def test_chunking_does_not_change_the_answer(self):
        sequences = ['MKTAYIAKQRQISFVKSHFSRQ' + 'A' * i for i in range(10)]
        whole = ms.best_kmer_cosine(sequences, sequences, chunk=100)
        chunked = ms.best_kmer_cosine(sequences, sequences, chunk=3)
        for a, b in zip(whole, chunked):
            self.assertAlmostEqual(a, b, places=6)


class ScoreTests(unittest.TestCase):
    def _table(self):
        return pd.DataFrame([
            {'uniprot': 'P1', 'seed_pfam': 'PF04183', 'pfam': 'PF04183',
             'interpro': 'IPR007310', 'length': 600, 'sequence': 'MKTAYIAK' * 8},
            {'uniprot': 'P2', 'seed_pfam': 'PF04183', 'pfam': 'PF04183',
             'interpro': '', 'length': 180, 'sequence': 'WWWW' * 8},
        ])

    def _profiles(self):
        return {'PF04183': {
            'signatures': [frozenset({'PF04183', 'IPR007310'})],
            'lengths': [600], 'sequences': ['MKTAYIAK' * 8]}}

    def test_the_matching_member_outscores_the_fragment(self):
        out = ms.score(self._table(), self._profiles()).set_index('uniprot')
        self.assertGreater(out.loc['P1', 'member_similarity'],
                           out.loc['P2', 'member_similarity'])

    def test_all_three_components_are_reported(self):
        out = ms.score(self._table(), self._profiles())
        for component in ms.COMPONENTS:
            self.assertIn(component, out.columns)

    def test_a_family_with_no_seed_profile_scores_zero_not_nan(self):
        table = self._table()
        table['seed_pfam'] = 'PF_UNKNOWN'
        out = ms.score(table, self._profiles())
        self.assertTrue((out.kmer_cosine == 0).all())
        self.assertFalse(out.member_similarity.isna().any())

    def test_a_missing_seed_column_does_not_raise(self):
        table = self._table().drop(columns=['seed_pfam'])
        out = ms.score(table, self._profiles())
        self.assertIn('member_similarity', out.columns)


class RankingIsolationTests(unittest.TestCase):
    """It may order tied rows. It may not become a score."""

    def test_similarity_is_absent_from_the_priority_formula(self):
        from amide_synthesis.candidate_selection import master_table
        source = Path(master_table.__file__).read_text().lower()
        for word in ('member_similarity', 'kmer_cosine', 'domain_jaccard'):
            self.assertNotIn(word, source)

    def test_shortlist_uses_it_only_after_master_priority(self):
        from amide_synthesis.candidate_selection import shortlist
        source = Path(shortlist.__file__).read_text()
        self.assertIn("['master_priority', 'member_similarity', 'uniprot']",
                      source)


if __name__ == '__main__':
    unittest.main()


class SeedProfileWiringTests(unittest.TestCase):
    """The wiring bug the function-level tests could not catch.

    Sequences live in enzymes.fasta, not in the candidate frame, so reading
    them off the frame gave every seed an empty sequence and kmer_cosine
    returned 0.0 for all 46,254 homologues while the stage reported success.
    """

    def _candidates(self):
        return pd.DataFrame([{
            'uniprot': 'P1', 'pfam': 'PF04183', 'interpro': 'IPR007310',
            'length': 600, 'substrate_distance': 0.83, 'family': 'NIS',
            'self_condensation': False,
        }])

    def test_sequences_are_taken_from_the_supplied_mapping(self):
        profiles = ms.seed_profiles(self._candidates(),
                                    {'P1': 'MKTAYIAK' * 8})
        self.assertEqual(profiles['PF04183']['sequences'], ['MKTAYIAK' * 8])

    def test_no_mapping_leaves_sequences_empty_rather_than_guessing(self):
        profiles = ms.seed_profiles(self._candidates(), {})
        self.assertEqual(profiles['PF04183']['sequences'], [])

    def test_signatures_and_lengths_still_populate_without_sequences(self):
        """The other two components must survive a missing FASTA."""
        profiles = ms.seed_profiles(self._candidates(), {})
        self.assertTrue(profiles['PF04183']['signatures'])
        self.assertEqual(profiles['PF04183']['lengths'], [600])

    def test_an_empty_profile_makes_the_cosine_zero_detectably(self):
        table = pd.DataFrame([{
            'uniprot': 'Q1', 'seed_pfam': 'PF04183', 'pfam': 'PF04183',
            'interpro': 'IPR007310', 'length': 600, 'sequence': 'MKTA' * 16}])
        out = ms.score(table, ms.seed_profiles(self._candidates(), {}))
        self.assertEqual(out.kmer_cosine.iloc[0], 0.0)
