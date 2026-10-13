"""Controls for the shortlist: bench filters, tier quotas, diversity caps.

The shortlist is the deliverable a human reads, so the failure modes here are
the ones that quietly produce a plausible-looking but wrong recommendation.
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from amide_synthesis import shortlist as sl
from amide_synthesis.master_table import EVIDENCE_WEIGHT


def candidate(accession, evidence='exact_reaction', priority=5.0,
              family='Fam', organism='Escherichia coli', length=500,
              clade=np.nan, pdb='', contradicted=False, sequence='MK' * 250):
    return {
        'uniprot': accession, 'gene': 'g', 'evidence_strength': evidence,
        'master_priority': priority, 'family': family, 'organism': organism,
        'length': length, 'clade_seed': clade, 'pdb': pdb,
        'sequence': sequence, 'self_condensation': False,
        'annotation_contradicts_chemistry': contradicted,
    }


class MissingValueGroupingTests(unittest.TestCase):
    """str(nan) is "nan", which is truthy - a real bug, shipped once.

    Every reaction-anchored row has no `clade_seed`. Treating the missing
    value as a group key collapsed all 215 eligible `exact_reaction` rows onto
    one clade and the per-clade cap of 3 admitted three of them.
    """

    def test_missing_values_are_not_a_group_key(self):
        for missing in (np.nan, None, 'nan', 'NaN', '<NA>', '', '  '):
            self.assertEqual(sl._text(missing), '')

    def test_rows_without_a_clade_are_not_capped_together(self):
        frame = pd.DataFrame([candidate(f'P{i}', family=f'Fam{i}',
                                        organism=f'Genus{i} sp.')
                              for i in range(20)])
        kept = sl.diversify(frame, limit=20)
        self.assertEqual(len(kept), 20,
                         'a missing clade must not act as a shared cap key')

    def test_a_real_shared_clade_is_still_capped(self):
        frame = pd.DataFrame([candidate(f'P{i}', family=f'Fam{i}',
                                        organism=f'Genus{i} sp.',
                                        clade='UniRef50_X')
                              for i in range(20)])
        self.assertEqual(len(sl.diversify(frame, limit=20, per_clade=3)), 3)


class BenchFilterTests(unittest.TestCase):
    def test_untestable_rows_are_excluded_with_a_reason(self):
        frame = pd.DataFrame([
            candidate('short', length=50),
            candidate('long', length=5000),
            candidate('noseq', sequence=''),
            candidate('bad', evidence='clade_homolog', contradicted=True),
            candidate('good'),
        ])
        out = sl.eligible(frame).set_index('uniprot').excluded_because
        self.assertEqual(out['short'], 'too_short')
        self.assertEqual(out['long'], 'too_long')
        self.assertEqual(out['noseq'], 'no_sequence')
        self.assertEqual(out['bad'], 'annotation_contradicts_chemistry')
        self.assertEqual(out['good'], '')

    def test_measured_activity_overrides_predicted_objections(self):
        frame = pd.DataFrame([
            candidate('assayed', evidence='measured_activity', length=5000),
        ])
        out = sl.eligible(frame).excluded_because.iloc[0]
        self.assertEqual(out, '', 'a measurement is not overruled by a heuristic')

    def test_measured_activity_without_a_sequence_is_still_excluded(self):
        frame = pd.DataFrame([
            candidate('assayed', evidence='measured_activity', sequence=''),
        ])
        self.assertEqual(sl.eligible(frame).excluded_because.iloc[0], 'no_sequence')


class TierQuotaTests(unittest.TestCase):
    """Diversity must not be allowed to outrank evidence."""

    def test_strong_tier_is_not_starved_by_a_broad_weak_tier(self):
        rows = [candidate(f'S{i}', 'exact_reaction', priority=9.0,
                          family=f'F{i % 4}', organism=f'G{i % 6} sp.')
                for i in range(60)]
        rows += [candidate(f'W{i}', 'clade_homolog', priority=8.9,
                           family=f'H{i}', organism=f'K{i} sp.',
                           clade=f'UniRef50_{i}')
                 for i in range(400)]
        out = sl.build(pd.DataFrame(rows), target=100)
        strong = (out.evidence_strength == 'exact_reaction').sum()
        self.assertGreater(strong, 10,
                           'the concentrated strong tier must get its quota')

    def test_assayed_proteins_are_always_kept(self):
        rows = [candidate(f'A{i}', 'measured_activity', priority=1.0,
                          family='Fam', organism='Escherichia coli')
                for i in range(20)]
        rows += [candidate(f'X{i}', 'clade_homolog', priority=99.0,
                           family=f'H{i}', organism=f'K{i} sp.')
                 for i in range(50)]
        out = sl.build(pd.DataFrame(rows), target=10)
        self.assertEqual((out.evidence_strength == 'measured_activity').sum(), 20)

    def test_output_is_deterministic(self):
        rows = [candidate(f'P{i}', family=f'F{i % 5}', organism=f'G{i % 7} sp.')
                for i in range(80)]
        a = sl.build(pd.DataFrame(rows), target=30)
        b = sl.build(pd.DataFrame(rows), target=30)
        self.assertEqual(a.uniprot.tolist(), b.uniprot.tolist())


class FastaTests(unittest.TestCase):
    def test_fasta_wraps_and_skips_empty_sequences(self):
        import tempfile
        from pathlib import Path
        frame = pd.DataFrame([candidate('P1', sequence='A' * 130),
                              candidate('P2', sequence='')])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'o.fasta'
            sl.write_fasta(frame, path)
            text = path.read_text()
        self.assertEqual(text.count('>'), 1)
        self.assertIn('A' * 60 + '\n', text)


if __name__ == '__main__':
    unittest.main(verbosity=2)


class UnallocatedTierTests(unittest.TestCase):
    """A tier missing from TIER_SHARE must fail loudly, not vanish.

    The protein-name route shipped with no share entry. Because the quotas
    already overfilled the target, the backfill never ran and all 3,469 of its
    candidates were silently unshortlistable while the summary looked normal.
    """

    def _pool(self, tier):
        return pd.DataFrame([{
            'uniprot': f'P{i:05d}', 'evidence_strength': tier,
            'master_priority': 1.0, 'length': 500,
            'organism': f'Org{i} sp.', 'family': f'Fam{i}',
            'sequence': 'M' * 500, 'protein_name': 'amide synthetase',
        } for i in range(5)])

    def test_an_unallocated_tier_raises(self):
        with self.assertRaises(ValueError) as caught:
            sl.build(self._pool('a_brand_new_route'), target=10)
        self.assertIn('a_brand_new_route', str(caught.exception))

    def test_an_allocated_tier_is_selected(self):
        out = sl.build(self._pool('enzyme_name_match'), target=10)
        self.assertGreater(len(out), 0)

    def test_every_evidence_weight_has_a_share(self):
        """Every route master_table can emit must be allocatable."""
        shares = {tier for tier, _ in sl.TIER_SHARE}
        for tier in EVIDENCE_WEIGHT:
            self.assertIn(tier, shares, f'{tier} has no shortlist share')


class BenchConstraintTests(unittest.TestCase):
    """Solubility constraints the module claimed but did not check."""

    def _row(self, keywords, evidence='exact_reaction'):
        return pd.DataFrame([{
            'uniprot': 'P00001', 'evidence_strength': evidence,
            'master_priority': 5.0, 'length': 500, 'sequence': 'M' * 500,
            'organism': 'E. coli', 'family': 'Fam', 'keywords': keywords,
        }])

    def test_a_transmembrane_protein_is_excluded(self):
        out = sl.eligible(self._row('Transmembrane;Membrane'))
        self.assertEqual(out.excluded_because.iloc[0], 'transmembrane')

    def test_a_signal_peptide_is_recorded_but_not_excluded(self):
        """A secreted soluble protein expresses fine."""
        out = sl.eligible(self._row('Signal;Cytoplasm'))
        self.assertEqual(out.excluded_because.iloc[0], '')
        self.assertTrue(bool(out.has_signal_peptide.iloc[0]))

    def test_a_peripheral_membrane_keyword_alone_does_not_exclude(self):
        out = sl.eligible(self._row('Membrane'))
        self.assertEqual(out.excluded_because.iloc[0], '')

    def test_measured_activity_overrides_the_transmembrane_filter(self):
        """A protein with a measured titre is a fact, not a prediction."""
        out = sl.eligible(self._row('Transmembrane', 'measured_activity'))
        self.assertEqual(out.excluded_because.iloc[0], '')

    def test_a_table_with_no_keywords_column_does_not_raise(self):
        frame = self._row('').drop(columns=['keywords'])
        out = sl.eligible(frame)
        self.assertEqual(out.excluded_because.iloc[0], '')


class RejectedFamilyTests(unittest.TestCase):
    """A family judged `reject` on its own seeds must stop supplying bench rows.

    This is the lever that makes the table more TARGETED rather than larger:
    the family verdict prunes homologues of families whose chemistry cannot
    reach the nylon bond.
    """

    def _row(self, rejected, evidence='family_homolog'):
        return pd.DataFrame([{
            'uniprot': 'P00001', 'evidence_strength': evidence,
            'master_priority': 5.0, 'length': 500, 'sequence': 'M' * 500,
            'organism': 'E. coli', 'family': 'Fam', 'keywords': '',
            'in_rejected_family': rejected,
        }])

    def test_a_member_of_a_rejected_family_is_excluded(self):
        out = sl.eligible(self._row(True))
        self.assertEqual(out.excluded_because.iloc[0], 'family_judged_reject')

    def test_a_member_of_an_accepted_family_is_kept(self):
        out = sl.eligible(self._row(False))
        self.assertEqual(out.excluded_because.iloc[0], '')

    def test_measured_activity_still_overrides(self):
        out = sl.eligible(self._row(True, 'measured_activity'))
        self.assertEqual(out.excluded_because.iloc[0], '')

    def test_a_table_without_the_column_is_unaffected(self):
        frame = self._row(False).drop(columns=['in_rejected_family'])
        self.assertEqual(sl.eligible(frame).excluded_because.iloc[0], '')
