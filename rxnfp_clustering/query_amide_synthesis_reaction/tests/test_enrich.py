"""Controls for the annotation stage.

The dangerous failure here is silent overwriting: a weaker lookup replacing a
stronger route's curated record, or a later stage discarding the enrichment.
"""
from __future__ import annotations

import unittest
from unittest.mock import Mock

import pandas as pd

from amide_synthesis import enrich


class BlankDetectionTests(unittest.TestCase):
    def test_nan_string_counts_as_blank(self):
        series = pd.Series(['', '  ', 'nan', 'NaN', None, 'real'])
        self.assertEqual(enrich.blank(series).tolist(),
                         [True, True, True, True, True, False])


class TargetSelectionTests(unittest.TestCase):
    def test_only_unannotated_uniprot_accessions_are_fetched(self):
        table = pd.DataFrame([
            {'uniprot': 'P0DXD6', 'catalytic_activity': 'CATALYTIC ACTIVITY: x'},
            {'uniprot': 'Q81RQ8', 'catalytic_activity': ''},
            {'uniprot': 'A0A090S1A3', 'catalytic_activity': 'nan'},
            {'uniprot': 'WP_083559276.1', 'catalytic_activity': ''},
        ])
        pending = enrich.needs_enrichment(table)
        self.assertIn('Q81RQ8', pending)
        self.assertIn('A0A090S1A3', pending)
        self.assertNotIn('P0DXD6', pending, 'already annotated')
        self.assertNotIn('WP_083559276.1', pending, 'not a UniProt accession')


class NeverOverwriteTests(unittest.TestCase):
    def test_existing_values_are_preserved(self):
        table = pd.DataFrame([{'uniprot': 'P1', 'organism': 'Real organism',
                               'ec': ''}])
        out, filled = enrich.apply_records(
            table, [{'Entry': 'P1', 'Organism': 'Wrong organism',
                     'EC number': '6.3.2.1'}])
        self.assertEqual(out.organism.iloc[0], 'Real organism',
                         'a curated value must win over a later lookup')
        self.assertEqual(out.ec.iloc[0], '6.3.2.1', 'a gap is filled')
        self.assertEqual(filled, 1)

    def test_missing_column_is_created(self):
        table = pd.DataFrame([{'uniprot': 'P1'}])
        out, _ = enrich.apply_records(table, [{'Entry': 'P1', 'Pfam': 'PF04183'}])
        self.assertEqual(out.pfam.iloc[0], 'PF04183')

    def test_no_records_is_a_no_op(self):
        table = pd.DataFrame([{'uniprot': 'P1', 'ec': ''}])
        out, filled = enrich.apply_records(table, [])
        self.assertEqual(filled, 0)
        self.assertEqual(len(out), 1)

    def test_records_without_an_entry_column_are_ignored(self):
        table = pd.DataFrame([{'uniprot': 'P1', 'ec': ''}])
        out, filled = enrich.apply_records(table, [{'Organism': 'x'}])
        self.assertEqual(filled, 0)


class BatchingTests(unittest.TestCase):
    def test_accessions_are_batched_into_or_queries(self):
        client = Mock(return_value=([], False, False))
        enrich.fetch([f'P{i}' for i in range(45)], search=client,
                     batch=20, threads=1)
        self.assertEqual(client.call_count, 3)
        self.assertIn(' OR ', client.call_args_list[0].args[0])

    def test_empty_input_issues_no_queries(self):
        client = Mock(return_value=([], False, False))
        rows, failures = enrich.fetch([], search=client, threads=1)
        self.assertEqual(client.call_count, 0)
        self.assertEqual((rows, failures), ([], 0))

    def test_failed_batches_are_counted_not_swallowed(self):
        client = Mock(return_value=([], True, False))
        _, failures = enrich.fetch(['P1'], search=client, threads=1)
        self.assertEqual(failures, 1)


class NcbiResolutionTests(unittest.TestCase):
    def test_partial_resolution_is_reported_not_faked(self):
        def client(query, **kw):
            if 'WP_1' in query:
                return [{'Entry': 'A0A1', 'Organism': 'Oceanococcus'}], False, False
            return [], False, False

        mapped = enrich.resolve_ncbi(['WP_1.1', 'KUO_2.1'],
                                     search=Mock(side_effect=client), threads=1)
        self.assertEqual(set(mapped), {'WP_1.1'})


if __name__ == '__main__':
    unittest.main(verbosity=2)
