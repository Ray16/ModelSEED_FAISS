"""Pagination, outage and cache behavior without public API requests."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from amide_synthesis.infrastructure import families, http, uniprot


class UniProtTests(unittest.TestCase):
    def test_next_link_is_not_assumed_to_be_first(self):
        headers = {'link': '<https://example.org/previous>; rel="prev", '
                           '<https://example.org/next>; rel="next"'}
        self.assertEqual(uniprot.next_page(headers), 'https://example.org/next')
        self.assertIsNone(uniprot.next_page({'Link': '<https://example.org/end>; rel="prev"'}))

    @patch.object(uniprot.time, 'sleep')
    def test_paginated_rows_sorted_and_budget_reports_saturation(self, sleep):
        pages = [('Entry\tProtein names\nZ\tLigase\n', {'Link': '<https://example.org/2>; rel="next"'}, False),
                 ('Entry\tProtein names\nA\tSynthase\n', {'Link': '<https://example.org/3>; rel="next"'}, False)]
        with patch.object(uniprot, 'get_text', side_effect=pages) as get:
            rows, failed, saturated = uniprot.search('ec:6.3.2.1', max_pages=2)
        self.assertEqual([r['Entry'] for r in rows], ['A', 'Z'])
        self.assertFalse(failed)
        self.assertTrue(saturated)
        self.assertEqual(get.call_args_list[1].args[0], 'https://example.org/2')
        sleep.assert_called_once()

    @patch.object(uniprot.time, 'sleep')
    def test_failed_later_page_retains_partial_rows(self, sleep):
        pages = [('Entry\nA\n', {'Link': '<https://example.org/2>; rel="next"'}, False),
                 ('', {}, True)]
        with patch.object(uniprot, 'get_text', side_effect=pages):
            rows, failed, saturated = uniprot.search('ec:6.3.2.1')
        self.assertEqual(rows, [{'Entry': 'A'}])
        self.assertTrue(failed)
        self.assertFalse(saturated)

    def test_multiline_tsv_fields_survive(self):
        with patch.object(uniprot, 'get_text', return_value=(
            'Entry\tProtein names\nA\t"line one\nline two"\n', {}, False
        )):
            rows, _, _ = uniprot.search('test')
        self.assertEqual(rows[0]['Protein names'], 'line one\nline two')


class FamilyCacheTests(unittest.TestCase):
    def test_missing_pfam_does_not_become_a_nan_family(self):
        self.assertEqual(families.split_accessions(float('nan')), [])
        self.assertEqual(families.split_accessions('nan; PF04183; '), ['PF04183'])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cache = Path(self.temp.name) / 'pfam.json'
        self.patch = patch.object(families, 'FAMILY_CACHE', self.cache)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    @patch.dict(os.environ, {'AMIDE_SYNTHESIS_OFFLINE': '1'})
    def test_offline_uses_cache_without_network_or_writes(self):
        self.cache.write_text(json.dumps({'PF04183': 'IucA/IucC'}))
        with patch.object(families, 'get_text') as get:
            names = families.pfam_names(['PF04183', 'PF99999'])
        get.assert_not_called()
        self.assertEqual(names, {'PF04183': 'IucA/IucC'})
        self.assertNotIn('PF99999', json.loads(self.cache.read_text()))

    @patch.dict(os.environ, {'AMIDE_SYNTHESIS_OFFLINE': '0'})
    @patch.object(families.time, 'sleep')
    def test_outage_does_not_poison_cache_and_next_run_retries(self, sleep):
        with patch.object(families, 'get_text', return_value=('', {}, True)):
            self.assertEqual(families.pfam_names(['PF04183']), {})
        self.assertFalse(self.cache.exists())
        response = json.dumps({'metadata': {'name': {'name': 'IucA/IucC'}}})
        with patch.object(families, 'get_text', return_value=(response, {}, False)):
            self.assertEqual(families.pfam_names(['PF04183']), {'PF04183': 'IucA/IucC'})
        self.assertEqual(json.loads(self.cache.read_text()), {'PF04183': 'IucA/IucC'})

    @patch.dict(os.environ, {'AMIDE_SYNTHESIS_OFFLINE': '1'})
    def test_http_offline_guard_never_opens_socket(self):
        with patch.object(http, 'urlopen') as open_url:
            self.assertEqual(http.get_text('https://example.org'), ('', {}, True))
        open_url.assert_not_called()
