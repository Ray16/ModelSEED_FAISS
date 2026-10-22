"""Controls for AlphaFold model confidence.

The failure modes that matter here are the quiet ones: a network error
recorded as a low score, and model confidence leaking into the ranking.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from amide_synthesis import model_quality as mq   # noqa: E402

PAYLOAD = [{
    'entryId': 'AF-A0KKW8-F1',
    'latestVersion': 6,
    'globalMetricValue': 93.88,
    'fractionPlddtVeryHigh': 0.81,
    'fractionPlddtConfident': 0.12,
    'fractionPlddtLow': 0.05,
    'fractionPlddtVeryLow': 0.02,
}]


def ok(payload=PAYLOAD):
    return lambda url, timeout=60: (json.dumps(payload), {}, False)


def dead():
    return lambda url, timeout=60: ('', {}, True)


def empty():
    return lambda url, timeout=60: ('[]', {}, False)


def malformed():
    return lambda url, timeout=60: ('not json', {}, False)


class FetchTests(unittest.TestCase):
    def test_confidence_is_parsed(self):
        row, failed = mq.fetch_one('A0KKW8', fetch=ok())
        self.assertFalse(failed)
        self.assertAlmostEqual(row['plddt_mean'], 93.88)
        self.assertAlmostEqual(row['fraction_plddt_very_high'], 0.81)
        self.assertEqual(row['alphafold_entry'], 'AF-A0KKW8-F1')

    def test_a_network_failure_is_not_a_low_score(self):
        """The distinction that matters: dead API vs genuinely no model."""
        row, failed = mq.fetch_one('A0KKW8', fetch=dead())
        self.assertTrue(failed)
        self.assertIsNone(row)

    def test_no_model_is_not_a_failure(self):
        row, failed = mq.fetch_one('A0KKW8', fetch=empty())
        self.assertFalse(failed)
        self.assertIsNone(row)

    def test_malformed_json_returns_none(self):
        row, failed = mq.fetch_one('A0KKW8', fetch=malformed())
        self.assertFalse(failed)
        self.assertIsNone(row)

    def test_a_missing_metric_is_none_not_zero(self):
        """A zero would read as a terrible model rather than an absent one."""
        row, _ = mq.fetch_one('A0KKW8', fetch=ok([{'entryId': 'AF-X-F1'}]))
        self.assertIsNone(row['plddt_mean'])

    def test_failures_are_counted_separately_from_results(self):
        frame, failures = mq.fetch_many(['A', 'B'], fetch=dead(), threads=1)
        self.assertEqual(len(frame), 0)
        self.assertEqual(failures, 2)


class ClassifyTests(unittest.TestCase):
    def test_bands(self):
        frame = mq.classify(pd.DataFrame({'plddt_mean': [95.0, 72.0, 55.0, 30.0]}))
        self.assertEqual(list(frame.model_confident), [True, True, False, False])
        self.assertEqual(list(frame.model_usable_for_cofold),
                         [True, True, True, False])

    def test_a_missing_plddt_is_not_called_usable(self):
        frame = mq.classify(pd.DataFrame({'plddt_mean': [None]}))
        self.assertFalse(bool(frame.model_usable_for_cofold.iloc[0]))

    def test_an_absent_column_does_not_raise(self):
        frame = mq.classify(pd.DataFrame({'uniprot': ['A0KKW8']}))
        self.assertIn('model_usable_for_cofold', frame.columns)


class RankingIsolationTests(unittest.TestCase):
    """Model confidence must not become a score gate - the sixth one."""

    def test_plddt_is_not_in_the_priority_formula(self):
        from amide_synthesis import master_table
        source = Path(master_table.__file__).read_text()
        self.assertNotIn('plddt', source.lower())

    def test_plddt_is_not_used_to_select_the_shortlist(self):
        from amide_synthesis import shortlist
        source = Path(shortlist.__file__).read_text()
        self.assertNotIn('plddt', source.lower())

class AccessionPatternTests(unittest.TestCase):
    """Both UniProt accession lengths must be queried.

    A first version matched only the 6-character form and silently queried 75
    of 301 shortlisted proteins while reporting zero failures.
    """

    def _match(self, value):
        import re
        return bool(re.match(mq.ACCESSION, value))

    def test_six_character_accessions(self):
        for accession in ('P07862', 'Q9L069', 'A0KKW8', 'O52547', 'B1YSS6'):
            self.assertTrue(self._match(accession), accession)

    def test_ten_character_accessions(self):
        for accession in ('A0A0E1RWL5', 'A0A011NE06', 'A0A2D9ZTX9'):
            self.assertTrue(self._match(accession), accession)

    def test_ncbi_and_suffixed_identifiers_are_skipped(self):
        for other in ('OPY85079.1', 'TEX51524.1', 'A0A2D9ZTX9-F1',
                      'MBB4155089.1'):
            self.assertFalse(self._match(other), other)

if __name__ == '__main__':
    unittest.main()
