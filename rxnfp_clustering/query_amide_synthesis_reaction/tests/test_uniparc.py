"""Controls for exact-sequence identity lookup.

CRC64 correctness is load-bearing: a wrong checksum returns zero hits, which
is indistinguishable from "not in UniParc". The first implementation was wrong
in exactly that way and reported 0 of 48, which looked like a finding.
"""
from __future__ import annotations

import json
import unittest

from amide_synthesis.infrastructure import uniparc


class Crc64Tests(unittest.TestCase):
    """Known-answer test against a checksum UniProt itself publishes.

    This is the guard that the original implementation would have failed. It
    returned plausible-looking hex and zero UniParc hits for every sequence,
    which reads as "none of these proteins are in UniParc" rather than as a
    bug. The sequence and checksum below are PuuA (P78061) verbatim from
    UniProt, so the test is offline but not self-referential.
    """

    PUUA = (
        'METNIVEVENFVQQSEERRGSAFTQEVKRYLERYPNTQYVDVLLTDLNGCFRGKRIPVSS'
        'LKKLEKGCYFPASVFAMDILGNVVEEAGLGQEMGEPDRTCVPVLGSLTPSAADPEFIGQM'
        'LLTMVDEDGAPFDVEPRNVLNRLWQQLRQRGLFPVVAVELEFYLLDRQRDAEGYLQPPCA'
        'PGTDDRNTQSQVYSVDNLNHFADVLNDIDELAQLQLIPADGAVAEASPGQFEINLYHTDN'
        'VLEACDDALALKRLVRLMAEKHKMHATFMAKPYEEHAGSGMHIHISMQNNRGENVLSDAE'
        'GEDSPLLKKMLAGMIDLMPSSMALLAPNVNSYRRFQPGMYVPTQASWGHNNRTVALRIPC'
        'GDRHNHRVEYRVAGADANPYLVMAAIFAGILHGLDNELPLQEEVEGNGLEQEGLPFPIRQ'
        'SDALGEFIENDHLRRYLGERFCHVYHACKNDELLQFERLITETEIEWMLKNA')
    PUUA_CRC64 = '9D2224C5A9FCC0A3'

    def test_matches_the_checksum_uniprot_publishes(self):
        self.assertEqual(uniparc.crc64(self.PUUA), self.PUUA_CRC64)

    def test_empty_sequence(self):
        self.assertEqual(uniparc.crc64(''), '0000000000000000')

    def test_checksum_is_sequence_dependent(self):
        self.assertNotEqual(uniparc.crc64('ACDEFGH'), uniparc.crc64('ACDEFGI'))

    def test_checksum_is_sixteen_hex_digits(self):
        value = uniparc.crc64('MKVLAT')
        self.assertEqual(len(value), 16)
        int(value, 16)


class SequenceCleaningTests(unittest.TestCase):
    def test_missing_sequence_is_rejected_at_the_boundary(self):
        """NaN is a float and not iterable; it must not reach the checksum."""
        for missing in (None, float('nan'), '', '   ', 'nan'):
            self.assertIsNone(uniparc.clean_sequence(missing))

    def test_whitespace_and_case_are_normalised(self):
        self.assertEqual(uniparc.clean_sequence(' mkv lat \n'), 'MKVLAT')

    def test_non_letters_are_dropped(self):
        self.assertEqual(uniparc.clean_sequence('MKV-LAT*'), 'MKVLAT')


class LookupTests(unittest.TestCase):
    SEARCH = json.dumps({'results': [{'uniParcId': 'UPI0009F7ED57'}]})
    ENTRY = json.dumps({'uniParcCrossReferences': [
        {'database': 'UniProtKB/TrEMBL', 'id': 'A0A1Y1SGY2', 'active': True,
         'organism': {'scientificName': 'Oceanococcus atlanticus'},
         'proteinName': 'Long-chain-fatty-acid--CoA ligase'},
        {'database': 'RefSeq', 'id': 'WP_083559276', 'active': True,
         'organism': {'scientificName': 'Oceanococcus atlanticus'}},
    ]})

    def two_step(self):
        def fetch(url, timeout=None):
            if '/search?' in url:
                return self.SEARCH, {}, False
            return self.ENTRY, {}, False
        return fetch

    def test_entry_is_fetched_for_cross_references(self):
        """The search endpoint returns a UPI but no cross-references.

        Stopping at the search yields a record with every field blank, which
        reads as "found but unannotated" rather than "not fetched".
        """
        record = uniparc.lookup('MKVLAT', fetch=self.two_step())
        self.assertEqual(record['uniparc_id'], 'UPI0009F7ED57')
        self.assertEqual(record['uniparc_uniprot'], 'A0A1Y1SGY2')
        self.assertEqual(record['uniparc_organism'], 'Oceanococcus atlanticus')
        self.assertIn('CoA ligase', record['uniparc_protein_name'])

    def test_inactive_cross_references_are_not_reported_as_uniprot(self):
        entry = json.dumps({'uniParcCrossReferences': [
            {'database': 'UniProtKB/TrEMBL', 'id': 'DEAD1', 'active': False,
             'organism': {'scientificName': 'X'}}]})

        def fetch(url, timeout=None):
            return (self.SEARCH if '/search?' in url else entry), {}, False
        record = uniparc.lookup('MKVLAT', fetch=fetch)
        self.assertEqual(record['uniparc_uniprot'], '')

    def test_absent_sequence_returns_none(self):
        def fetch(url, timeout=None):
            return json.dumps({'results': []}), {}, False
        self.assertIsNone(uniparc.lookup('MKVLAT', fetch=fetch))

    def test_http_failure_returns_none(self):
        self.assertIsNone(uniparc.lookup(
            'MKVLAT', fetch=lambda u, timeout=None: ('', {}, True)))

    def test_malformed_json_returns_none(self):
        self.assertIsNone(uniparc.lookup(
            'MKVLAT', fetch=lambda u, timeout=None: ('{oops', {}, False)))

    def test_lookup_many_skips_missing_sequences(self):
        calls = []

        def fetch(url, timeout=None):
            calls.append(url)
            return self.SEARCH if '/search?' in url else self.ENTRY, {}, False
        out = uniparc.lookup_many({'a': 'MKVLAT', 'b': float('nan'), 'c': ''},
                                  fetch=fetch, threads=1)
        self.assertEqual(set(out), {'a'})


if __name__ == '__main__':
    unittest.main(verbosity=2)
