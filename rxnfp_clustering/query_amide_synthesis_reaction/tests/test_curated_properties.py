"""Controls for curated UniProt properties, including KM parsing.

The quiet failures here are unit handling (a KM in uM read as mM is a
thousand-fold error in the wrong direction) and treating an absent field as a
measured zero.
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from amide_synthesis import curated_properties as cp   # noqa: E402

REAL = ('BIOPHYSICOCHEMICAL PROPERTIES:  Kinetic parameters: KM=0.38 mM for '
        'glyoxal (at pH 7.4 and 37 degrees Celsius) '
        '{ECO:0000269|PubMed:26678554}; KM=0.06 mM for methylglyoxal')


class KmParsingTests(unittest.TestCase):
    def test_the_tightest_km_is_taken(self):
        self.assertAlmostEqual(cp.smallest_km_mm(REAL), 0.06)

    def test_micromolar_is_converted_not_compared_raw(self):
        """40 uM is 0.04 mM; read raw it would look like the weakest binder."""
        self.assertAlmostEqual(cp.smallest_km_mm('KM=40 uM for L-cysteine'),
                               0.04)

    def test_nanomolar_is_converted(self):
        self.assertAlmostEqual(cp.smallest_km_mm('KM=500 nM for X'), 0.0005)

    def test_units_are_compared_after_conversion(self):
        """The smaller NUMBER is not the tighter KM across mixed units."""
        self.assertAlmostEqual(
            cp.smallest_km_mm('KM=2 uM for A; KM=0.5 mM for B'), 0.002)

    def test_absent_kinetics_is_none_not_zero(self):
        for value in ('', None, 'no kinetic data', float('nan')):
            self.assertIsNone(cp.smallest_km_mm(value))

    def test_a_malformed_number_is_skipped_not_fatal(self):
        self.assertIsNone(cp.smallest_km_mm('KM=abc mM for X'))

    def test_case_insensitive_units(self):
        self.assertAlmostEqual(cp.smallest_km_mm('Km=10 UM for X'), 0.01)


class TidyTests(unittest.TestCase):
    def _rows(self):
        return [{'Entry': 'P45470', 'Kinetics': REAL,
                 'pH dependence': 'Optimum pH is 7.5',
                 'Temperature dependence': '', 'Activity regulation': '',
                 'Subcellular location [CC]': 'SUBCELLULAR LOCATION: Cytoplasm.',
                 'Domain [FT]': 'DOMAIN 101..303; /note="ATP-grasp"',
                 'Sequence similarities': 'SIMILARITY: Belongs to the X family.'},
                {'Entry': 'P07862', 'Kinetics': '', 'pH dependence': '',
                 'Temperature dependence': '', 'Activity regulation': '',
                 'Subcellular location [CC]': '', 'Domain [FT]': '',
                 'Sequence similarities': ''}]

    def test_columns_are_renamed_to_ours(self):
        frame = cp.tidy(self._rows())
        for column in ('uniprot', 'kinetics', 'ph_optimum',
                       'subcellular_location', 'domain_architecture',
                       'sequence_similarities', 'km_min_mm'):
            self.assertIn(column, frame.columns)

    def test_measured_kinetics_flag_distinguishes_present_from_absent(self):
        frame = cp.tidy(self._rows()).set_index('uniprot')
        self.assertTrue(bool(frame.loc['P45470', 'has_measured_kinetics']))
        self.assertFalse(bool(frame.loc['P07862', 'has_measured_kinetics']))

    def test_km_is_parsed_only_where_present(self):
        frame = cp.tidy(self._rows()).set_index('uniprot')
        self.assertAlmostEqual(frame.loc['P45470', 'km_min_mm'], 0.06)
        self.assertTrue(pd.isna(frame.loc['P07862', 'km_min_mm']))

    def test_empty_input_gives_an_empty_frame_with_columns(self):
        frame = cp.tidy([])
        self.assertEqual(len(frame), 0)
        self.assertIn('uniprot', frame.columns)

    def test_a_missing_field_in_the_response_does_not_raise(self):
        frame = cp.tidy([{'Entry': 'P1', 'Kinetics': REAL}])
        self.assertIn('subcellular_location', frame.columns)

    def test_duplicate_accessions_collapse(self):
        rows = self._rows() + self._rows()
        self.assertEqual(len(cp.tidy(rows)), 2)


class FieldNameTests(unittest.TestCase):
    def test_the_documented_field_name_is_not_used(self):
        """`cc_biophysicochemical_properties` returns HTTP 400; `kinetics` works."""
        self.assertIn('kinetics', cp.FIELDS)
        self.assertNotIn('cc_biophysicochemical_properties', cp.FIELDS)

    def test_every_requested_field_has_a_returned_mapping(self):
        self.assertEqual(sorted(cp.WANTED.values()),
                         sorted(cp.RETURNED.values()))


class AccessionTests(unittest.TestCase):
    def test_both_uniprot_lengths_match(self):
        for accession in ('P07862', 'A0A0E1RWL5'):
            self.assertTrue(re.match(cp.ACCESSION, accession), accession)

    def test_panel_ncbi_accessions_are_skipped(self):
        for other in ('OPY85079.1', 'A0A2D9ZTX9-F1'):
            self.assertFalse(re.match(cp.ACCESSION, other), other)


class RankingIsolationTests(unittest.TestCase):
    def test_kinetics_does_not_enter_the_priority(self):
        from amide_synthesis import master_table
        source = Path(master_table.__file__).read_text().lower()
        for word in ('km_min_mm', 'kinetics'):
            self.assertNotIn(word, source)


if __name__ == '__main__':
    unittest.main()


class ProseContradictionTests(unittest.TestCase):
    """The family-prose cross-check, and the rescue that keeps E2 subunits.

    Reading the prose alone flagged 148 rows of the 2-oxoacid dehydrogenase
    family, 59 of which are genuine succinyltransferases - and succinyl is a
    C4 diacid donor, chemistry this project wants.
    """

    def _flag(self, prose, name):
        return bool(cp.prose_contradicts(pd.Series([prose]),
                                         pd.Series([name])).iloc[0])

    def test_a_phosphatase_family_is_flagged(self):
        self.assertTrue(self._flag(
            'SIMILARITY: Belongs to the histidine acid phosphatase family.',
            'Inositol hexakisphosphate kinase'))

    def test_a_peptidase_family_is_flagged(self):
        self.assertTrue(self._flag(
            'SIMILARITY: Belongs to the peptidase C56 family.', 'HchA'))

    def test_the_e1_decarboxylase_component_is_flagged(self):
        self.assertTrue(self._flag(
            'SIMILARITY: Belongs to the alpha-ketoglutarate dehydrogenase family.',
            '2-oxoglutarate dehydrogenase E1 component'))

    def test_the_e2_succinyltransferase_component_is_kept(self):
        """The prose names the complex; the subunit is an acyl transferase."""
        self.assertFalse(self._flag(
            'SIMILARITY: Belongs to the 2-oxoacid dehydrogenase family.',
            'Dihydrolipoyllysine-residue succinyltransferase component of '
            '2-oxoglutarate dehydrogenase complex'))

    def test_a_ligase_family_is_not_flagged(self):
        self.assertFalse(self._flag(
            'SIMILARITY: Belongs to the D-alanine--D-alanine ligase family.',
            'D-alanine--D-alanine ligase'))

    def test_absent_prose_is_not_a_contradiction(self):
        for prose in ('', None):
            self.assertFalse(self._flag(prose, 'Amide synthetase'))

    def test_a_qualifying_word_in_the_prose_also_rescues(self):
        self.assertFalse(self._flag(
            'SIMILARITY: Belongs to the glutamine synthetase family. '
            'Related to the amidohydrolase fold.', ''))
