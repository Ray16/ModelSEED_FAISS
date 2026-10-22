"""Controls for the SABIO-RK kinetics route.

The route exists because SABIO-RK was recorded as dead on the evidence of a
1,182-byte HTML reply. It is a single-page-app shell served for every
unmatched path, which is why four different ECs returned byte-identical
bodies. The controls here are about the parsing, which is where a silent
wrong number would come from.
"""
from __future__ import annotations

import unittest

import pandas as pd

from amide_synthesis.candidate_selection import kinetics as kin


def _record(**over):
    base = {
        'id': 1,
        'publication': {'pubmed_id': '12345'},
        'enzyme_description': {
            'ec_number': '6.3.2.17',
            'enzyme_name': 'tetrahydrofolate synthase',
            'wildtype': 'Wildtype',
            'proteins': [{'uniprot_id': 'P08192'}, {'uniprot_id': None}],
        },
        'kineticlaw': {'parameter': [
            {'parameter_type': {'name': 'Km'}, 'start_value': 160.0,
             'unit': {'name': 'µM', 'n_name': 'M'}, 'n_start_value': 1.6e-4,
             'species': {'species_key': '1 | L-Glutamate | Substrate'}},
            {'parameter_type': {'name': 'concentration'},
             'start_value': 5.0, 'unit': {'name': 'mM', 'n_name': 'M'},
             'n_start_value': 0.005, 'species': {'species_key': '1 | ATP | S'}},
        ]},
    }
    base.update(over)
    return base


class TestParameterExtraction(unittest.TestCase):

    def test_only_wanted_parameters_are_kept(self):
        """`concentration` is an experimental condition, not a constant."""
        rows = kin.parameters(_record())
        self.assertEqual([r['parameter'] for r in rows], ['Km'])

    def test_si_value_and_unit_are_carried(self):
        row = kin.parameters(_record())[0]
        self.assertAlmostEqual(row['value_si'], 1.6e-4)
        self.assertEqual(row['unit_si'], 'M')
        self.assertEqual(row['value'], 160.0)
        self.assertEqual(row['unit'], 'µM')

    def test_species_is_taken_from_the_middle_field(self):
        self.assertEqual(kin.parameters(_record())[0]['species'],
                         'L-Glutamate')

    def test_null_uniprot_ids_are_dropped_not_stringified(self):
        self.assertEqual(kin.parameters(_record())[0]['uniprot_ids'],
                         'P08192')

    def test_missing_sections_do_not_raise(self):
        self.assertEqual(kin.parameters({}), [])
        self.assertEqual(kin.parameters({'kineticlaw': None}), [])


class TestCandidateECs(unittest.TestCase):

    def test_only_complete_numeric_ecs(self):
        table = pd.DataFrame({'ec': ['6.3.2.17', '2.3.1.-', '6.3.-.-',
                                     '6.3.2.M5', '1.1.1.1;2.7.7.9', None]})
        self.assertEqual(kin.candidate_ecs(table),
                         ['1.1.1.1', '2.7.7.9', '6.3.2.17'])


class TestSummarise(unittest.TestCase):

    def test_median_is_over_si_values(self):
        frame = pd.DataFrame({
            'ec': ['1.1.1.1'] * 3, 'parameter': ['Km'] * 3,
            'value_si': [1e-4, 2e-4, 3e-4], 'unit_si': ['M'] * 3})
        out = kin.summarise(frame)
        self.assertEqual(int(out.n.iloc[0]), 3)
        self.assertAlmostEqual(float(out.median_si.iloc[0]), 2e-4)

    def test_unparseable_values_are_excluded_not_zeroed(self):
        frame = pd.DataFrame({
            'ec': ['1.1.1.1'] * 3, 'parameter': ['Km'] * 3,
            'value_si': [1e-4, None, 'n/a'], 'unit_si': ['M'] * 3})
        out = kin.summarise(frame)
        self.assertEqual(int(out.n.iloc[0]), 1)

    def test_empty_input_gives_empty_frame(self):
        self.assertTrue(kin.summarise(pd.DataFrame()).empty)


if __name__ == '__main__':
    unittest.main()
