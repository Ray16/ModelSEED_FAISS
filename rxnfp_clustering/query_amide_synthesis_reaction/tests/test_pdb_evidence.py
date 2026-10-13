"""Controls for experimental structural evidence.

Every field here is a fact about a deposited model, so the failure modes are
mis-parsing and silently treating a failed lookup as "no cofactor".
"""
from __future__ import annotations

import json
import unittest

import pandas as pd

from amide_synthesis import pdb_evidence as pe


def fake(payload, failed=False):
    def fetch(url, timeout=None):
        if failed:
            return '', {}, True
        return json.dumps(payload), {}, False
    return fetch


class EntryParsingTests(unittest.TestCase):
    def test_atp_analogue_is_recognised(self):
        facts = pe.entry_facts('1ABC', fetch=fake({
            'exptl': [{'method': 'X-RAY DIFFRACTION'}],
            'rcsb_entry_info': {'resolution_combined': [1.8],
                                'nonpolymer_bound_components': ['ANP', 'MG']}}))
        self.assertTrue(facts['has_atp_like'])
        self.assertEqual(facts['atp_like_ligand'], 'ANP')
        self.assertTrue(facts['has_metal'])
        self.assertEqual(facts['resolution_a'], 1.8)

    def test_unrelated_ligand_is_not_a_cofactor(self):
        facts = pe.entry_facts('2ABC', fetch=fake({
            'rcsb_entry_info': {'nonpolymer_bound_components': ['GOL', 'SO4']}}))
        self.assertFalse(facts['has_atp_like'])
        self.assertFalse(facts['has_metal'])

    def test_apo_structure_has_no_components(self):
        facts = pe.entry_facts('3ABC', fetch=fake({'rcsb_entry_info': {}}))
        self.assertFalse(facts['has_atp_like'])
        self.assertFalse(facts['lookup_failed'])

    def test_failed_lookup_is_flagged_not_read_as_apo(self):
        """Silence must not be mistaken for 'no cofactor bound'."""
        facts = pe.entry_facts('4ABC', fetch=fake({}, failed=True))
        self.assertTrue(facts['lookup_failed'])
        self.assertNotIn('has_atp_like', facts)

    def test_malformed_json_is_flagged(self):
        facts = pe.entry_facts('5ABC', fetch=lambda u, timeout=None: ('{oops', {}, False))
        self.assertTrue(facts['lookup_failed'])


class IdExtractionTests(unittest.TestCase):
    def test_semicolon_list_is_split_and_nan_ignored(self):
        table = pd.DataFrame([{'uniprot': 'P1', 'pdb': '1abc;2DEF;'},
                              {'uniprot': 'P2', 'pdb': 'nan'},
                              {'uniprot': 'P3', 'pdb': ''}])
        out = pe.pdb_ids_of(table)
        self.assertEqual(out, {'P1': ['1ABC', '2DEF']})


class PerProteinTests(unittest.TestCase):
    def test_cofactor_bound_structure_is_preferred_over_better_resolution(self):
        facts = pd.DataFrame([
            {'pdb_id': 'APO', 'resolution_a': 1.2, 'method': 'X', 'has_atp_like': False,
             'atp_like_ligand': '', 'has_metal': False, 'bound_components': ''},
            {'pdb_id': 'HOLO', 'resolution_a': 2.4, 'method': 'X', 'has_atp_like': True,
             'atp_like_ligand': 'ATP', 'has_metal': True, 'bound_components': 'ATP;MG'},
        ])
        out = pe.summarise_per_protein({'P1': ['APO', 'HOLO']}, facts)
        self.assertEqual(out.best_pdb.iloc[0], 'HOLO',
                         'an ATP position is worth more than 1.2 angstroms')
        self.assertTrue(out.pdb_has_atp_like.iloc[0])

    def test_best_resolution_wins_among_equivalent_structures(self):
        facts = pd.DataFrame([
            {'pdb_id': 'A', 'resolution_a': 3.0, 'method': 'X', 'has_atp_like': False,
             'atp_like_ligand': '', 'has_metal': False, 'bound_components': ''},
            {'pdb_id': 'B', 'resolution_a': 1.5, 'method': 'X', 'has_atp_like': False,
             'atp_like_ligand': '', 'has_metal': False, 'bound_components': ''},
        ])
        out = pe.summarise_per_protein({'P1': ['A', 'B']}, facts)
        self.assertEqual(out.best_pdb.iloc[0], 'B')

    def test_protein_with_no_retrieved_entry_is_omitted(self):
        out = pe.summarise_per_protein({'P1': ['MISSING']}, pd.DataFrame(
            columns=['pdb_id', 'resolution_a', 'method', 'has_atp_like',
                     'atp_like_ligand', 'has_metal', 'bound_components']))
        self.assertTrue(out.empty)


if __name__ == '__main__':
    unittest.main(verbosity=2)
