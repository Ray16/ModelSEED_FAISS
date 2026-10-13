"""Controls for crystallographic ligand evidence.

The risks: a metal or an unparseable component reading as a substrate, and a
cofactor being mistaken for one.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from amide_synthesis import ligand_evidence as le   # noqa: E402


def response(name, smiles, kind='SMILES_CANONICAL'):
    payload = {'chem_comp': {'name': name},
               'pdbx_chem_comp_descriptor': (
                   [{'type': kind, 'descriptor': smiles}] if smiles else [])}
    return lambda url, timeout=40: (json.dumps(payload), {}, False)


class FetchTests(unittest.TestCase):
    def test_name_and_smiles_are_parsed(self):
        got = le.fetch_component('API', response('DIAMINOPIMELIC ACID',
                                                 'OC(=O)C(N)CCCC(N)C(=O)O'))
        self.assertEqual(got['ligand_name'], 'DIAMINOPIMELIC ACID')
        self.assertIn('N', got['smiles'])

    def test_a_network_failure_returns_none(self):
        got = le.fetch_component('API', lambda url, timeout=40: ('', {}, True))
        self.assertIsNone(got)

    def test_malformed_json_returns_none(self):
        got = le.fetch_component('API', lambda u, timeout=40: ('{', {}, False))
        self.assertIsNone(got)


class ClassifyTests(unittest.TestCase):
    def _one(self, name, smiles):
        return le.classify([{'component': 'X', 'ligand_name': name,
                             'smiles': smiles}]).iloc[0]

    def test_diaminopimelate_is_bifunctional_both_ways(self):
        """meso-DAP: a C7 diacid AND diamine - the motif nylon needs."""
        row = self._one('DAP', 'OC(=O)C(N)CCCC(N)C(=O)O')
        self.assertGreaterEqual(row.ligand_acids, 2)
        self.assertGreaterEqual(row.ligand_amines, 2)
        self.assertTrue(row.ligand_bifunctional)

    def test_a_monoacid_monoamine_is_not_bifunctional(self):
        row = self._one('ALA', 'C[C@@H](N)C(=O)O')
        self.assertFalse(row.ligand_bifunctional)

    def test_a_dicarboxylate_counts(self):
        row = self._one('MLI', 'OC(=O)CC(=O)O')
        self.assertTrue(row.ligand_bifunctional)

    def test_a_metal_is_not_a_substrate(self):
        row = self._one('MG', '[Mg+2]')
        self.assertFalse(row.ligand_bifunctional)
        self.assertEqual(row.ligand_acids, 0)

    def test_an_unparseable_component_does_not_raise(self):
        """Some PDB components (metal-coordinated cobalamins) are not SMILES."""
        row = self._one('B12', '[Co++]|1|2|3|N4C5=CC6=N|1C')
        self.assertFalse(row.ligand_bifunctional)

    def test_a_missing_smiles_does_not_raise(self):
        row = self._one('XXX', None)
        self.assertFalse(row.ligand_bifunctional)

    def test_atp_is_not_counted_as_a_bifunctional_substrate(self):
        """The cofactor is already handled by pdb_has_atp_like."""
        atp = ('Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)'
               '[C@@H](O)[C@H]1O')
        self.assertFalse(self._one('ATP', atp).ligand_bifunctional)


class AttachTests(unittest.TestCase):
    def _chemistry(self):
        return pd.DataFrame([
            {'component': 'API', 'ligand_name': 'DAP', 'smiles': 'x',
             'ligand_acids': 2, 'ligand_amines': 2, 'ligand_heavy_atoms': 13,
             'ligand_bifunctional': True},
            {'component': 'MG', 'ligand_name': 'MAGNESIUM', 'smiles': 'x',
             'ligand_acids': 0, 'ligand_amines': 0, 'ligand_heavy_atoms': 1,
             'ligand_bifunctional': False},
            {'component': 'GSH', 'ligand_name': 'GLUTATHIONE', 'smiles': 'x',
             'ligand_acids': 2, 'ligand_amines': 3, 'ligand_heavy_atoms': 20,
             'ligand_bifunctional': True},
        ])

    def _evidence(self):
        return pd.DataFrame([
            {'uniprot': 'P22188', 'pdb_bound_components': 'API;MG;UAG'},
            {'uniprot': 'P00001', 'pdb_bound_components': 'MG'},
            {'uniprot': 'P00002', 'pdb_bound_components': None},
            {'uniprot': 'P0AES0', 'pdb_bound_components': 'API;GSH'},
        ])

    def test_a_bound_bifunctional_ligand_is_recorded(self):
        out = le.attach(self._evidence(),
                        self._chemistry()).set_index('uniprot')
        self.assertEqual(out.loc['P22188', 'bound_bifunctional_ligand'], 'API')
        self.assertEqual(out.loc['P22188', 'bound_ligand_acids'], 2)

    def test_a_metal_only_structure_records_nothing(self):
        out = le.attach(self._evidence(),
                        self._chemistry()).set_index('uniprot')
        self.assertIsNone(out.loc['P00001', 'bound_bifunctional_ligand'])

    def test_no_bound_components_is_not_fatal(self):
        out = le.attach(self._evidence(),
                        self._chemistry()).set_index('uniprot')
        self.assertIsNone(out.loc['P00002', 'bound_bifunctional_ligand'])

    def test_the_most_aminated_ligand_is_reported(self):
        out = le.attach(self._evidence(),
                        self._chemistry()).set_index('uniprot')
        self.assertEqual(out.loc['P0AES0', 'bound_ligand_amines'], 3)
        self.assertIn('GSH', out.loc['P0AES0', 'bound_bifunctional_ligand'])

    def test_empty_chemistry_does_not_raise(self):
        out = le.attach(self._evidence(), pd.DataFrame())
        self.assertIn('uniprot', out.columns)


class RankingIsolationTests(unittest.TestCase):
    def test_ligand_evidence_is_absent_from_the_priority_formula(self):
        from amide_synthesis import master_table
        source = Path(master_table.__file__).read_text().lower()
        self.assertNotIn('bound_ligand', source)
        self.assertNotIn('bound_bifunctional', source)


if __name__ == '__main__':
    unittest.main()
