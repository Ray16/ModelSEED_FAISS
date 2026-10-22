"""Expose the existing curated chemistry controls to unittest discovery."""
import unittest
from rdkit import Chem

from amide_synthesis.acylation import acylated_acceptors, count_free_amines
from amide_synthesis.normalize import normalize_smiles
from amide_synthesis.enzyme_resolution import product_names
from scripts.test_screens import REPAIR_CASES, AMINE_CASES, ACYLATION_CASES, PRODUCT_NAME_CASES


class ChemistryControls(unittest.TestCase):
    def test_structure_repairs(self):
        for name, broken, expected in REPAIR_CASES:
            with self.subTest(name=name):
                self.assertEqual(normalize_smiles(broken), Chem.CanonSmiles(expected))

    def test_amine_classifier(self):
        for name, smiles, expected in AMINE_CASES:
            with self.subTest(name=name):
                self.assertEqual(count_free_amines(Chem.MolFromSmiles(smiles)), expected)

    def test_acylation_screen(self):
        for name, reaction, expected_pass, expected_amines in ACYLATION_CASES:
            with self.subTest(name=name):
                acceptors = acylated_acceptors(reaction)
                self.assertEqual(bool(acceptors), expected_pass)
                self.assertEqual(max((n for _, n in acceptors), default=0), expected_amines)

    def test_product_name_controls(self):
        for name, definition, expected in PRODUCT_NAME_CASES:
            with self.subTest(name=name):
                self.assertEqual(product_names(definition), expected)
