"""Check chemical distinctions that whole-reaction fingerprints can obscure."""
import unittest
from rdkit import Chem
from nylon_chemical_review import assess, primary_amines, span


class NylonChemicalReviewTests(unittest.TestCase):
    def test_primary_amine_excludes_hydroxylamine_secondary_and_guanidine(self):
        cases = [('NCCCCCCN', 2), ('NCCCCNO', 1), ('NCCCCNCCCN', 2),
                 ('N[C@@H](CCCNC(=N)N)C(=O)O', 1), ('NCC(=O)N', 1)]
        for smiles, expected in cases:
            with self.subTest(smiles=smiles):
                self.assertEqual(len(primary_amines(Chem.MolFromSmiles(smiles))), expected)

    def test_target_is_a_diacid_primary_diamine_analogue(self):
        result = assess('O=C(O)CCCCC(=O)O.NCCCCCCN>>O=C(O)CCCCC(=O)NCCCCCCN.O')
        self.assertTrue(result['shortlist_eligible'])
        self.assertEqual(result['chemical_tier'], 'diacid–primary-diamine analogue')
        self.assertEqual((result['acid_spacer_carbons'], result['amine_spacer_carbons']), (4, 6))
        self.assertTrue(result['product_retains_primary_amine'] and result['product_retains_carboxyl'])

    def test_n_hydroxy_acylation_is_not_plain_primary_amine_acylation(self):
        result = assess('O=C(O)CCC(=O)O.NCCCCNO>>NCCCCN(O)C(=O)CCC(=O)O.O')
        self.assertFalse(result['shortlist_eligible'])
        self.assertFalse(result['reacts_at_primary_amine'])

    def test_ab_self_condensation_is_distinguished(self):
        result = assess('NCC(=O)O.NCC(=O)O>>NCC(=O)NCC(=O)O.O')
        self.assertTrue(result['shortlist_eligible'])
        self.assertTrue(result['self_condensation'])

    def test_omega_amino_acid_self_condensation_is_allowed(self):
        result = assess('NCCCCCC(=O)O.NCCCCCC(=O)O>>NCCCCCC(=O)NCCCCCC(=O)O.O')
        self.assertTrue(result['shortlist_eligible'])
        self.assertEqual(result['carboxyl_groups_on_donor'], 1)
        self.assertEqual(result['primary_amines_on_acceptor'], 1)

    def test_aromatic_and_cyclic_substrates_are_not_excluded(self):
        for reaction in ['O=C(O)c1ccccc1.CN>>CNC(=O)c1ccccc1.O',
                         'O=C(O)C1CCCCC1.CN>>CNC(=O)C1CCCCC1.O']:
            result = assess(reaction)
            self.assertTrue(result['shortlist_eligible'])
            self.assertFalse(result['acyclic_aliphatic_partners'])

    def test_legacy_rules_are_explicit_and_not_the_default(self):
        reaction = 'NCC(=O)O.NCC(=O)O>>NCC(=O)NCC(=O)O.O'
        self.assertTrue(assess(reaction)['shortlist_eligible'])
        self.assertFalse(assess(reaction, strict=True)['shortlist_eligible'])

    def test_spacer_does_not_count_internal_nitrogen_as_carbon(self):
        mol = Chem.MolFromSmiles('NCCCCNCCCN')
        self.assertEqual(span(mol, primary_amines(mol)), (7, 1))


if __name__ == '__main__': unittest.main()
