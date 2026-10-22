"""Meaningful controls for fingerprints, direction and balanced inputs."""
import unittest
import numpy as np
from drfp import DrfpEncoder
from compare_native_reactions import build_inputs, reaction, similarity


class NativeSimilarityTests(unittest.TestCase):
    def test_comparison_inputs_keep_native_cofactors(self):
        records, _, _, _ = build_inputs()
        by_name = {r['name']: r['rxn_smiles'] for r in records}
        self.assertEqual(set(by_name), {'target_amp', 'target_adp', 'DdaG_full', 'SfaB_full', 'McbA_full', 'PylC_full'})
        for rxn in by_name.values():
            left, right = rxn.split('>>')
            self.assertIn('P', left)
            self.assertIn('P', right)
        self.assertNotEqual(by_name['target_amp'], by_name['target_adp'])

    def test_binary_metrics_are_distinct(self):
        a, b = np.array([1, 1, 0]), np.array([1, 0, 1])
        self.assertAlmostEqual(similarity(a, b, 'tanimoto'), 1 / 3)
        self.assertAlmostEqual(similarity(a, b, 'cosine'), 1 / 2)
        with self.assertRaises(AssertionError):
            similarity(np.array([-1, 1]), np.array([1, 1]), 'tanimoto')

    def test_signed_difference_preserves_direction(self):
        v = np.array([-2, 3, 1])
        self.assertAlmostEqual(similarity(v, v, 'cosine'), 1)
        self.assertAlmostEqual(similarity(v, -v, 'cosine'), -1)

    def test_invalid_balance_and_polymer_are_rejected(self):
        with self.assertRaises(ValueError): reaction(['CC(=O)O', 'CN'], ['CC(=O)NC'])
        with self.assertRaises(ValueError): reaction(['*C(=O)O', 'CN'], ['*C(=O)NC', 'O'])

    def test_drfp_reverse_direction_is_not_independent_evidence(self):
        forward = reaction(['CC(=O)O', 'CN'], ['CC(=O)NC', 'O'])
        left, right = forward.split('>>')
        vectors = np.asarray(DrfpEncoder.encode([forward, right + '>>' + left], n_folded_length=2048, radius=3))
        self.assertAlmostEqual(similarity(vectors[0], vectors[1], 'tanimoto'), 1)


if __name__ == '__main__': unittest.main()
