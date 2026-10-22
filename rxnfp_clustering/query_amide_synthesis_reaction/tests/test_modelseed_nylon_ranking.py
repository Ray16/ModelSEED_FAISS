"""Protect route matching and the separation of selection from scoring."""
import unittest
from rank_modelseed_nylon import canonical_reaction, organic_group, target_route


class ModelseedNylonRankingTests(unittest.TestCase):
    def setUp(self):
        self.species = {'ATP': 'ATP', 'AMP': 'AMP', 'ADP': 'ADP', 'PPI': 'PPi', 'PI': 'Pi', 'COA': 'CoA'}

    def test_route_is_selected_from_native_cofactors(self):
        self.assertEqual(target_route('acid.amine.ATP>>amide.AMP.PPi', self.species), 0)
        self.assertEqual(target_route('acid.amine.ATP>>amide.ADP.Pi', self.species), 1)
        self.assertEqual(target_route('acylCoA.amine>>amide.CoA', self.species), 2)
        self.assertEqual(target_route('acid.amine>>amide.O', self.species), 3)

    def test_alternative_bookkeeping_is_grouped_without_altering_full_reaction(self):
        amp = 'acid.acid.ATP>>dimer.AMP.PPi'
        adp = 'acid.acid.ATP>>dimer.ADP.Pi'
        trimer = 'acid.acid.acid.ATP>>trimer.O.AMP.PPi'
        self.assertEqual(organic_group(amp, self.species), organic_group(adp, self.species))
        self.assertNotEqual(organic_group(amp, self.species), organic_group(trimer, self.species))
        self.assertIn('ATP', amp)
        self.assertIn('ADP', adp)

    def test_canonicalization_keeps_stoichiometric_copies(self):
        self.assertEqual(canonical_reaction('N.CC(=O)O.CC(=O)O>>CC(N)=O'),
                         canonical_reaction('CC(=O)O.N.CC(=O)O>>CC(N)=O'))
        self.assertNotEqual(canonical_reaction('CC(=O)O.N.CC(=O)O>>CC(N)=O'),
                            canonical_reaction('CC(=O)O.N>>CC(N)=O'))


if __name__ == '__main__': unittest.main()
