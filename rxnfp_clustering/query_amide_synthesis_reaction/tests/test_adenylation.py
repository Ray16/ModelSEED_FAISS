"""Controls for adenylation geometry measured on deposited structures.

The finding here was negative - no deposited structure holds a substrate
carboxylate at the phosphorus - so the guards that matter are the ones that
stop a buffer ion being reported as a substrate.
"""
from __future__ import annotations

import math
import unittest

from amide_synthesis.structure_modeling import adenylation_geometry as ag


class GeometryMathTests(unittest.TestCase):
    def test_inline_attack_is_180_degrees(self):
        angle = ag._angle((-1, 0, 0), (0, 0, 0), (1, 0, 0))
        self.assertAlmostEqual(angle, 180.0, places=4)

    def test_right_angle(self):
        self.assertAlmostEqual(ag._angle((1, 0, 0), (0, 0, 0), (0, 1, 0)),
                               90.0, places=4)

    def test_degenerate_geometry_returns_none(self):
        self.assertIsNone(ag._angle((0, 0, 0), (0, 0, 0), (1, 0, 0)))

    def test_distance(self):
        self.assertAlmostEqual(ag._distance((0, 0, 0), (3, 4, 0)), 5.0)


class ArtifactTests(unittest.TestCase):
    """A sulfate ion is not a substrate, however well placed it is."""

    def test_buffer_components_are_classified_as_artifacts(self):
        for ligand in ('SO4', 'PO4', 'GOL', 'EDO', 'MES', 'CIT'):
            self.assertIn(ligand, ag.ARTIFACT_LIGANDS)

    def test_a_real_substrate_is_not_an_artifact(self):
        for ligand in ('ADP', 'ATP', 'GLU', 'LYS'):
            self.assertNotIn(ligand, ag.ARTIFACT_LIGANDS)

    def test_cofactor_is_never_offered_as_its_own_nucleophile(self):
        """The ATP oxygens must not be read as the attacking carboxylate."""
        self.assertTrue(ag.ATP_LIKE.isdisjoint(ag.ARTIFACT_LIGANDS))


class PhosphorusChoiceTests(unittest.TestCase):
    def test_alpha_phosphate_is_preferred(self):
        """Adenylation attacks P-alpha; PB/PG are the leaving groups."""
        self.assertLess(ag.PHOSPHORUS_PREFERENCE.index('PA'),
                        ag.PHOSPHORUS_PREFERENCE.index('PB'))
        self.assertLess(ag.PHOSPHORUS_PREFERENCE.index('PB'),
                        ag.PHOSPHORUS_PREFERENCE.index('PG'))


class ResidueClassificationTests(unittest.TestCase):
    def test_waters_and_amino_acids_are_not_treated_as_ligands(self):
        self.assertIn('HOH', ag.WATERS)
        for residue in ('ALA', 'ASP', 'MSE'):
            self.assertIn(residue, ag.AMINO_ACIDS)

    def test_carboxylate_side_chains_are_enumerated(self):
        self.assertEqual(ag.PROTEIN_CARBOXYLATE['ASP'], ('OD1', 'OD2'))
        self.assertEqual(ag.PROTEIN_CARBOXYLATE['GLU'], ('OE1', 'OE2'))


class FailureHandlingTests(unittest.TestCase):
    def test_failed_download_is_reported(self):
        out = ag.ensure_structure('XXXX', fetch=lambda u, timeout=None: (b'', {}, True))
        self.assertIsNone(out)

    def test_unparseable_file_is_flagged(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'bad.cif'
            path.write_text('not a structure')
            facts = ag.measure(path, pdb_id='BAD')
        self.assertTrue(facts.get('parse_failed') or facts.get('no_phosphorus'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
