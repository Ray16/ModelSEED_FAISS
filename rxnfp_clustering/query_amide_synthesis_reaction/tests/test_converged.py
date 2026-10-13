"""Controls for evidence convergence.

The failure that matters is counting correlated observations as independent,
which inflates the exact quantity this file measures. The first version did
it: cofactor-bound structure and in-line adenylation geometry were separate
axes, but the geometry is measured FROM those structures.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from amide_synthesis import converged as cv   # noqa: E402


def row(**kw):
    base = {'uniprot': 'P1', 'evidence_strength': 'family_homolog',
            'pfam': 'PF00501', 'interpro': 'IPR000873',
            'substrate_distance_measured': 0.0,
            'substrate_distance_own_rhea': 0.0,
            'pdb_has_atp_like': False, 'd_nuc_P': None,
            'bound_bifunctional_ligand': None, 'struct_S': None}
    base.update(kw)
    return pd.DataFrame([base])


class IndependenceTests(unittest.TestCase):
    def test_a_cofactor_structure_and_its_geometry_count_once(self):
        """The geometry is derived from the structure; it is one observation."""
        both = cv.axes(row(pdb_has_atp_like=True, d_nuc_P=3.2), set())
        self.assertEqual(int(both.sum(axis=1).iloc[0]), 1)

    def test_geometry_alone_still_counts(self):
        one = cv.axes(row(d_nuc_P=3.2), set())
        self.assertTrue(bool(one.adenylation_competent_structure.iloc[0]))

    def test_distinct_observations_each_count(self):
        many = cv.axes(row(substrate_distance_measured=0.9,
                           bound_bifunctional_ligand='API',
                           struct_S=0.5), set())
        self.assertEqual(int(many.sum(axis=1).iloc[0]), 3)


class AxisTests(unittest.TestCase):
    def test_chemistry_uses_the_best_of_modelseed_and_curated(self):
        self.assertTrue(bool(cv.axes(
            row(substrate_distance_own_rhea=0.8), set())
            .own_reaction_chemistry.iloc[0]))

    def test_inherited_distance_does_not_count_as_the_protein_s_own(self):
        """A relative's chemistry is not an independent line about this protein."""
        frame = row(substrate_distance_measured=0.0)
        frame['substrate_distance_inherited'] = 0.95
        self.assertFalse(bool(cv.axes(frame, set())
                              .own_reaction_chemistry.iloc[0]))

    def test_below_the_floor_does_not_count(self):
        self.assertFalse(bool(cv.axes(
            row(substrate_distance_measured=cv.CHEMISTRY_FLOOR - 0.01), set())
            .own_reaction_chemistry.iloc[0]))

    def test_a_lead_family_signature_counts(self):
        self.assertTrue(bool(cv.axes(row(), {'PF00501'})
                             .lead_family_verdict.iloc[0]))

    def test_a_non_lead_signature_does_not(self):
        self.assertFalse(bool(cv.axes(row(), {'PF99999'})
                              .lead_family_verdict.iloc[0]))

    def test_measured_activity_counts(self):
        self.assertTrue(bool(cv.axes(
            row(evidence_strength='measured_activity'), set())
            .measured_nylon_activity.iloc[0]))

    def test_a_candidate_with_nothing_scores_zero(self):
        self.assertEqual(int(cv.axes(row(), set()).sum(axis=1).iloc[0]), 0)

    def test_absent_columns_do_not_raise(self):
        bare = pd.DataFrame([{'uniprot': 'P1',
                              'evidence_strength': 'clade_homolog'}])
        self.assertEqual(int(cv.axes(bare, set()).sum(axis=1).iloc[0]), 0)


class LeadSignatureTests(unittest.TestCase):
    def test_only_lead_families_contribute(self):
        evaluation = pd.DataFrame([
            {'verdict': 'lead', 'signatures': 'PF1;IPR1'},
            {'verdict': 'reject', 'signatures': 'PF2'},
            {'verdict': 'watch', 'signatures': 'PF3'}])
        self.assertEqual(cv.lead_signatures(evaluation), {'PF1', 'IPR1'})


if __name__ == '__main__':
    unittest.main()
