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

class RejectedAxisTests(unittest.TestCase):
    """A curated ATP-binding site is a precondition, not an independent line.

    Measured: adding it takes the >= 3-line list from 8 to 1,759, because
    binding ATP is shared by essentially every ATP-dependent ligase and this
    candidate set is selected for ligases.
    """

    def test_the_rejection_is_recorded_with_its_reason(self):
        self.assertIn('curated_atp_binding_site', cv.REJECTED_AXES)
        self.assertIn('1,759', cv.REJECTED_AXES['curated_atp_binding_site'])

    def test_atp_binding_is_not_an_axis(self):
        found = cv.axes(row(), set())
        for name in found.columns:
            self.assertNotIn('atp', name.lower())

    def test_a_protein_that_only_binds_atp_scores_nothing(self):
        frame = row()
        frame['binding_site'] = ('BINDING 150..205; /ligand="ATP"; '
                                 '/ligand_id="ChEBI:CHEBI:30616";')
        self.assertEqual(int(cv.axes(frame, set()).sum(axis=1).iloc[0]), 0)

class DiamineAxisTests(unittest.TestCase):
    """Acylating a free DIAMINE is the acceptor-side nylon requirement.

    It qualifies where curated ATP binding did not: 0.35% coverage against
    22.4%, and Jaccard 0.01 against the chemistry axis rather than measuring
    the same thing.
    """

    def test_a_proven_free_diamine_acceptor_counts(self):
        found = cv.axes(row(acylates_free_diamine=True), set())
        self.assertTrue(bool(found.acylates_a_free_diamine.iloc[0]))

    def test_it_is_a_discriminating_axis_not_a_weak_one(self):
        self.assertNotIn('acylates_a_free_diamine', cv.WEAK_AXES)

    def test_absence_does_not_count(self):
        self.assertFalse(bool(cv.axes(row(acylates_free_diamine=False), set())
                              .acylates_a_free_diamine.iloc[0]))

    def test_a_missing_column_does_not_raise(self):
        self.assertFalse(bool(cv.axes(row(), set())
                              .acylates_a_free_diamine.iloc[0]))

    def test_it_is_independent_of_the_chemistry_axis(self):
        """A diamine acceptor with weak distance still earns its own line."""
        found = cv.axes(row(acylates_free_diamine=True,
                            substrate_distance_measured=0.1), set())
        self.assertTrue(bool(found.acylates_a_free_diamine.iloc[0]))
        self.assertFalse(bool(found.own_reaction_chemistry.iloc[0]))


if __name__ == '__main__':
    unittest.main()
