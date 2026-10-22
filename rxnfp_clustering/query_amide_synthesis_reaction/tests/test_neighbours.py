"""Controls for the chemistry-neighbour route.

This route INFERS rather than looks up, so the guards matter more than usual:
a protein borrowed from a chemically different reaction is just noise.
"""
from __future__ import annotations

import unittest

import pandas as pd

from amide_synthesis.enzyme_mapping import chemistry_neighbours as cn

NYLON = ('O=C([O-])CCCCC(=O)[O-].[NH3+]CCCCCC[NH3+]'
         '>>[NH3+]CCCCCCNC(=O)CCCCC(=O)[O-].O')
NYLON_C5 = ('O=C([O-])CCCC(=O)[O-].[NH3+]CCCCCC[NH3+]'
            '>>[NH3+]CCCCCCNC(=O)CCCC(=O)[O-].O')
ACETYL = 'CC(=O)[O-].[NH3+]CC>>CC(=O)NCC.O'


def hits(rows):
    return pd.DataFrame(rows)


def enzymes(rows):
    return pd.DataFrame(rows)


class NeighbourTests(unittest.TestCase):
    def setUp(self):
        self.enz = enzymes([{'rxn_id': 'solved', 'uniprot': 'P1',
                             'gene': 'g', 'protein_name': 'synthetase',
                             'organism': 'o', 'sequence': 'MK', 'ec': '6.3.2.1',
                             'pfam': 'PF04183'}])

    def test_equivalent_chemistry_borrows_the_enzyme(self):
        frame = cn.build(hits([
            {'rxn_id': 'solved', 'rxn_smiles': NYLON, 'substrate_distance': .9},
            {'rxn_id': 'open', 'rxn_smiles': NYLON_C5, 'substrate_distance': .9},
        ]), self.enz)
        self.assertEqual(frame.uniprot.tolist(), ['P1'])
        self.assertEqual(frame.borrowed_from_rxn.iloc[0], 'solved')

    def test_provenance_is_recorded_not_hidden(self):
        frame = cn.build(hits([
            {'rxn_id': 'solved', 'rxn_smiles': NYLON, 'substrate_distance': .9},
            {'rxn_id': 'open', 'rxn_smiles': NYLON_C5, 'substrate_distance': .9},
        ]), self.enz)
        row = frame.iloc[0]
        self.assertEqual(row.evidence_strength, 'chemistry_neighbour')
        self.assertIn('borrowed_from_rxn', frame.columns)
        self.assertGreaterEqual(row.chemistry_feature_distance, 0)

    def test_different_chemistry_is_not_bridged(self):
        """A terminal acetamide must not borrow a polycondensation enzyme."""
        frame = cn.build(hits([
            {'rxn_id': 'solved', 'rxn_smiles': NYLON, 'substrate_distance': .9},
            {'rxn_id': 'open', 'rxn_smiles': ACETYL, 'substrate_distance': .9},
        ]), self.enz, max_distance=3.0)
        self.assertTrue(frame.empty)

    def test_low_scoring_reactions_are_not_bridged(self):
        frame = cn.build(hits([
            {'rxn_id': 'solved', 'rxn_smiles': NYLON, 'substrate_distance': .9},
            {'rxn_id': 'open', 'rxn_smiles': NYLON_C5, 'substrate_distance': .1},
        ]), self.enz, min_sd=.6)
        self.assertTrue(frame.empty)

    def test_already_resolved_reactions_are_not_bridged(self):
        frame = cn.build(hits([
            {'rxn_id': 'solved', 'rxn_smiles': NYLON, 'substrate_distance': .9},
        ]), self.enz)
        self.assertTrue(frame.empty)

    def test_borrowing_is_capped(self):
        many = enzymes([dict(rxn_id='solved', uniprot=f'P{i}', gene='g',
                             protein_name='s', organism='o', sequence='MK',
                             ec='', pfam='') for i in range(40)])
        frame = cn.build(hits([
            {'rxn_id': 'solved', 'rxn_smiles': NYLON, 'substrate_distance': .9},
            {'rxn_id': 'open', 'rxn_smiles': NYLON_C5, 'substrate_distance': .9},
        ]), many, max_borrowed=5)
        self.assertEqual(len(frame), 5)

    def test_no_resolved_pool_is_not_fatal(self):
        frame = cn.build(hits([
            {'rxn_id': 'open', 'rxn_smiles': NYLON, 'substrate_distance': .9},
        ]), enzymes([]).reindex(columns=['rxn_id', 'uniprot']))
        self.assertTrue(frame.empty)


if __name__ == '__main__':
    unittest.main(verbosity=2)
