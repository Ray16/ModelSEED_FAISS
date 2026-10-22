"""Controls for the polymer-vs-protein distinction in the macromolecular screen.

A POLYMER acceptor is not a PROTEIN acceptor, and collapsing them excluded the
enzyme class closest to a nylon polymerase: an enzyme extending its own growing
polyamide. Protein acylation stays excluded - you cannot offer HMDA to an
enzyme whose acceptor is a protein lysine.
"""
from __future__ import annotations

import unittest

from amide_synthesis.reaction_search.amide_enzymes import is_macromolecular, is_polymer_elongation


class CoenzymeFalsePositiveTests(unittest.TestCase):
    """The stem 'enzyme' is a substring of 'coenzyme'. Shipped bug.

    It flagged every coenzyme-bearing reaction as macromolecular, which
    silently excluded the coenzyme F420 polyglutamate ligases (CofE/CofF) -
    iterative glutamylation, among the most nylon-relevant chemistry here.
    """

    def test_coenzyme_is_not_a_macromolecule(self):
        self.assertFalse(is_macromolecular(
            '(1) L-Glutamate[0] + (1) GTP[0] + (1) Coenzyme F420-1[0] <=> ...',
            [], {}, {}))

    def test_hyphenated_co_enzyme_too(self):
        self.assertFalse(is_macromolecular('(1) co-enzyme A[0] <=> ...',
                                           [], {}, {}))

    def test_a_real_enzyme_substrate_is_still_flagged(self):
        self.assertTrue(is_macromolecular(
            '(1) ATP[0] + (1) enzyme N6-(lipoyl)lysine[0] <=> ...', [], {}, {}))

    def test_compound_names_are_masked_too(self):
        self.assertFalse(is_macromolecular(
            '', ['cpd1'], {'cpd1': 'Coenzyme F420-2'}, {}))


class PolymerElongationTests(unittest.TestCase):
    def test_cyanophycin_is_polymer_elongation(self):
        self.assertTrue(is_polymer_elongation(
            '(1) ATP[0] + (1) L-Arginine[0] + (1) Cyanophycin-aspartate[0] <=> ...',
            [], {}))

    def test_poly_gamma_glutamyl_elongation(self):
        self.assertTrue(is_polymer_elongation(
            '(1) ATP[0] + (1) L-Glutamate[0] '
            '+ (1) gamma-L-glutamyl-L-glutamate residue[0] <=> ...', [], {}))

    def test_carrier_protein_is_not_readmitted(self):
        """Protein evidence wins: this acceptor is a folded protein."""
        self.assertFalse(is_polymer_elongation(
            '(1) ATP[0] + (1) alpha-Aminoadipate carrier protein LysW[0] '
            '+ (1) Protein-L-lysine[0] <=> ...', [], {}))

    def test_protein_evidence_beats_a_polymer_term(self):
        self.assertFalse(is_polymer_elongation(
            '(1) polyglutamate carrier protein[0] <=> ...', [], {}))

    def test_trna_and_ribosome_stay_excluded(self):
        for text in ('(1) L-alanyl-tRNA[0] <=> ...',
                     '(1) ribosomal protein L16[0] <=> ...'):
            self.assertFalse(is_polymer_elongation(text, [], {}))

    def test_an_ordinary_small_molecule_reaction_is_not_polymer_elongation(self):
        self.assertFalse(is_polymer_elongation(
            '(1) acetyl-CoA[0] + (1) putrescine[0] <=> ...', [], {}))

    def test_compound_names_contribute_evidence(self):
        self.assertTrue(is_polymer_elongation(
            '', ['cpd1'], {'cpd1': 'cyanophycin primer-L-aspartate'}))


if __name__ == '__main__':
    unittest.main(verbosity=2)


class TubulinTests(unittest.TestCase):
    """A glutamate chain on TUBULIN is protein modification, not polyamide.

    rxn58922/rxn58972 matched the polymer term `glutamyl-l-glutamate` and
    were re-admitted, because no generic stem catches the name "tubulin".
    """

    def test_tubulin_polyglutamylation_is_not_polymer_elongation(self):
        definition = ('(1) ATP + (1) L-Glutamate + (1) gamma-L-glutamyl-'
                      'L-glutamate residue <=> (1) ADP + (1) Phosphate + '
                      '(1) a [tubulin]-(alpha-glutamyl-gamma-L-glutamyl)-'
                      'L-glutamate')
        self.assertFalse(is_polymer_elongation(definition, [], {}))

    def test_beta_tubulin_too(self):
        self.assertFalse(is_polymer_elongation(
            'a [beta-tubulin]-(alpha-L-glutamyl-gamma-L-glutamyl)-L-glutamate',
            [], {}))

    def test_a_genuine_polyamide_is_still_admitted(self):
        """The fix must not cost the chemistry the re-admission was for."""
        self.assertTrue(is_polymer_elongation(
            'cyanophycin + L-aspartate + ATP', [], {}))
        self.assertTrue(is_polymer_elongation(
            'coenzyme F420-2 + L-glutamate', [], {}))
