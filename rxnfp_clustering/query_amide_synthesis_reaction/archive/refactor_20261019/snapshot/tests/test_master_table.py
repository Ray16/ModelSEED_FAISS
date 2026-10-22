"""Tests for the name-based disqualifier in `master_table`."""

import re
import unittest

from amide_synthesis.master_table import DISQUALIFYING


class TestBindingProteinDisqualifier(unittest.TestCase):
    """`binding protein` must not disqualify the ANL adenylating family.

    The bare term was meant for periplasmic/ABC solute receptors but matched
    "AMP-binding protein", UniProt's standard name for the superfamily the
    three MEASURED nylon-bond producers belong to. It was silently penalising
    36 adenylating enzymes, 12 of them homologs of a measured active.
    """

    PATTERN = '|'.join(DISQUALIFYING)

    def _disqualified(self, name):
        return bool(re.search(self.PATTERN, name.lower()))

    def test_nucleotide_binding_names_survive(self):
        for name in ('AMP-binding protein',
                     'ATP-binding protein with DUF201 domain',
                     'AMP binding protein',
                     'NAD-binding protein'):
            self.assertFalse(self._disqualified(name), name)

    def test_transport_and_nucleic_acid_receptors_still_disqualified(self):
        for name in ('Periplasmic binding protein',
                     'Substrate-binding protein',
                     'Maltose/maltodextrin-binding protein',
                     'ABC transporter solute-binding protein',
                     'Branched-chain amino acid binding protein',
                     'DNA-binding protein',
                     'Penicillin-binding protein'):
            self.assertTrue(self._disqualified(name), name)
