"""Controls for the KEGG orthology route.

The route exists because a reaction with no Rhea id, no EC and no usable name
is unresolvable through UniProt but may still carry a KEGG ortholog. The
failure modes worth pinning are the parsing ones: a partial EC that would
expand to a whole sub-subclass, and a malformed alias row reaching the
network.
"""
from __future__ import annotations

import unittest

import pandas as pd

from amide_synthesis.enzyme_mapping import kegg_orthologs as ko


class TestKeggReactionPattern(unittest.TestCase):

    def test_only_well_formed_reaction_ids_match(self):
        for good in ('R00004', 'R09256', 'R10745'):
            self.assertTrue(ko.KEGG_REACTION.match(good), good)
        for bad in ('R_R00004_c', 'R1234', 'R123456', 'rxn00001', 'C00001',
                    'R00004.c', ''):
            self.assertFalse(ko.KEGG_REACTION.match(bad), bad)


class TestCompleteEC(unittest.TestCase):
    """A partial EC names a sub-subclass, not an activity.

    `2.3.1.-` would expand to every acyltransferase in the sub-subclass,
    which is the mass-admission this project refuses everywhere else.
    """

    def test_complete_ecs_accepted(self):
        for good in ('6.3.2.11', '2.3.1.64', '1.1.1.1'):
            self.assertTrue(ko.COMPLETE_EC.match(good), good)

    def test_partial_ecs_rejected(self):
        for bad in ('2.3.1.-', '6.3.-.-', '6.-.-.-', '2.3.1', '6.3.2.M5',
                    '2.3.1.dd'):
            self.assertFalse(ko.COMPLETE_EC.match(bad), bad)


class TestECExtraction(unittest.TestCase):

    def test_ec_pulled_out_of_a_ko_name(self):
        found = ko.EC_IN_NAME.search('carnosine synthase [EC:6.3.2.11]')
        self.assertEqual(found.group(1).strip(), '6.3.2.11')

    def test_name_without_an_ec(self):
        self.assertIsNone(ko.EC_IN_NAME.search('puromycin biosynthesis Pur6'))

    def test_only_the_complete_ec_survives_a_mixed_name(self):
        name = 'agmatine coumaroyltransferase [EC:2.3.1.64 2.3.1.-]'
        found = ko.EC_IN_NAME.search(name)
        kept = [e for e in found.group(1).split()
                if ko.COMPLETE_EC.match(e)]
        self.assertEqual(kept, ['2.3.1.64'])


class TestAliasFiltering(unittest.TestCase):
    """Malformed alias rows must never reach the network."""

    def test_non_kegg_and_malformed_rows_are_dropped(self):
        frame = pd.DataFrame({
            'ModelSEED ID': ['rxn1', 'rxn2', 'rxn3', 'rxn4'],
            'External ID': ['R00004', 'R_R00004_c', 'R00005', 'MNXR1'],
            'Source': ['KEGG', 'AlgaGEM', 'KEGG', 'metanetx.reaction'],
        })
        path = self.make_file(frame)
        out = ko.kegg_aliases(path)
        self.assertEqual(sorted(out.external), ['R00004', 'R00005'])

    def make_file(self, frame):
        import tempfile
        import pathlib
        import shutil
        directory = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, directory)
        path = directory / 'aliases.txt'
        frame.to_csv(path, sep='\t', index=False)
        return path


class TestRouteIsRegistered(unittest.TestCase):
    """A route whose file is not registered is silently invisible.

    `active_homolog` was added without being listed in `ec_substrate`, and
    its 822 proteins got no EC ceiling while every file looked healthy.
    """

    def test_registered_with_master_and_ec_substrate(self):
        from amide_synthesis.candidate_selection import ec_substrate, master_table
        self.assertIn('kegg_ortholog_candidates.tsv',
                      master_table.ROUTE_FILES)
        self.assertIn('kegg_ortholog_candidates.tsv',
                      {name for name, _ in ec_substrate.SOURCES})

    def test_evidence_tier_has_a_weight_and_a_share(self):
        from amide_synthesis.candidate_selection import shortlist
        from amide_synthesis.candidate_selection.master_table import EVIDENCE_WEIGHT
        # The route reuses `activity_ec` rather than adding a tier.
        self.assertIn('activity_ec', EVIDENCE_WEIGHT)
        self.assertIn('activity_ec', dict(shortlist.TIER_SHARE))


if __name__ == '__main__':
    unittest.main()
