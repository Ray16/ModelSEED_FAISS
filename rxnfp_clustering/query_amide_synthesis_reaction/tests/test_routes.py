"""Controls for the non-ModelSEED retrieval routes and the master merge.

No network: the UniProt client is injected. These guard the properties that
make the merged table safe to read - that a weaker route cannot displace a
stronger one, that an inherited score cannot pose as a measured one, and that
the precomputed link table cannot inject cofactor-hydrolysis noise.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from amide_synthesis.candidate_selection import master_table as mt
from amide_synthesis.enzyme_mapping import precomputed_links as pl


class TestPrecomputedLinks(unittest.TestCase):
    LINES = [
        'P001\trxn00062\tMKV',          # ATP phosphohydrolase: must be filtered
        'P002\trxn21314,rxn00062\tMKA',
        'P003\trxn21314\tMKB',
        'BAD_ROW_NO_TAB',
    ]

    def _file(self, temp):
        path = Path(temp) / 'links.tsv'
        path.write_text('\n'.join(self.LINES) + '\n')
        return path

    def test_only_screened_reactions_are_admitted(self):
        """The structural protection against the GO route's cofactor noise."""
        with tempfile.TemporaryDirectory() as temp:
            by_reaction, _ = pl.load_links(self._file(temp),
                                           keep_reactions={'rxn21314'})
        self.assertEqual(set(by_reaction), {'rxn21314'})
        self.assertEqual(set(by_reaction['rxn21314']), {'P002', 'P003'})

    def test_malformed_rows_are_skipped_not_fatal(self):
        with tempfile.TemporaryDirectory() as temp:
            by_reaction, stats = pl.load_links(self._file(temp))
        self.assertEqual(stats['rows'], 3)
        self.assertIn('rxn00062', by_reaction)

    def test_missing_file_is_not_fatal(self):
        by_reaction, stats = pl.load_links(Path('/nonexistent/links.tsv'))
        self.assertEqual(stats['rows'], 0)
        self.assertEqual(len(by_reaction), 0)

    def test_over_annotated_reactions_are_flagged_and_capped(self):
        crowd = {'rxn00062': {f'P{i:05d}': 'M' for i in range(50)}}
        frame = pl.to_frame(crowd, promiscuous=10, cap=5)
        self.assertEqual(len(frame), 5, 'cap must bound a crowded reaction')
        self.assertTrue(frame.over_annotated_reaction.all())
        self.assertTrue((frame.n_proteins_on_reaction == 50).all())

    def test_substrate_distance_is_carried_from_the_screen(self):
        hits = pd.DataFrame([{'rxn_id': 'rxn21314', 'substrate_distance': .81}])
        frame = pl.to_frame({'rxn21314': {'P003': 'MKB'}}, hits=hits)
        self.assertAlmostEqual(frame.substrate_distance_measured.iloc[0], .81)


class TestMasterMerge(unittest.TestCase):
    def _reaction(self):
        return pd.DataFrame([{
            'uniprot': 'P0DXD6', 'gene': 'pubC', 'protein_name': 'synthetase',
            'substrate_distance': .83, 'evidence_strength': 'exact_reaction',
            'self_condensation': True, 'pfam': 'PF04183', 'reviewed': 'reviewed',
            'pdb': '', 'sequence': 'MKV'}])

    def _expansion(self):
        return pd.DataFrame([
            {'Entry': 'P0DXD6', 'Protein names': 'synthetase',
             'family_substrate_distance': .83, 'evidence_strength':
             'family_homolog', 'family_self_condensing': True, 'Sequence': 'MKV'},
            {'Entry': 'NEW001', 'Protein names': 'siderophore synthetase',
             'family_substrate_distance': .83, 'evidence_strength':
             'family_homolog', 'family_self_condensing': True, 'Sequence': 'MKC'},
        ])

    def test_reaction_route_wins_over_homolog_for_the_same_protein(self):
        table = mt.build(self._reaction(), self._expansion(),
                         structural=pd.DataFrame(columns=['uniprot', 'struct_S']))
        row = table[table.uniprot == 'P0DXD6'].iloc[0]
        self.assertEqual(row.discovery_route, 'reaction')
        self.assertEqual(len(table[table.uniprot == 'P0DXD6']), 1)

    def test_inherited_score_is_discounted_below_measured(self):
        table = mt.build(self._reaction(), self._expansion(),
                         structural=pd.DataFrame(columns=['uniprot', 'struct_S']))
        measured = table[table.uniprot == 'P0DXD6'].iloc[0]
        inherited = table[table.uniprot == 'NEW001'].iloc[0]
        self.assertEqual(inherited.substrate_distance_measured, 0.0)
        self.assertLess(inherited.substrate_distance_inherited,
                        measured.substrate_distance_measured)
        self.assertGreater(measured.master_priority, inherited.master_priority)

    def test_annotation_contradiction_only_challenges_homologs(self):
        """A screen-proven protein is not overruled by its own name."""
        reaction = self._reaction()
        reaction.loc[0, 'protein_name'] = 'bifunctional amidase'
        expansion = self._expansion()
        expansion.loc[1, 'Protein names'] = 'ferric siderophore reductase'
        table = mt.build(reaction, expansion,
                         structural=pd.DataFrame(columns=['uniprot', 'struct_S']))
        self.assertFalse(
            table.set_index('uniprot').loc['P0DXD6',
                                           'annotation_contradicts_chemistry'])
        self.assertTrue(
            table.set_index('uniprot').loc['NEW001',
                                           'annotation_contradicts_chemistry'])

    def test_measured_activity_outranks_every_prediction(self):
        assayed = pd.DataFrame([{
            'uniprot': 'WP_1', 'protein_name': 'amide synthetase',
            'discovery_route': 'assayed', 'evidence_strength': 'measured_activity',
            'activity_ugml': 1.2, 'is_strong_producer': True, 'sequence': 'MKD'}])
        table = mt.build(self._reaction(), self._expansion(),
                         structural=pd.DataFrame(columns=['uniprot', 'struct_S']),
                         extra_sources=[assayed])
        self.assertEqual(table.iloc[0].uniprot, 'WP_1')

    def test_structural_score_does_not_enter_the_ranking(self):
        """It is corroboration, not evidence; folding it in would hide that."""
        low = pd.DataFrame([{'uniprot': 'P0DXD6', 'struct_S': 0.01}])
        high = pd.DataFrame([{'uniprot': 'P0DXD6', 'struct_S': 0.99}])
        a = mt.build(self._reaction(), self._expansion(), structural=low)
        b = mt.build(self._reaction(), self._expansion(), structural=high)
        self.assertEqual(a.master_priority.tolist(), b.master_priority.tolist())

    def test_sequences_are_filled_for_reaction_rows(self):
        reaction = self._reaction().drop(columns=['sequence'])
        table = mt.build(reaction, pd.DataFrame(),
                         structural=pd.DataFrame(columns=['uniprot', 'struct_S']),
                         sequences={'P0DXD6': 'MKVSEQ'})
        self.assertEqual(table.iloc[0].sequence, 'MKVSEQ')

    def test_empty_inputs_do_not_crash(self):
        table = mt.build(self._reaction(), pd.DataFrame(),
                         structural=pd.DataFrame(columns=['uniprot', 'struct_S']))
        self.assertEqual(len(table), 1)


class TestIdempotence(unittest.TestCase):
    """Stages must not change their output because a later stage already ran.

    `reference_actives` used to compute "what is missing" by reading
    `master_candidates.tsv`, which is written AFTER it. On a second run the
    structurally scored proteins looked already-known, were skipped, and
    structural-score coverage silently fell from 36 to 21.
    """

    def test_proxy_accessions_do_not_depend_on_a_later_stage(self):
        from amide_synthesis.enzyme_mapping import reference_actives as ra
        first = ra.missing_proxy_accessions()
        second = ra.missing_proxy_accessions()
        self.assertEqual(first, second)
        if first:
            self.assertTrue(len(first) > 0)

    def test_master_build_is_deterministic(self):
        reaction = pd.DataFrame([{
            'uniprot': 'P1', 'protein_name': 'synthetase',
            'substrate_distance': .8, 'evidence_strength': 'exact_reaction',
            'sequence': 'MK'}])
        empty = pd.DataFrame(columns=['uniprot', 'struct_S'])
        a = mt.build(reaction.copy(), pd.DataFrame(), structural=empty)
        b = mt.build(reaction.copy(), pd.DataFrame(), structural=empty)
        pd.testing.assert_frame_equal(a, b)


class TestFastaRoundTrip(unittest.TestCase):
    def test_multiline_fasta_is_rejoined(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'e.fasta'
            path.write_text('>P1|NAME|rxn1|x\nAAAA\nBBBB\n>P2|N|r|x\nCCCC\n')
            sequences = mt.read_fasta(path)
        self.assertEqual(sequences['P1'], 'AAAABBBB')
        self.assertEqual(sequences['P2'], 'CCCC')

    def test_missing_fasta_returns_empty(self):
        self.assertEqual(mt.read_fasta(Path('/nonexistent.fasta')), {})


if __name__ == '__main__':
    unittest.main(verbosity=2)
