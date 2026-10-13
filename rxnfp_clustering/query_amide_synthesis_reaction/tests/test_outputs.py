"""Regression controls for stale outputs and empty discovery runs."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from amide_synthesis import candidate_table, fetch_enzymes, native_substrates


def screened_hits():
    return pd.DataFrame([{
        'rxn_id': 'rxn1', 'query': 'atp_amp_ligase', 'reaction_name': 'amide ligase',
        'cosine_similarity': .8, 'acylates_free_diamine': True,
        'ec': '6.3.2.1', 'definition': '', 'n_free_amines_on_donor': 2,
        'external_ids': 'Rhea:1', 'substrate_distance': .6,
        'found_via': 'substrate_distance', 'self_condensation': False,
    }])


def enzyme(accession, reaction, route):
    row = dict.fromkeys(fetch_enzymes.ENZYME_COLUMNS, '')
    row.update(uniprot=accession, rxn_id=reaction, resolved_via=route,
               evidence_strength='exact_reaction' if route.startswith('Rhea') else 'activity_ec',
               cosine_similarity=.8, acylates_free_diamine=True,
               is_prokaryote=True, has_pdb=False, family='IucA/IucC', family_all='IucA/IucC')
    return row


class OutputTests(unittest.TestCase):
    def test_structured_hit_exports_evidence_annotations_and_wrapped_fasta(self):
        with tempfile.TemporaryDirectory() as temp:
            results = Path(temp)
            screened_hits().to_csv(results / 'amide_forming_neighbours.tsv', sep='\t', index=False)
            entry = {'Entry': 'P12345', 'Entry Name': 'TEST_BACTERIA',
                     'Taxonomic lineage': 'Bacteria', 'Organism': 'Test organism',
                     'Reviewed': 'reviewed', 'Sequence': 'A' * 62,
                     'Pfam': 'PF04183;', 'Protein existence': 'Evidence at protein level'}
            with patch.object(fetch_enzymes, 'RES', results), \
                 patch.object(fetch_enzymes, 'ood_floor', return_value=.5), \
                 patch.object(fetch_enzymes, 'load_aliases', return_value={'rxn1': {'Rhea': {'1'}}}), \
                 patch.object(fetch_enzymes, 'seed_to_rhea', return_value={}), \
                 patch.object(fetch_enzymes, 'search', return_value=([entry], False, False)), \
                 patch('amide_synthesis.paths.ensure_directories'), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertIsNone(fetch_enzymes.main())
            enzymes = pd.read_csv(results / 'enzymes_uniprot.tsv', sep='\t')
            self.assertEqual(enzymes.uniprot.tolist(), ['P12345'])
            self.assertEqual(enzymes.evidence_strength.tolist(), ['exact_reaction'])
            annotations = pd.read_csv(results / 'enzyme_annotations.tsv', sep='\t')
            self.assertEqual(annotations.pfam.tolist(), ['PF04183;'])
            self.assertTrue(pd.read_csv(results / 'reactions_without_enzymes.tsv', sep='\t').empty)
            header, first, second = (results / 'enzymes.fasta').read_text().splitlines()
            self.assertTrue(header.startswith('>P12345|TEST_BACTERIA|rxn1|'))
            self.assertEqual([first, second], ['A' * 60, 'AA'])

    def test_table_excludes_stale_reactions_and_retains_best_protein_evidence(self):
        enzymes = pd.DataFrame([enzyme('A', 'rxn1', 'EC:6.3.2.1'),
                                enzyme('A', 'rxn1', 'Rhea:1'),
                                enzyme('B', 'stale', 'Rhea:1')])
        with contextlib.redirect_stdout(io.StringIO()):
            result = candidate_table.build_candidate_table(enzymes, screened_hits())
        self.assertEqual(result.uniprot.tolist(), ['A'])
        self.assertEqual(result.evidence_strength.tolist(), ['exact_reaction'])
        # 2 exact_reaction + 2 free diamine + 1 prokaryote + 0 pdb
        # + 3 x substrate_distance(0.6). Cosine deliberately contributes
        # nothing: it is cofactor-dominated and ranking on it put AsbA third
        # while the independent structural proxy ranked it 32 of 36.
        self.assertEqual(result.priority.tolist(), [6.8])

    def test_cosine_does_not_influence_priority(self):
        """Guards the whole point of the substrate-distance rewrite."""
        hits_low = screened_hits()
        hits_high = screened_hits()
        hits_high.loc[0, 'cosine_similarity'] = .99
        enzymes = pd.DataFrame([enzyme('A', 'rxn1', 'Rhea:1')])
        with contextlib.redirect_stdout(io.StringIO()):
            low = candidate_table.build_candidate_table(enzymes, hits_low)
            high = candidate_table.build_candidate_table(enzymes, hits_high)
        self.assertEqual(low.priority.tolist(), high.priority.tolist())

    def test_substrate_distance_does_influence_priority(self):
        hits_near = screened_hits()
        hits_near.loc[0, 'substrate_distance'] = .9
        enzymes = pd.DataFrame([enzyme('A', 'rxn1', 'Rhea:1')])
        with contextlib.redirect_stdout(io.StringIO()):
            base = candidate_table.build_candidate_table(enzymes, screened_hits())
            near = candidate_table.build_candidate_table(enzymes, hits_near)
        self.assertGreater(near.priority.iloc[0], base.priority.iloc[0])

    def run_empty_lookup(self, failed):
        with tempfile.TemporaryDirectory() as temp:
            results = Path(temp)
            screened_hits().to_csv(results / 'amide_forming_neighbours.tsv', sep='\t', index=False)
            for name in ('enzymes_uniprot.tsv', 'enzyme_annotations.tsv', 'enzymes.fasta'):
                (results / name).write_text('stale output')
            with patch.object(fetch_enzymes, 'RES', results), \
                 patch.object(fetch_enzymes, 'ood_floor', return_value=.5), \
                 patch.object(fetch_enzymes, 'load_aliases', return_value={}), \
                 patch.object(fetch_enzymes, 'seed_to_rhea', return_value={}), \
                 patch.object(fetch_enzymes, 'search', return_value=([], failed, False)), \
                 patch('amide_synthesis.paths.ensure_directories'), \
                 contextlib.redirect_stdout(io.StringIO()):
                code = fetch_enzymes.main()
            self.assertEqual(code, int(failed))
            self.assertTrue(pd.read_csv(results / 'enzymes_uniprot.tsv', sep='\t').empty)
            self.assertTrue(pd.read_csv(results / 'enzyme_annotations.tsv', sep='\t').empty)
            self.assertEqual((results / 'enzymes.fasta').read_text(), '')
            unresolved = pd.read_csv(results / 'reactions_without_enzymes.tsv', sep='\t')
            self.assertEqual(unresolved.lookup_status.tolist(), ['lookup_failed' if failed else 'no_hits'])
            if not failed:
                with patch.object(candidate_table, 'RES', results), \
                     patch.object(candidate_table, 'ood_floor', return_value=.5), \
                     patch.object(native_substrates, 'RES', results), \
                     patch('amide_synthesis.paths.ensure_directories'), \
                     contextlib.redirect_stdout(io.StringIO()):
                    candidate_table.main()
                    native_substrates.main()
                for name in ('candidates_ranked.tsv', 'native_substrates.tsv', 'enzyme_families.tsv'):
                    self.assertTrue(pd.read_csv(results / name, sep='\t').empty, name)
                self.assertEqual((results / 'candidates_top.fasta').read_text(), '')

    def test_successful_empty_lookup_clears_stale_files_and_runs_reporting(self):
        self.run_empty_lookup(False)

    def test_outage_clears_stale_enzyme_files_and_returns_failure(self):
        self.run_empty_lookup(True)

    def test_native_statement_keeps_evidence_and_substrates(self):
        blob = ('CATALYTIC ACTIVITY: Reaction=2 monomer + 2 ATP = dimer + 2 AMP; '
                'Xref=Rhea:RHEA:79295; EC=6.3.2.64; '
                'Evidence={ECO:0000269|PubMed:18956041};')
        statement, = native_substrates.statements(blob)
        self.assertEqual(statement['pubmed'], '18956041')
        self.assertEqual(statement['rhea'], 'RHEA:79295')
        self.assertIn('ECO:0000269', statement['evidence'])
        self.assertEqual(native_substrates.substrates(statement['native_reaction']), ['monomer'])
