"""Offline controls for enzyme lookup and evidence selection."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd

from amide_synthesis.enzyme_resolution import (
    rank_entries, resolve_reaction, select_reactions, product_names,
)


def protein(accession, route='EC:6.3.2.1', *, prokaryote=True,
            reviewed='reviewed', existence='Evidence at protein level'):
    return {'Entry': accession, '_route': route, 'Reviewed': reviewed,
            'Taxonomic lineage': 'Bacteria' if prokaryote else 'Eukaryota',
            'Protein existence': existence}


class SelectionTests(unittest.TestCase):
    def test_exact_unreviewed_beats_reviewed_ec_and_duplicates_do_not_take_slots(self):
        rows = [protein('A'), protein('A', 'Rhea[ModelSEED]:1', reviewed='unreviewed'),
                protein('B'), protein('C', 'Name:siderophore')]
        result = rank_entries(rows, limit=3)
        self.assertEqual([r['Entry'] for r in result], ['A', 'B', 'C'])
        self.assertEqual(result[0]['_route'], 'Rhea[ModelSEED]:1')

    def test_taxonomy_cap_when_prokaryotes_fill_slots(self):
        rows = [protein(f'E{i}', 'Rhea:1', prokaryote=False) for i in range(8)]
        rows += [protein(f'B{i}') for i in range(8)]
        result = rank_entries(rows)
        self.assertEqual(len(result), 8)
        self.assertEqual(sum(r['Taxonomic lineage'] == 'Eukaryota' for r in result), 3)

    def test_eukaryotes_backfill_if_only_two_prokaryotes_exist(self):
        rows = [protein(f'E{i}', prokaryote=False) for i in range(8)]
        rows += [protein(f'B{i}') for i in range(2)]
        result = rank_entries(rows)
        self.assertEqual(len(result), 8)
        self.assertEqual(sum(r['Taxonomic lineage'] == 'Bacteria' for r in result), 2)

    def test_existence_and_accession_break_ties(self):
        rows = [protein('A', reviewed='unreviewed', existence='Predicted'),
                protein('C', reviewed='unreviewed'), protein('B', reviewed='unreviewed')]
        self.assertEqual([r['Entry'] for r in rank_entries(rows)], ['B', 'C', 'A'])
        self.assertEqual(rank_entries(rows), rank_entries(list(reversed(rows))))

    def test_zero_slots(self):
        self.assertEqual(rank_entries([protein('A')], limit=0), [])

    def test_ood_floor_and_best_framing(self):
        hits = pd.DataFrame({'rxn_id': ['a', 'a', 'b', 'c', 'd'],
                             'cosine_similarity': [.4, .8, .5, .49, float('nan')]})
        result = select_reactions(hits, .5)
        self.assertEqual(result.rxn_id.tolist(), ['a', 'b'])
        self.assertEqual(result.cosine_similarity.tolist(), [.8, .5])


class ResolutionTests(unittest.TestCase):
    def setUp(self):
        self.hit = SimpleNamespace(rxn_id='rxn1', ec='6.3.2.1',
                                   definition='(1) ATP[0] => (1) pre-alcaligin[0]')

    def test_both_structured_routes_run_and_client_rows_are_not_mutated(self):
        entry = protein('A')
        entry.pop('_route')
        client = Mock(side_effect=[([entry], False, False), ([entry], False, False)])
        result = resolve_reaction(self.hit, {'rxn1': {'Rhea': {'1'}}},
                                  {'rxn1': ['2']}, search_client=client)
        self.assertEqual(client.call_args_list[0].args, ('rhea:1 OR rhea:2',))
        self.assertEqual(client.call_args_list[1].args, ('ec:6.3.2.1',))
        self.assertEqual(len(result.entries), 1)
        self.assertEqual(result.entries[0]['_route'], 'Rhea[ModelSEED+MetaNetX]:1,2')
        self.assertNotIn('_route', entry)

    def test_name_fallback_also_runs_after_successful_empty_structured_lookup(self):
        client = Mock(side_effect=[([], False, False), ([], False, False),
                                   ([protein('AlcC')], False, False)])
        result = resolve_reaction(self.hit, {}, {}, search_client=client)
        self.assertEqual(result.entries[0]['_route'], 'Name:alcaligin')
        self.assertEqual(client.call_args_list[-1].args,
                         ('protein_name:"alcaligin" AND (synthase OR synthetase OR ligase)',))
        self.assertEqual(client.call_args_list[-1].kwargs, {'size': 25})

    def test_truncation_and_partial_failure_are_retained(self):
        client = Mock(side_effect=[([protein('A')], False, True), ([], True, False)])
        result = resolve_reaction(self.hit, {'rxn1': {'Rhea': {'1'}}}, {}, search_client=client)
        self.assertEqual(result.status, 'partial_lookup')
        self.assertEqual(result.truncated, ('Rhea',))
        self.assertTrue(result.failed)

    def test_failed_lookup_differs_from_empty_lookup(self):
        self.hit.definition = ''
        failed = resolve_reaction(self.hit, {}, {}, search_client=Mock(return_value=([], True, False)))
        empty = resolve_reaction(self.hit, {}, {}, search_client=Mock(return_value=([], False, False)))
        self.assertEqual(failed.status, 'lookup_failed')
        self.assertEqual(empty.status, 'no_hits')

    def test_no_identifier_makes_no_request(self):
        self.hit.ec, self.hit.definition = 'nan', float('nan')
        client = Mock()
        result = resolve_reaction(self.hit, {}, {}, search_client=client)
        client.assert_not_called()
        self.assertEqual(result.status, 'no_identifier')
        self.assertEqual(product_names(float('nan')), [])
