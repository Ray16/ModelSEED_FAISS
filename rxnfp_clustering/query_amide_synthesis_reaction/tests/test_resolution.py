"""Offline controls for enzyme lookup and evidence selection."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd

from amide_synthesis.enzyme_mapping.enzyme_resolution import enzyme_like_name, evidence_strength, rank_entries, resolve_reaction, select_reactions, product_names


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

    def test_best_framing_per_reaction_is_kept(self):
        hits = pd.DataFrame({'rxn_id': ['a', 'a', 'b', 'c', 'd'],
                             'cosine_similarity': [.4, .8, .5, .49, float('nan')]})
        result = select_reactions(hits, .5)
        self.assertEqual(result.rxn_id.tolist(), ['a', 'b', 'c', 'd'])
        self.assertEqual(result.cosine_similarity.tolist()[:3], [.8, .5, .49])

    def test_ood_floor_labels_rather_than_refuses(self):
        """Admission is decided once, in stage 03, not re-litigated here.

        Refusing a reaction that stage 03 already admitted - against a
        DIFFERENT threshold - cost real enzymes: rxn14123 (EC 2.3.2.2) and
        rxn19001 (EC 6.3.1.10) sit at cosine 0.499 against a floor of 0.5041,
        and their ECs return 100 and 26 UniProt entries respectively.
        """
        hits = pd.DataFrame({'rxn_id': ['a', 'b', 'c'],
                             'cosine_similarity': [.8, .5, .49]})
        result = select_reactions(hits, .5)
        self.assertEqual(len(result), 3, 'nothing is dropped')
        self.assertEqual(result.set_index('rxn_id').below_ood_floor.tolist(),
                         [False, False, True])

    def test_substrate_distance_clears_the_ood_label(self):
        hits = pd.DataFrame({'rxn_id': ['c'], 'cosine_similarity': [.1],
                             'substrate_distance': [.9]})
        result = select_reactions(hits, .5)
        self.assertFalse(result.below_ood_floor.iloc[0])


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
        # Respond by query content, not call order: the number of queries is
        # a property under test elsewhere and must not break this fixture.
        def client_fn(query, **kwargs):
            if query.startswith('protein_name:"alcaligin"'):
                return [protein('AlcC')], False, False
            return [], False, False

        client = Mock(side_effect=client_fn)
        result = resolve_reaction(self.hit, {}, {}, search_client=client)
        self.assertEqual(result.entries[0]['_route'], 'Name:alcaligin')
        queries = [c.args[0] for c in client.call_args_list]
        self.assertIn('protein_name:"alcaligin" AND (synthase OR synthetase OR ligase)',
                      queries)
        self.assertEqual(client.call_args_list[-1].kwargs, {'size': 25})

    def test_name_route_queries_the_catalytic_activity_field_too(self):
        """A product name belongs in the activity line, not the protein name.

        An enzyme that makes putrebactin may simply be called "PubC"; the
        product appears in its catalytic-activity statement. Searching only
        protein_name is the wrong field for the 79 unresolved reactions that
        have nothing but a MetaCyc identifier.
        """
        client = Mock(return_value=([], False, False))
        resolve_reaction(self.hit, {}, {}, search_client=client)
        queries = [c.args[0] for c in client.call_args_list]
        self.assertIn(
            'cc_catalytic_activity:"alcaligin" AND (synthase OR synthetase OR ligase)',
            queries)

    def test_name_route_does_not_stop_at_the_first_hit(self):
        """Later product names were being discarded for no reason."""
        self.hit.definition = '(1) ATP[0] => (1) pre-alcaligin[0] + (1) bisucaberin[0]'
        client = Mock(return_value=([], False, False))
        resolve_reaction(self.hit, {}, {}, search_client=client)
        queries = ' '.join(c.args[0] for c in client.call_args_list)
        self.assertIn('bisucaberin', queries)
        self.assertIn('alcaligin', queries)

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


class ForeignPivotTests(unittest.TestCase):
    """Rhea reachable through a sibling identifier, not just the SEED id.

    MetaNetX may not register a reaction under `seed.reaction:` while knowing
    it as MetaCyc or KEGG. Measured: 14 of the 77 unresolved MetaCyc-bearing
    reactions gain a Rhea this way, including rxn19370, which looked
    undeposited under the SEED pivot alone.
    """

    def setUp(self):
        self.hit = SimpleNamespace(rxn_id='rxn1', ec='nan', definition='')

    def test_metacyc_pivot_supplies_rhea_when_seed_pivot_is_empty(self):
        client = Mock(return_value=([protein('A')], False, False))
        resolve_reaction(self.hit,
                         {'rxn1': {'MetaCyc': {'RXN-10996'}}}, {},
                         search_client=client,
                         foreign={'MetaCyc': {'RXN-10996': ['123']}})
        self.assertEqual(client.call_args_list[0].args, ('rhea:123',))

    def test_route_label_records_the_pivot_used(self):
        client = Mock(return_value=([protein('A')], False, False))
        result = resolve_reaction(self.hit,
                                  {'rxn1': {'MetaCyc': {'RXN-1'}}}, {},
                                  search_client=client,
                                  foreign={'MetaCyc': {'RXN-1': ['9']}})
        self.assertIn('MetaNetX-xref', result.entries[0]['_route'])

    def test_absent_foreign_map_is_not_fatal(self):
        client = Mock(return_value=([], False, False))
        result = resolve_reaction(self.hit, {}, {}, search_client=client)
        self.assertEqual(result.status, 'no_identifier')

    def test_pivots_are_unioned_not_preferred(self):
        client = Mock(return_value=([protein('A')], False, False))
        resolve_reaction(self.hit,
                         {'rxn1': {'Rhea': {'1'}, 'KEGG': {'R001'}}},
                         {'rxn1': ['2']},
                         search_client=client,
                         foreign={'KEGG': {'R001': ['3']}})
        self.assertEqual(client.call_args_list[0].args,
                         ('rhea:1 OR rhea:2 OR rhea:3',))


class ReactionNameRouteTests(unittest.TestCase):
    """ModelSEED's reaction name is often the enzyme's name outright.

    It is a different string from any product in the definition, so the
    product-name route never tries it. For the 82 reactions with no
    identifier at all it is the only handle left, and it recovered the
    L-amino acid ligases - an ATP-grasp family forming amide bonds between
    two amino acids that no other route reached.
    """

    def test_enzyme_like_name_accepts_a_real_enzyme_name(self):
        self.assertEqual(enzyme_like_name('L-amino acid ligase'),
                         'L-amino acid ligase')

    def test_trailing_parenthetical_is_stripped(self):
        self.assertEqual(
            enzyme_like_name('UDP-MurNAc:L-lysine ligase (ADP-forming)'),
            'UDP-MurNAc:L-lysine ligase')

    def test_bare_class_word_is_refused(self):
        """"ligase" alone matches tens of thousands of unrelated proteins."""
        for name in ('ligase', 'synthase', 'Synthetase'):
            self.assertIsNone(enzyme_like_name(name))

    def test_substrate_shaped_name_is_refused(self):
        self.assertIsNone(enzyme_like_name('ATP + L-glutamate + NH3'))
        self.assertIsNone(enzyme_like_name(''))
        self.assertIsNone(enzyme_like_name(None))

    def test_route_is_labelled_as_the_weakest_evidence(self):
        self.assertEqual(evidence_strength('ReactionName:L-amino acid ligase'),
                         'reaction_name_match')

    def test_route_only_fires_when_everything_else_is_empty(self):
        hit = SimpleNamespace(rxn_id='r', ec='6.3.2.1', definition='',
                              reaction_name='L-amino acid ligase')
        client = Mock(return_value=([protein('A')], False, False))
        resolve_reaction(hit, {}, {}, search_client=client)
        queries = [c.args[0] for c in client.call_args_list]
        self.assertEqual(queries, ['ec:6.3.2.1'],
                         'a successful EC lookup must not trigger the fallback')
