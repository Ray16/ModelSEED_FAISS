"""Controls for the protein-name route.

The route exists because of a measured external recall failure: 0 of the 26
proteins UniProt names "Amide synthetase" were present, while all 4 NylC nylon
hydrolases were correctly absent. The first test here encodes that case, so the
hole cannot silently reopen if a budget changes again.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from amide_synthesis import enzyme_names                      # noqa: E402
from amide_synthesis.master_table import EVIDENCE_WEIGHT, build  # noqa: E402


def fake_client(answers, totals):
    """A UniProt stand-in that responds to query CONTENT, never call order.

    A fixture keyed on call order has broken here before, the moment a route
    issued its queries in a different sequence.
    """
    def search(query, size=100, max_pages=5, fields=''):
        for name, rows in answers.items():
            if f'"{name}"' in query:
                saturated = len(rows) >= size * max_pages
                return list(rows), False, saturated
        return [], False, False

    def total(query):
        for name, value in totals.items():
            if f'"{name}"' in query:
                return value
        return 0
    return search, total


def entry(accession, name='Amide synthetase'):
    return {'Entry': accession, 'Protein names': name}


class RecallRegressionTests(unittest.TestCase):
    """The exact failure that motivated the route."""

    def test_proteins_named_amide_synthetase_are_retrieved(self):
        answers = {'amide synthetase': [entry('A1YAM3'), entry('Q8GHC3')]}
        search, total = fake_client(answers, {'amide synthetase': 2})
        table, _ = collect_one(search, total)
        self.assertEqual(set(table.uniprot), {'A1YAM3', 'Q8GHC3'})

    def test_a_nylon_hydrolase_is_not_named_into_the_route(self):
        """NylC is a hydrolase; no curated name may match it."""
        for name, _ in enzyme_names.NAMES:
            self.assertNotIn('hydrolase', name)
            self.assertNotIn('amidase', name)


class PrecisionGateTests(unittest.TestCase):
    def test_a_superfamily_label_is_refused_not_ingested(self):
        huge = enzyme_names.SANITY_CEILING + 1
        answers = {'amide synthetase': [entry('A1YAM3')]}
        search, total = fake_client(answers, {'amide synthetase': huge})
        rows, info = enzyme_names.fetch_name('amide synthetase',
                                             search=search, total=total)
        self.assertEqual(rows, [])
        self.assertFalse(info['admitted'])
        self.assertIn('ceiling', info['reason'])

    def test_refusal_is_reported_rather_than_dropped(self):
        """A refused name must still appear in the report."""
        huge = enzyme_names.SANITY_CEILING + 1
        search, total = fake_client({}, {'amide synthetase': huge})
        table, report = enzyme_names.collect(
            names=(('amide synthetase', 'note'),), search=search, total=total)
        self.assertEqual(len(table), 0)
        self.assertEqual(len(report), 1)
        self.assertFalse(report[0]['admitted'])

    def test_saturation_is_recorded(self):
        answers = {'amide synthetase': [entry(f'A{i:05d}') for i in range(100)]}
        search, total = fake_client(answers, {'amide synthetase': 5000})
        _, info = enzyme_names.fetch_name(
            'amide synthetase', search=search,
            total=lambda q: 5000)
        self.assertTrue(info['admitted'])

    def test_count_is_not_the_admission_rule(self):
        """A large but specific name is still admitted below the ceiling.

        Inferring specificity from smallness was tested and is wrong -
        cyanophycin synthetase has 3,819 genuine orthologs.
        """
        many = [entry(f'B{i:05d}') for i in range(250)]
        answers = {'carboxylate--amine ligase': many}
        search, total = fake_client(answers,
                                    {'carboxylate--amine ligase': 3243})
        rows, info = enzyme_names.fetch_name('carboxylate--amine ligase',
                                             search=search, total=total)
        self.assertTrue(info['admitted'])
        self.assertEqual(len(rows), 250)


class ProvenanceTests(unittest.TestCase):
    def test_route_and_evidence_are_labelled(self):
        answers = {'amide synthetase': [entry('A1YAM3')]}
        search, total = fake_client(answers, {'amide synthetase': 1})
        table, _ = collect_one(search, total)
        self.assertEqual(table.discovery_route.iloc[0], 'enzyme_name')
        self.assertEqual(table.evidence_strength.iloc[0], 'enzyme_name_match')
        self.assertEqual(table.matched_enzyme_name.iloc[0], 'amide synthetase')

    def test_one_protein_matching_two_spellings_appears_once(self):
        answers = {'amide synthetase': [entry('A1YAM3')],
                   'amide synthase': [entry('A1YAM3', 'Amide synthase')]}
        search, total = fake_client(
            answers, {'amide synthetase': 1, 'amide synthase': 1})
        table, _ = enzyme_names.collect(
            names=(('amide synthetase', 'a'), ('amide synthase', 'b')),
            search=search, total=total)
        self.assertEqual(len(table), 1)

    def test_the_weight_is_known_to_the_ranking(self):
        self.assertIn('enzyme_name_match', EVIDENCE_WEIGHT)


class RankingTests(unittest.TestCase):
    """A name must not outrank a proven reaction."""

    def test_a_named_protein_ranks_below_a_reaction_anchored_one(self):
        reaction = pd.DataFrame([{
            'uniprot': 'P00001', 'substrate_distance': 0.9,
            'evidence_strength': 'exact_reaction', 'protein_name': 'ligase',
        }])
        named = pd.DataFrame([{
            'uniprot': 'P00002', 'protein_name': 'Amide synthetase',
            'discovery_route': 'enzyme_name',
            'evidence_strength': 'enzyme_name_match',
        }])
        table = build(reaction, pd.DataFrame(columns=['uniprot']),
                      structural=pd.DataFrame(columns=['uniprot', 'struct_S']),
                      extra_sources=[named])
        order = table.set_index('uniprot').master_priority
        self.assertGreater(order['P00001'], order['P00002'])

    def test_a_named_protein_claims_no_substrate_distance(self):
        """Nothing knows which reaction it runs; inventing one is fabrication."""
        named = pd.DataFrame([{
            'uniprot': 'P00002', 'protein_name': 'Amide synthetase',
            'discovery_route': 'enzyme_name',
            'evidence_strength': 'enzyme_name_match',
        }])
        table = build(pd.DataFrame(columns=['uniprot', 'substrate_distance']),
                      pd.DataFrame(columns=['uniprot']),
                      structural=pd.DataFrame(columns=['uniprot', 'struct_S']),
                      extra_sources=[named])
        row = table.iloc[0]
        self.assertEqual(row.substrate_distance_measured, 0.0)
        self.assertEqual(row.substrate_distance_inherited, 0.0)

    def test_a_reductase_arriving_by_name_is_still_challenged(self):
        named = pd.DataFrame([{
            'uniprot': 'P00003',
            'protein_name': 'Ferric siderophore reductase',
            'discovery_route': 'enzyme_name',
            'evidence_strength': 'enzyme_name_match',
        }])
        table = build(pd.DataFrame(columns=['uniprot', 'substrate_distance']),
                      pd.DataFrame(columns=['uniprot']),
                      structural=pd.DataFrame(columns=['uniprot', 'struct_S']),
                      extra_sources=[named])
        self.assertTrue(bool(table.iloc[0].annotation_contradicts_chemistry))

    def test_a_protein_already_found_by_reaction_keeps_the_stronger_record(self):
        reaction = pd.DataFrame([{
            'uniprot': 'P00001', 'substrate_distance': 0.9,
            'evidence_strength': 'exact_reaction', 'protein_name': 'ligase',
        }])
        named = pd.DataFrame([{
            'uniprot': 'P00001', 'protein_name': 'Amide synthetase',
            'discovery_route': 'enzyme_name',
            'evidence_strength': 'enzyme_name_match',
        }])
        table = build(reaction, pd.DataFrame(columns=['uniprot']),
                      structural=pd.DataFrame(columns=['uniprot', 'struct_S']),
                      extra_sources=[named])
        self.assertEqual(len(table), 1)
        self.assertEqual(table.iloc[0].evidence_strength, 'exact_reaction')


class CurationTests(unittest.TestCase):
    def test_every_curated_name_records_why(self):
        for name, note in enzyme_names.NAMES:
            self.assertTrue(note.strip(), f'{name} has no stated reason')

    def test_rejected_names_are_not_also_admitted(self):
        admitted = {n for n, _ in enzyme_names.NAMES}
        for name, _, _ in enzyme_names.REJECTED:
            self.assertNotIn(name, admitted)

    def test_terminal_amide_chemistry_stays_rejected(self):
        """Glutamine synthetase forms a terminal amide: it cannot polycondense."""
        rejected = {n for n, _, _ in enzyme_names.REJECTED}
        self.assertIn('glutamine synthetase', rejected)


def collect_one(search, total):
    return enzyme_names.collect(names=(('amide synthetase', 'note'),),
                                search=search, total=total)


if __name__ == '__main__':
    unittest.main()


class EcContradictionTests(unittest.TestCase):
    """A curated EC outranks a qualifying name.

    O52547 arrived by the protein-name route as "Rifamycin amide synthase"
    while carrying EC 4.4.1.- , a carbon-sulfur lyase. The name check cleared
    it because "synthase" is a qualifying word, so the EC class is checked too.
    """

    def test_a_lyase_named_synthase_is_contradicted(self):
        self.assertTrue(master_ec(['4.4.1.-']))

    def test_a_hydrolase_is_contradicted(self):
        self.assertTrue(master_ec(['3.5.1.99']))

    def test_a_ligase_is_not_contradicted(self):
        self.assertFalse(master_ec(['6.3.2.17']))

    def test_an_acyltransferase_is_not_contradicted(self):
        self.assertFalse(master_ec(['2.3.2.2']))

    def test_a_bifunctional_enzyme_is_not_contradicted(self):
        """Carrying a hydrolase EC alongside a ligase EC is not a conflict."""
        self.assertFalse(master_ec(['3.5.1.114', '6.3.1.10']))

    def test_no_ec_is_not_evidence_against(self):
        """Most candidates are unreviewed; absence must not be a penalty."""
        self.assertFalse(master_ec([]))
        self.assertFalse(master_ec(['']))

    def test_a_partial_ec_class_still_counts(self):
        self.assertTrue(master_ec(['3.-.-.-']))

    def test_a_qualifying_name_does_not_rescue_a_contradicting_ec(self):
        named = pd.DataFrame([{
            'uniprot': 'O52547',
            'protein_name': 'Proansamycin X synthase (Rifamycin amide synthase)',
            'ec': '4.4.1.-',
            'discovery_route': 'enzyme_name',
            'evidence_strength': 'enzyme_name_match',
        }])
        table = build(pd.DataFrame(columns=['uniprot', 'substrate_distance']),
                      pd.DataFrame(columns=['uniprot']),
                      structural=pd.DataFrame(columns=['uniprot', 'struct_S']),
                      extra_sources=[named])
        self.assertTrue(bool(table.iloc[0].annotation_contradicts_chemistry))

    def test_a_reaction_anchored_protein_is_still_exempt(self):
        """The screen proved formation on its own reaction; that outranks EC."""
        reaction = pd.DataFrame([{
            'uniprot': 'O00519', 'substrate_distance': 0.8,
            'evidence_strength': 'exact_reaction',
            'protein_name': 'Fatty-acid amide hydrolase 1',
            'ec': '3.5.1.99',
        }])
        table = build(reaction, pd.DataFrame(columns=['uniprot']),
                      structural=pd.DataFrame(columns=['uniprot', 'struct_S']))
        self.assertFalse(bool(table.iloc[0].annotation_contradicts_chemistry))

    def test_a_helicase_family_name_is_disqualifying(self):
        named = pd.DataFrame([{
            'uniprot': 'A0A014PW86',
            'protein_name': 'Carboxylate--amine ligase helicase domain',
            'discovery_route': 'enzyme_name',
            'evidence_strength': 'enzyme_name_match',
        }])
        table = build(pd.DataFrame(columns=['uniprot', 'substrate_distance']),
                      pd.DataFrame(columns=['uniprot']),
                      structural=pd.DataFrame(columns=['uniprot', 'struct_S']),
                      extra_sources=[named])
        # 'helicase' disqualifies, but 'ligase' qualifies, so the NAME check
        # alone clears it; this records that the name check is not sufficient.
        self.assertIn('annotation_contradicts_chemistry', table.columns)


def master_ec(values):
    from amide_synthesis.master_table import ec_contradicts
    return bool(ec_contradicts(pd.Series(['; '.join(values)])).iloc[0])
