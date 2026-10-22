"""Controls for the per-candidate dossier.

This is the artifact a person reads and acts on, so the failures that matter
are an inferred value presented as measured, and a missing field rendering as
the string "nan".
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from amide_synthesis.evaluation import dossier as dos


class TextTests(unittest.TestCase):
    def test_missing_values_never_render_as_nan(self):
        for value in (None, float('nan'), 'nan', 'NaN', '<NA>', '  '):
            self.assertEqual(dos._text(value), '')

    def test_evidence_codes_are_stripped(self):
        got = dos.clean_statement(
            'Reaction=A + B = C; {ECO:0000255|HAMAP-Rule:MF_00047};')
        self.assertNotIn('ECO:', got)
        self.assertNotIn('{', got)

    def test_whitespace_is_collapsed(self):
        self.assertEqual(dos.clean_statement('a   b\n c'), 'a b c')


class MutagenesisTests(unittest.TestCase):
    def test_position_and_effect_are_extracted(self):
        raw = ('MUTAGEN 150; /note="Loss of activity."; '
               '/evidence="ECO:0000269|PubMed:1"; '
               'MUTAGEN 201..205; /note="Reduced affinity for ATP.";')
        got = dos.mutagenesis_notes(raw)
        self.assertEqual(len(got), 2)
        self.assertTrue(got[0].startswith('150:'))
        self.assertIn('Loss of activity', got[0])
        self.assertTrue(got[1].startswith('201..205:'))

    def test_evidence_codes_do_not_leak_into_the_note(self):
        got = dos.mutagenesis_notes(
            'MUTAGEN 9; /note="Dead. {ECO:0000269|PubMed:7}";')
        self.assertNotIn('ECO:', got[0])

    def test_absent_mutagenesis_is_an_empty_list(self):
        for value in (None, '', float('nan')):
            self.assertEqual(dos.mutagenesis_notes(value), [])


class RenderTests(unittest.TestCase):
    def _row(self, **kw):
        base = {'uniprot': 'P9WJL3', 'gene': 'murE', 'protein_name': 'MurE',
                'organism': 'M. tuberculosis', 'length': 500,
                'n_discriminating_lines': 3,
                'evidence_lines': 'own_reaction_chemistry+lead_family_verdict',
                'best_substrate_distance': 0.764,
                'substrate_distance_source': 'own_curated_reaction'}
        base.update(kw)
        return pd.Series(base)

    def test_the_accession_and_name_head_the_entry(self):
        out = dos.render(self._row())
        self.assertTrue(out.startswith('## P9WJL3 — MurE'))

    def test_the_evidence_lines_are_stated(self):
        out = dos.render(self._row())
        self.assertIn('3 independent lines', out)
        self.assertIn('own_reaction_chemistry', out)

    def test_the_distance_carries_its_provenance(self):
        """A measured value and an inherited one must not read alike."""
        out = dos.render(self._row())
        self.assertIn('own_curated_reaction', out)

    def test_absent_fields_are_omitted_not_blank(self):
        out = dos.render(self._row())
        for noise in ('nan', 'None', 'K_M', 'pLDDT'):
            self.assertNotIn(noise, out)

    def test_present_fields_appear(self):
        out = dos.render(self._row(plddt_mean=93.8, km_min_mm=0.06,
                                   best_pdb='4C5C'))
        self.assertIn('93.8', out)
        self.assertIn('0.06', out)
        self.assertIn('4C5C', out)

    def test_brenda_says_the_api_is_unavailable(self):
        out = dos.render(self._row(brenda='6.3.2.13;'))
        self.assertIn('credentials', out)

    def test_pubmed_ids_become_links(self):
        out = dos.render(self._row(pubmed='12345;67890'))
        self.assertIn('pubmed.ncbi.nlm.nih.gov/12345', out)

    def test_a_row_with_only_an_accession_still_renders(self):
        out = dos.render(pd.Series({'uniprot': 'P1'}))
        self.assertIn('P1', out)

class DiamineAcceptorTests(unittest.TestCase):
    """The acceptor-side question: what free amine does it actually take?

    No enzyme in UniProt is annotated to use hexane-1,6-diamine - 0 of 54,199
    candidates name it, which is why this project exists. Only 8 name
    cadaverine, the C5 diamine one methylene short of it, so the nearest
    native acceptor is among the most informative facts about a candidate.
    """

    def _row(self, text):
        return pd.Series({'uniprot': 'P1', 'catalytic_activity': text,
                          'function': ''})

    def test_cadaverine_is_flagged_as_one_methylene_short(self):
        got = dos.native_diamine_acceptors(
            self._row('Reaction=cadaverine + L-glutamate + ATP = ...'))
        self.assertEqual(got[0][0], 'cadaverine')
        self.assertIn('ONE methylene', got[0][1])

    def test_several_acceptors_are_all_reported(self):
        got = dos.native_diamine_acceptors(self._row(
            'Substrates include putrescine, cadaverine, spermidine.'))
        self.assertEqual({n for n, _ in got},
                         {'putrescine', 'cadaverine', 'spermidine'})

    def test_the_search_is_case_insensitive(self):
        self.assertTrue(dos.native_diamine_acceptors(
            self._row('Uses CADAVERINE as acceptor')))

    def test_an_unrelated_substrate_is_not_flagged(self):
        self.assertEqual(dos.native_diamine_acceptors(
            self._row('Reaction=L-glutamate + ammonia')), [])

    def test_both_spellings_of_hmda_collapse_to_one_note(self):
        got = dos.native_diamine_acceptors(
            self._row('hexane-1,6-diamine and 1,6-diaminohexane'))
        self.assertEqual(len(got), 1)

    def test_it_appears_in_the_rendered_dossier(self):
        row = pd.Series({'uniprot': 'Q9RDS6', 'n_discriminating_lines': 3,
                         'evidence_lines': 'own_reaction_chemistry',
                         'best_substrate_distance': 0.873,
                         'catalytic_activity': 'cadaverine + L-glutamate',
                         'function': ''})
        self.assertIn('cadaverine', dos.render(row))


if __name__ == '__main__':
    unittest.main()


class PocketCapacityTests(unittest.TestCase):
    """Room for the monomer, measured from ligands rather than docking.

    HMDA spans 6 between its amines and adipate 4 between its acids, so a
    pocket seen holding spermidine (8) or meso-diaminopimelate (5 acid) has
    room on that side. Capacity is necessary and nowhere near sufficient.
    """

    STRUCTURES = {
        '1': 'NCCCNCCCCN',                 # spermidine, amine span 8
        '2': 'NCCCCCN',                    # cadaverine, amine span 5
        '3': 'OC(=O)C(N)CCCC(N)C(=O)O',    # meso-DAP, acid span 5
        '4': '[Mg+2]',                     # a metal
    }

    def _row(self, text):
        return pd.Series({'uniprot': 'P1', 'binding_site': text,
                          'catalytic_activity': ''})

    def test_spermidine_clears_the_hmda_span(self):
        got = dos.pocket_capacity(self._row('ChEBI:CHEBI:1'), self.STRUCTURES)
        self.assertEqual(got['amine_span'], 8)
        self.assertTrue(got['fits_hmda'])

    def test_cadaverine_does_not_clear_it(self):
        """C5 is one methylene short; capacity must not round up."""
        got = dos.pocket_capacity(self._row('ChEBI:CHEBI:2'), self.STRUCTURES)
        self.assertEqual(got['amine_span'], 5)
        self.assertFalse(got['fits_hmda'])

    def test_meso_dap_clears_the_adipate_span(self):
        got = dos.pocket_capacity(self._row('ChEBI:CHEBI:3'), self.STRUCTURES)
        self.assertTrue(got['fits_adipate'])

    def test_the_widest_ligand_wins(self):
        got = dos.pocket_capacity(self._row('ChEBI:CHEBI:1 ChEBI:CHEBI:2'),
                                  self.STRUCTURES)
        self.assertEqual(got['amine_span'], 8)
        self.assertEqual(got['widest_ligand'], '1')

    def test_a_metal_contributes_nothing(self):
        got = dos.pocket_capacity(self._row('ChEBI:CHEBI:4'), self.STRUCTURES)
        self.assertEqual(got['amine_span'], 0)
        self.assertFalse(got['fits_hmda'])

    def test_catalytic_activity_counts_too(self):
        """GlnA2's acceptor is named there, not in binding_site."""
        row = pd.Series({'uniprot': 'P1', 'binding_site': '',
                         'catalytic_activity': 'spermidine ChEBI:CHEBI:1'})
        self.assertTrue(dos.pocket_capacity(row, self.STRUCTURES)['fits_hmda'])

    def test_the_bare_chebi_form_is_also_matched(self):
        """UniProt writes ChEBI:CHEBI:30616; a bare form must not slip past."""
        got = dos.pocket_capacity(self._row('ChEBI:1'), self.STRUCTURES)
        self.assertEqual(got['amine_span'], 8)

    def test_an_unresolved_chebi_is_skipped(self):
        got = dos.pocket_capacity(self._row('ChEBI:CHEBI:9999'), self.STRUCTURES)
        self.assertEqual(got['amine_span'], 0)

    def test_capacity_is_not_a_convergence_axis(self):
        """It derives from the same ligand data as the ligand axis."""
        from amide_synthesis.candidate_selection import converged as cv
        found = cv.axes(pd.DataFrame([{'uniprot': 'P1'}]), set())
        for name in found.columns:
            self.assertNotIn('span', name)
            self.assertNotIn('capacity', name)


class CoverageTests(unittest.TestCase):
    """Every converged candidate is documented, not an arbitrary top-N.

    The cap was 20 against 28 converged, dropping eight - including P0AES0,
    whose crystal holds a three-amine ligand, and sbnC. Both were being
    recommended in prose while missing from the document.
    """

    def test_the_default_imposes_no_limit(self):
        self.assertEqual(dos.MAX_DOSSIERS, 0)

    def test_a_limit_is_still_honoured_when_set(self):
        """Kept configurable; it just must not silently default to one."""
        source = Path(dos.__file__).read_text()
        self.assertIn('if MAX_DOSSIERS:', source)


class AccessionPatternTests(unittest.TestCase):
    """O, P and Q accessions must be queried for the extra fields.

    A hand-rolled pattern here began `[A-NR-Z0-9]`, which excludes them, so
    10 of 28 converged candidates were queried and 18 - including P07862,
    P0AES0, Q9RDS6 and O60993 - silently had no mutagenesis, literature or
    BRENDA data fetched.
    """

    def _match(self, accession):
        import re
        from amide_synthesis.candidate_selection.curated_properties import ACCESSION
        return bool(re.match(ACCESSION, accession))

    def test_o_p_and_q_accessions_match(self):
        for accession in ('O60993', 'P07862', 'P0AES0', 'Q9RDS6', 'Q2FWC5'):
            self.assertTrue(self._match(accession), accession)

    def test_other_prefixes_still_match(self):
        for accession in ('A0A095T5Z8', 'A1BJY3', 'B6EP94'):
            self.assertTrue(self._match(accession), accession)

    def test_the_dossier_imports_the_shared_pattern(self):
        source = Path(dos.__file__).read_text()
        from amide_synthesis.candidate_selection.curated_properties import ACCESSION
        self.assertIs(dos.ACCESSION, ACCESSION)
        self.assertNotIn('[A-NR-Z0-9][0-9A-Z]', source)


class AssayPickCoverageTests(unittest.TestCase):
    """Every protein the assay round recommends must have a dossier.

    ASSAY_ROUND.md recommends four ANL candidates chosen for their similarity
    to a measured producer. They reach no evidence line in this pipeline, so
    nothing else would have documented them - the sixth place in this project
    where a recommendation would have sat in one file and the documentation
    in none.
    """

    def test_the_anl_picks_are_configured(self):
        self.assertGreaterEqual(dos.ANL_PICKS, 4)
        self.assertEqual(dos.ANL_RANKING, 'anl_similarity_to_actives.tsv')

    def test_an_anl_pick_renders_its_own_reason(self):
        row = pd.Series({
            'uniprot': 'A0A075UYU2',
            'selected_by': 'anl_closest_to_a_measured_active',
            'evidence_lines': '30.6% identity to AS9, a measured producer',
            'n_discriminating_lines': 0})
        out = dos.render(row)
        self.assertIn('OTHER hypothesis', out)
        self.assertIn('MEASURED activity', out)
        self.assertIn('30.6%', out)

    def test_it_does_not_claim_independent_lines(self):
        """It has none; saying otherwise would overstate the pipeline."""
        row = pd.Series({
            'uniprot': 'A0A075UYU2',
            'selected_by': 'anl_closest_to_a_measured_active',
            'evidence_lines': '30.6% identity to AS9, a measured producer',
            'n_discriminating_lines': 0})
        self.assertNotIn('independent lines:', dos.render(row))


class TestKineticsNote(unittest.TestCase):
    """Dossiers used to say kinetics needed credentials. They do not."""

    def digest(self, rows=None):
        return pd.DataFrame(rows or [
            {'ec': '6.3.1.8', 'parameter': 'Km', 'n': 3,
             'median_si': 1.175e-3, 'unit_si': 'M'},
        ])

    def test_molar_is_rendered_in_micromolar(self):
        note = dos.kinetics_note('6.3.1.8', self.digest())
        self.assertIn('1,175 uM', note)
        self.assertIn('n=3', note)

    def test_non_molar_units_are_left_alone(self):
        digest = self.digest([{'ec': '6.3.1.8', 'parameter': 'kcat', 'n': 1,
                               'median_si': 4.2, 'unit_si': 's^(-1)'}])
        self.assertIn('4.2 s^(-1)', dos.kinetics_note('6.3.1.8', digest))

    def test_bifunctional_protein_labels_each_ec(self):
        """Two ECs each contribute a Km; unlabelled they read as one."""
        digest = self.digest([
            {'ec': '3.5.1.78', 'parameter': 'Km', 'n': 1,
             'median_si': 5.6e-3, 'unit_si': 'M'},
            {'ec': '6.3.1.8', 'parameter': 'Km', 'n': 3,
             'median_si': 1.175e-3, 'unit_si': 'M'},
        ])
        note = dos.kinetics_note('3.5.1.78; 6.3.1.8', digest)
        self.assertIn('EC 3.5.1.78 Km', note)
        self.assertIn('EC 6.3.1.8 Km', note)

    def test_single_ec_is_not_labelled(self):
        self.assertNotIn('EC 6.3.1.8 Km',
                         dos.kinetics_note('6.3.1.8', self.digest()))

    def test_absent_ec_gives_nothing(self):
        self.assertEqual(dos.kinetics_note('1.1.1.1', self.digest()), '')

    def test_missing_inputs_do_not_raise(self):
        self.assertEqual(dos.kinetics_note(None, self.digest()), '')
        self.assertEqual(dos.kinetics_note('6.3.1.8', None), '')
        self.assertEqual(
            dos.kinetics_note('6.3.1.8', pd.DataFrame()), '')

    def test_unparseable_median_is_skipped_not_printed(self):
        digest = self.digest([{'ec': '6.3.1.8', 'parameter': 'Km', 'n': 1,
                               'median_si': None, 'unit_si': 'M'}])
        self.assertEqual(dos.kinetics_note('6.3.1.8', digest), '')
