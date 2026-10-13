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

from amide_synthesis import dossier as dos   # noqa: E402


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
        from amide_synthesis import converged as cv
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
