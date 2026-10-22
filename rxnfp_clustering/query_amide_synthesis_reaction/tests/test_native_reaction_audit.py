"""Protect identity validation from accession and namespace collisions."""
import importlib.util
import json
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'audit_native_reactions.py'
SPEC = importlib.util.spec_from_file_location('native_reaction_audit', SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class NativeReactionAuditTests(unittest.TestCase):
    def test_cross_reference_namespace_and_whole_identifier(self):
        self.assertTrue(audit.has_crossref('Rhea: 44340; 44341|KEGG: R00001', 'Rhea:44340'))
        self.assertFalse(audit.has_crossref('Other: 44340|Rhea: 144340', 'Rhea:44340'))

    def test_fasta_accession_and_single_record(self):
        self.assertEqual(audit.fasta_sequence('>gb|ADN39487.1| example\nMKT\nAA', 'ADN39487.1'), 'MKTAA')
        for text in ['>ADN39487.10\nMKT', '>ADN39487.1\nMKT\n>OTHER\nAAA', '<html>error', '>ADN39487.1\n']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                audit.fasta_sequence(text, 'ADN39487.1')

    def test_uniprot_wrong_or_multiple_records_rejected(self):
        entry = {'primaryAccession': 'E2JA32', 'sequence': {'value': 'MKT'}}
        self.assertEqual(audit.uniprot_entry(json.dumps({'results': [entry]}), 'E2JA32'), entry)
        with self.assertRaises(ValueError):
            audit.uniprot_entry(json.dumps(entry), 'WRONG')
        with self.assertRaises(ValueError):
            audit.uniprot_entry(json.dumps({'results': [entry, entry]}), 'E2JA32')


if __name__ == '__main__':
    unittest.main()
