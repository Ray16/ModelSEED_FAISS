#!/usr/bin/env python3
"""Export GO cross-references for the exact five literature UniProt accessions.

By default retrieve current UniProtKB JSON. --evidence-dir reproduces a saved
protein snapshot without network access. Each run preserves raw responses,
GO evidence codes, empty annotation results and input/output hashes.
"""
import argparse
import csv
from datetime import datetime, timezone
import html
import json
from pathlib import Path

from audit_native_reactions import get_url, sha256, uniprot_entry

PROJECT = Path(__file__).resolve().parent
ASPECTS = {'F': 'molecular_function', 'P': 'biological_process', 'C': 'cellular_component'}
NOTES = {
    'SfaB': 'Electronic acid-thiol ligase annotation is broad and does not specify native acid adenylation or demonstrate amidation.',
    'McbA': 'Electronic carbon-sulfur bond ligase annotation does not describe the experimentally characterized carbon-nitrogen amide formation. Retained as reported by UniProt.',
    'SuCphA1': 'Accession BAA17890.1 / P73833 is PCC6803/Kazusa; the nylon paper calls it SuCphA1. The structural study uses UTEX2470.',
}


def extract(entry, enzyme):
    rows = []
    for crossref in entry.get('uniProtKBCrossReferences', []):
        if crossref['database'] != 'GO':
            continue
        properties = {p['key']: p['value'] for p in crossref.get('properties', [])}
        raw_term = properties.get('GoTerm', '')
        aspect, separator, term = raw_term.partition(':')
        evidence = properties.get('GoEvidenceType', '')
        code, _, provider = evidence.partition(':')
        rows.append(dict(enzyme=enzyme, uniprot=entry['primaryAccession'],
                         go_id=crossref['id'], term=term if separator else raw_term,
                         aspect=ASPECTS.get(aspect, aspect), evidence_code=code,
                         evidence_provider=provider,
                         go_url=f'https://www.ebi.ac.uk/QuickGO/term/{crossref["id"]}',
                         uniprot_url=f'https://www.uniprot.org/uniprotkb/{entry["primaryAccession"]}/entry'))
    return rows


def write_tsv(path, rows, fields):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-dir', type=Path, help='Read saved *_uniprot.json files, without network access')
    parser.add_argument('--output-root', type=Path, default=PROJECT / 'results/native_enzyme_go')
    args = parser.parse_args()
    now = datetime.now(timezone.utc)
    out = args.output_root / now.strftime('%Y%m%dT%H%M%S%fZ')
    raw_dir = out / 'evidence'
    raw_dir.mkdir(parents=True, exist_ok=False)
    reference_path = PROJECT / 'data/native_reaction_references.json'
    reference = json.loads(reference_path.read_text())
    rows, summaries, metadata, failures = [], [], [], []
    for ref in reference['enzymes']:
        name, accession = ref['enzyme'], ref['uniprot']
        url = f'https://rest.uniprot.org/uniprotkb/{accession}.json'
        print(f'GO: {name} / {accession}', flush=True)
        error = ''
        subset = []
        try:
            raw = ((args.evidence_dir / f'{name}_uniprot.json').read_bytes()
                   if args.evidence_dir else get_url(url))
            path = raw_dir / f'{name}_uniprot.json'
            path.write_bytes(raw)
            metadata.append({'enzyme': name, 'accession': accession, 'source_url': url,
                             'sha256': sha256(path), 'file': str(path.relative_to(out)),
                             'retrieved_or_copied_utc': datetime.now(timezone.utc).isoformat()})
            entry = uniprot_entry(raw, accession)
            subset = extract(entry, name)
            rows.extend(subset)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            failures.append(f'{name}: {error}')
        summary = dict(enzyme='CphA1 (paper: SuCphA1)' if name == 'SuCphA1' else name,
                       genbank=ref['genbank'], uniprot=accession,
                       go_count=len(subset) if not error else '',
                       retrieval_status='failed' if error else 'retrieved' if args.evidence_dir is None else 'saved_snapshot',
                       molecular_function='', biological_process='', cellular_component='',
                       notes=NOTES.get(name, ''), error=error)
        for aspect in ASPECTS.values():
            summary[aspect] = '; '.join(f'{r["go_id"]} {r["term"]} [{r["evidence_code"]}]'
                                       for r in subset if r['aspect'] == aspect)
        if not subset and not error:
            summary['notes'] = 'No GO cross-references listed in this exact UniProtKB entry; this does not imply absence of a biological function.'
        summaries.append(summary)
    fields = ['enzyme', 'uniprot', 'go_id', 'term', 'aspect', 'evidence_code', 'evidence_provider', 'go_url', 'uniprot_url']
    write_tsv(out / 'go_annotations.tsv', rows, fields)
    write_tsv(out / 'enzyme_go_summary.tsv', summaries, list(summaries[0]))
    (out / 'go_annotations.json').write_text(json.dumps({'summaries': summaries, 'annotations': rows}, indent=2) + '\n')
    esc = lambda x: html.escape(str(x))
    columns = ['enzyme', 'uniprot', 'molecular_function', 'biological_process', 'cellular_component', 'notes']
    table = '<tr>' + ''.join(f'<th>{esc(c.replace("_", " "))}</th>' for c in columns) + '</tr>'
    for row in summaries:
        table += '<tr>' + ''.join(f'<td>{esc(row[c]) if row[c] else "—"}</td>' for c in columns) + '</tr>'
    (out / 'report.html').write_text('<!doctype html><meta charset="utf-8"><title>GO annotations</title>'
        '<style>body{font:16px/1.5 system-ui;margin:25px}table{border-collapse:collapse}'
        'th,td{padding:10px;border:1px solid #ddd;text-align:left;vertical-align:top}</style>'
        '<h1>GO annotations for the five reference proteins</h1><p>Exact-accession UniProtKB cross-references. '
        'IEA: electronic annotation; IBA: inference from biological ancestry. These are annotation evidence codes, '
        'not experimental validation of nylon synthesis. GO cross-references are not an exhaustive GO association-file export.</p>'
        f'<p>Source mode: {"saved snapshot" if args.evidence_dir else "live UniProtKB retrieval"}; {esc(now.isoformat())}.</p>'
        '<table>' + table + '</table><p><a href="go_annotations.tsv">All terms and evidence providers (TSV)</a></p>')
    manifest = dict(created_utc=now.isoformat(),
                    mode='saved_snapshot' if args.evidence_dir else 'live_uniprot_retrieval',
                    source_snapshot=str(args.evidence_dir.resolve()) if args.evidence_dir else None,
                    scope='UniProtKB GO cross-references for exact accessions, without inferred or homolog-transferred additions',
                    source_records=metadata, failures=failures, success=not failures,
                    input_sha256={str(Path(__file__).resolve()): sha256(__file__),
                                  str(PROJECT / 'audit_native_reactions.py'): sha256(PROJECT / 'audit_native_reactions.py'),
                                  str(reference_path): sha256(reference_path)},
                    output_sha256={p.name: sha256(p) for p in out.iterdir() if p.is_file()})
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(out.resolve())
    print(f'{len(rows)} GO cross-references across {len(summaries)} proteins; {len(failures)} retrieval failures')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
