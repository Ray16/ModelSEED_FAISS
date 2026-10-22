#!/usr/bin/env python3
"""Audit the five paper enzymes against GenBank, UniProt and local ModelSEED.

Standard library only. Default: reproduce the saved protein evidence. --refresh:
fetch both protein records live, without silently falling back to old evidence.
Each invocation creates a new timestamped directory; previous runs are preserved.
Native reactions are literature-curated, not inferred from sequence identity.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.request import Request, urlopen

PROJECT = Path(__file__).resolve().parent


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def get_url(url):
    for attempt in range(3):
        try:
            request = Request(url, headers={'User-Agent': 'NativeReactionAudit/1.0'})
            with urlopen(request, timeout=25) as response:
                return response.read()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def fasta_sequence(text, accession):
    lines = text.strip().splitlines()
    if not lines or not lines[0].startswith('>'):
        raise ValueError('Expected a single FASTA record')
    if accession not in re.split(r'[\s|]+', lines[0][1:]):
        raise ValueError(f'FASTA header does not identify {accession}')
    if any(line.startswith('>') for line in lines[1:]):
        raise ValueError('Expected exactly one FASTA record')
    sequence = ''.join(''.join(lines[1:]).split()).upper()
    if not sequence or not re.fullmatch(r'[A-Z]+', sequence):
        raise ValueError('Empty or invalid protein sequence')
    return sequence


def uniprot_entry(raw, accession):
    data = json.loads(raw)
    # Archived evidence uses the UniProt search response wrapper.
    if 'results' in data:
        if len(data['results']) != 1:
            raise ValueError('Expected exactly one UniProt entry')
        data = data['results'][0]
    if data['primaryAccession'] != accession:
        raise ValueError(f'UniProt accession changed: {data["primaryAccession"]}')
    if not data['sequence']['value']:
        raise ValueError('UniProt entry has no sequence')
    return data


def has_crossref(aliases, expected):
    """Match identifier within its namespace, never by numeric substring."""
    namespace, identifier = expected.split(':', 1)
    for group in aliases.split('|'):
        if ':' not in group:
            continue
        ns, values = group.split(':', 1)
        if ns.strip() == namespace and identifier in [v.strip() for v in values.split(';')]:
            return True
    return False


def database_scan(directory, references):
    files = sorted(directory.glob('reaction_[0-9][0-9].tsv'))
    compounds = sorted(directory.glob('compound_[0-9][0-9].tsv'))
    if not files or not compounds:
        raise ValueError(f'Missing ModelSEED reaction/compound TSVs in {directory}')
    patterns = {e['enzyme']: re.compile(e['search_regex'], re.I) for e in references}
    expected = {rid for e in references for rid in e['modelseed_crossrefs']}
    records, candidates, hashes = {}, {n: [] for n in patterns}, {}
    counts = {'reaction': 0, 'compound': 0}
    csv.field_size_limit(10 ** 8)
    for kind, paths in [('reaction', files), ('compound', compounds)]:
        for path in paths:
            before = sha256(path)
            with path.open() as stream:
                for row in csv.DictReader(stream, delimiter='\t'):
                    counts[kind] += 1
                    if kind == 'reaction' and row['id'] in expected:
                        if row['id'] in records:
                            raise ValueError(f'Duplicate reaction {row["id"]}')
                        records[row['id']] = row
                    text = ' '.join(row.get(k, '') for k in ('name', 'definition', 'aliases'))
                    for name, pattern in patterns.items():
                        if pattern.search(text):
                            candidates[name].append({
                                'kind': kind, 'id': row['id'], 'name': row.get('name', ''),
                                'definition': row.get('definition', ''),
                                'aliases': row.get('aliases', ''), 'file': str(path),
                            })
            if before != sha256(path):
                raise ValueError(f'Database changed during scan: {path}')
            hashes[str(path)] = before
    return records, candidates, hashes, counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh', action='store_true', help='Fetch current GenBank and UniProt records')
    parser.add_argument('--db', type=Path, default=PROJECT.parents[1] / 'ModelSEEDDatabase/Biochemistry')
    parser.add_argument('--references', type=Path, default=PROJECT / 'data/native_reaction_references.json')
    parser.add_argument('--evidence-dir', type=Path,
                        default=PROJECT / 'results/colleague_review_20261017/evidence')
    parser.add_argument('--output-root', type=Path, default=PROJECT / 'results/native_reaction_audit')
    args = parser.parse_args(argv)
    stamp = datetime.now(timezone.utc)
    run = args.output_root.resolve() / stamp.strftime('%Y%m%dT%H%M%S%fZ')
    run.mkdir(parents=True, exist_ok=False)
    evidence = run / 'evidence'
    evidence.mkdir()
    reference_hash = sha256(args.references)
    script_hash = sha256(__file__)
    reference = json.loads(args.references.read_text())
    records, candidates, db_hashes, counts = database_scan(args.db.resolve(), reference['enzymes'])
    rows, details, errors, evidence_meta = [], [], [], []
    fasta = []
    for enzyme in reference['enzymes']:
        name, gb, up = (enzyme[k] for k in ('enzyme', 'genbank', 'uniprot'))
        print(f'Checking {name}: {gb} / {up}', flush=True)
        urls = {
            'uniprot': f'https://rest.uniprot.org/uniprotkb/{up}.json',
            'genbank': 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
                       f'?db=protein&id={gb}&rettype=fasta&retmode=text',
        }
        row = {k: enzyme[k] for k in ('enzyme', 'genbank', 'uniprot', 'native_reaction', 'native_role', 'source', 'caveat')}
        row.update(genbank_url=f'https://www.ncbi.nlm.nih.gov/protein/{gb}',
                   uniprot_url=f'https://www.uniprot.org/uniprotkb/{up}/entry',
                   protein_evidence_mode='live_retrieval' if args.refresh else 'archived_snapshot',
                   organism='', length='', exact_sequence_match='', sequence_sha256='',
                   reviewed='', protein_error='')
        protein = None
        try:
            if args.refresh:
                raw_up = get_url(urls['uniprot'])
                raw_gb = get_url(urls['genbank'])
                time.sleep(0.4)  # Respect NCBI's unauthenticated request limit.
            else:
                raw_up = (args.evidence_dir / f'{name}_uniprot.json').read_bytes()
                raw_gb = (args.evidence_dir / f'{name}_{gb}.fasta').read_bytes()
            up_path, gb_path = evidence / f'{name}_uniprot.json', evidence / f'{name}_{gb}.fasta'
            up_path.write_bytes(raw_up)
            gb_path.write_bytes(raw_gb)
            for kind, path in [('uniprot', up_path), ('genbank', gb_path)]:
                evidence_meta.append({'enzyme': name, 'database': kind, 'url': urls[kind],
                                      'file': str(path.relative_to(run)), 'sha256': sha256(path),
                                      'copied_or_retrieved_utc': datetime.now(timezone.utc).isoformat(),
                                      'origin': 'live' if args.refresh else str(args.evidence_dir.resolve())})
            protein = uniprot_entry(raw_up, up)
            sequence = fasta_sequence(raw_gb.decode(), gb)
            match = sequence == protein['sequence']['value']
            row.update(organism=protein['organism']['scientificName'], length=len(sequence),
                       exact_sequence_match=match,
                       reviewed=protein.get('entryType', 'not supplied by snapshot'),
                       sequence_sha256=hashlib.sha256(sequence.encode()).hexdigest())
            fasta.append(f'>{name}|{gb}|{up}\n{sequence}\n')
            if not match:
                raise ValueError('GenBank and UniProt sequences differ')
        except Exception as exc:
            row['protein_error'] = f'{type(exc).__name__}: {exc}'
            errors.append(f'{name}: {row["protein_error"]}')
        verified, problems = [], []
        for rid, crossref in enzyme['modelseed_crossrefs'].items():
            if rid not in records:
                problems.append(f'{rid} missing')
            elif not has_crossref(records[rid].get('aliases', ''), crossref):
                problems.append(f'{rid} lacks expected {crossref}')
            else:
                verified.append(rid)
        row['modelseed_native_reaction_ids'] = ';'.join(verified)
        row['modelseed_status'] = ('verification_failed' if problems else
                                  'present' if verified else 'no_exact_native_entry_located')
        row['modelseed_error'] = '; '.join(problems)
        errors.extend(f'{name}: {problem}' for problem in problems)
        row['search_limit'] = ('Literature-curated cross-reference checks plus compound/reaction name and alias search; '
                               'not exhaustive structure matching or proof of absence. Reaction coverage does not establish protein mapping.')
        rows.append(row)
        details.append({'summary': row, 'curated_reference': enzyme,
                        'modelseed_records': [records[r] for r in enzyme['modelseed_crossrefs'] if r in records],
                        'name_search_candidates_not_automatically_native_matches': candidates[name],
                        'uniprot_description': protein.get('proteinDescription') if protein else None})
    if reference_hash != sha256(args.references) or script_hash != sha256(__file__):
        errors.append('Reference data or script changed during execution; rerun required')
    with (run / 'enzymes.tsv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    (run / 'enzymes.json').write_text(json.dumps(details, indent=2) + '\n')
    (run / 'enzymes.fasta').write_text(''.join(fasta))
    try:
        commit = subprocess.check_output(['git', '-C', str(args.db.resolve()), 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    manifest = {
        'started_utc': stamp.isoformat(), 'completed_utc': datetime.now(timezone.utc).isoformat(),
        'protein_evidence_mode': 'live_retrieval' if args.refresh else 'archived_snapshot',
        'database_scope': 'local ModelSEED checkout; not a live ModelSEED retrieval',
        'modelseed_commit': commit, 'database_record_counts': counts,
        'inputs_sha256': {str(Path(__file__).resolve()): script_hash,
                          str(args.references.resolve()): reference_hash, **db_hashes},
        'protein_evidence': evidence_meta, 'paper': reference['paper'],
        'accession_source': reference['accession_source'],
        'reaction_equation_convention': 'Literature summaries omit protonation-dependent H+ terms; raw ModelSEED definitions retained in enzymes.json.',
        'absence_interpretation': 'No exact entry located by this audit; not proof of database absence.',
        'errors': errors, 'success': not errors,
    }
    columns = ['enzyme', 'genbank', 'uniprot', 'organism', 'exact_sequence_match',
               'native_reaction', 'modelseed_native_reaction_ids', 'modelseed_status', 'caveat', 'source']
    esc = lambda value: html.escape(str(value))
    table = '<tr>' + ''.join(f'<th>{esc(k)}</th>' for k in columns) + '</tr>'
    for row in rows:
        table += '<tr>' + ''.join(f'<td>{esc(row[k])}</td>' for k in columns) + '</tr>'
    (run / 'report.html').write_text(
        '<!doctype html><html><meta charset="utf-8"><title>Five native reactions</title>'
        '<style>body{font:16px/1.5 system-ui;margin:30px}table{border-collapse:collapse}'
        'th,td{padding:10px;border:1px solid #ccd;vertical-align:top}th{background:#eef}</style>'
        '<h1>Five literature enzymes: identity and native reactions</h1>'
        f'<p>Protein evidence: {esc(manifest["protein_evidence_mode"])}. '
        f'ModelSEED: local checkout {esc(commit)}. Generated {esc(manifest["completed_utc"])}.</p>'
        '<p>Equations omit protonation-dependent H+ terms. Name-search candidates are retained in enzymes.json; '
        'they are not automatically counted as native matches. No exact entry located does not prove absence.</p>'
        f'<p>Validation errors: {esc(errors)}</p><table>{table}</table></html>\n')
    manifest['outputs_sha256'] = {p.name: sha256(p) for p in run.iterdir() if p.is_file()}
    (run / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Results: {run}')
    print(f'{sum(r["exact_sequence_match"] is True for r in rows)}/5 exact sequence matches; '
          f'{sum(r["modelseed_status"] == "present" for r in rows)}/5 native reactions represented.')
    for error in errors:
        print(error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
