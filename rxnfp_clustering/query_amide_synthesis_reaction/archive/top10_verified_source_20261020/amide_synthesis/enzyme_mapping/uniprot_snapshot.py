"""Complete, release-checked UniProt snapshots with reaction-scoped evidence."""
from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

FIELDS = ('accession,id,reviewed,protein_name,gene_names,organism_name,organism_id,'
          'length,sequence,ec,cc_catalytic_activity,xref_proteomes,fragment,'
          'protein_existence,rhea')


class SnapshotClient:
    """Cache raw successful responses; a failure can never become a zero count."""

    def __init__(self, directory, *, source=None, offline=False):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.source = Path(source) if source is not None else self.directory
        self.offline = offline
        self.session = requests.Session()
        self.session.headers['User-Agent'] = 'ModelSEED-reaction-mapping-audit'

    def get(self, name, url, params=None):
        path = self.directory / name
        meta_path = path.with_suffix(path.suffix + '.meta.json')
        requested = requests.Request('GET', url, params=params).prepare().url
        source_path = self.source / name
        source_meta = source_path.with_suffix(source_path.suffix + '.meta.json')
        if source_path.exists() and source_meta.exists():
            meta = json.loads(source_meta.read_text())
            if meta['requested_url'] != requested:
                raise ValueError('Snapshot cache query changed: ' + name)
            if hashlib.sha256(source_path.read_bytes()).hexdigest() != meta['sha256']:
                raise ValueError('Snapshot cache bytes changed: ' + name)
            if source_path.resolve() != path.resolve():
                path.write_bytes(source_path.read_bytes())
                meta_path.write_bytes(source_meta.read_bytes())
            return source_path.read_text(), meta
        if self.offline:
            raise FileNotFoundError('Offline snapshot response or metadata missing: ' + name)
        for attempt in range(4):
            try:
                response = self.session.get(url, params=params, timeout=(20, 90))
                response.raise_for_status()
                body = response.content
                meta = {
                    'requested_url': requested, 'response_url': response.url,
                    'retrieved_utc': datetime.now(timezone.utc).isoformat(),
                    'headers': dict(response.headers),
                    'sha256': hashlib.sha256(body).hexdigest(),
                    'next_url': response.links.get('next', {}).get('url'),
                }
                temporary = path.with_suffix(path.suffix + '.partial')
                temporary.write_bytes(body)
                temporary.replace(path)
                meta_path.write_text(json.dumps(meta, indent=2))
                return body.decode(), meta
            except requests.RequestException:
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))

    def proteins(self, name, query):
        url = 'https://rest.uniprot.org/uniprotkb/search'
        params = {'query': query, 'format': 'json', 'size': 500, 'fields': FIELDS}
        rows, pages, release, total = [], [], None, None
        visited = set()
        while url:
            if url in visited:
                raise ValueError('UniProt pagination repeated a cursor')
            visited.add(url)
            body, meta = self.get(f'{name}_page{len(pages) + 1}.json', url, params)
            headers = {k.lower(): v for k, v in meta['headers'].items()}
            page_release = headers.get('x-uniprot-release')
            page_total = int(headers['x-total-results'])
            if pages and (page_release != release or page_total != total):
                raise ValueError('UniProt release/count changed during pagination')
            release, total = page_release, page_total
            rows.extend(json.loads(body)['results'])
            pages.append(meta)
            url, params = meta['next_url'], None
            if url and not self.offline:
                time.sleep(0.2)
        accessions = [r['primaryAccession'] for r in rows]
        if len(set(accessions)) != len(accessions) or len(rows) != total:
            raise ValueError(f'Incomplete/duplicated UniProt snapshot: {name}')
        return sorted(rows, key=lambda r: r['primaryAccession']), {
            'query': query, 'release': release, 'total': total, 'pages': pages,
            'complete': True,
        }


def reaction_annotations(record, wanted):
    """Read exact JSON reaction/direction fields without borrowing their evidence."""
    wanted = {str(x).removeprefix('RHEA:') for x in wanted}
    result = []
    for comment in record.get('comments', []):
        if comment.get('commentType') != 'CATALYTIC ACTIVITY':
            continue
        scopes = [('reaction', comment.get('reaction', {}))]
        scopes.extend(('direction', d) for d in comment.get('physiologicalDirections', []))
        for scope, annotation in scopes:
            evidence = annotation.get('evidences', [])
            codes = {e['evidenceCode'] for e in evidence}
            tier = ('experimental_direct' if 'ECO:0000269' in codes else
                    'homology_supported_direct' if 'ECO:0000250' in codes else
                    'computational_direct' if codes & {'ECO:0000255', 'ECO:0000256'} else
                    'curated_direct')
            for ref in annotation.get('reactionCrossReferences', []):
                match = re.fullmatch(r'RHEA:(\d+)', ref.get('id', ''))
                if ref.get('database') != 'Rhea' or not match or match[1] not in wanted:
                    continue
                result.append({
                    'rhea_id': match[1], 'scope': scope, 'evidence_tier': tier,
                    'evidence': evidence, 'equation': annotation.get('name', ''),
                    'ec': annotation.get('ecNumber', ''),
                })
    return result


def protein_metadata(record):
    description = record.get('proteinDescription', {})
    names = [description.get('recommendedName', {})]
    names += description.get('submissionNames', []) + description.get('alternativeNames', [])
    ec = {x['value'] for name in names for x in name.get('ecNumbers', [])}
    ec.update(c['reaction']['ecNumber'] for c in record.get('comments', [])
              if c.get('commentType') == 'CATALYTIC ACTIVITY' and c.get('reaction', {}).get('ecNumber'))
    sequence = record.get('sequence', {}).get('value', '')
    if not sequence or not re.fullmatch(r'[A-Z]+', sequence):
        raise ValueError('Missing or invalid protein sequence: ' + record['primaryAccession'])
    return {
        'accession': record['primaryAccession'], 'sequence': sequence,
        'sequence_sha256': hashlib.sha256(sequence.encode()).hexdigest(),
        'length': len(sequence),
        'protein_name': description.get('recommendedName', {}).get('fullName', {}).get('value', '')
                        or next((n.get('fullName', {}).get('value', '') for n in names if n.get('fullName')), ''),
        'all_names': '; '.join(n.get('fullName', {}).get('value', '') for n in names),
        'ec': ';'.join(sorted(ec)),
        'gene_names': ';'.join(x['value'] for gene in record.get('genes', [])
                             for key in ['geneName', 'synonyms', 'orderedLocusNames']
                             for x in ([gene[key]] if key == 'geneName' and key in gene else gene.get(key, []))),
        'taxon_id': record.get('organism', {}).get('taxonId', ''),
        'organism': record.get('organism', {}).get('scientificName', ''),
        'proteomes': ';'.join(sorted({r['id'] for r in record.get('uniProtKBCrossReferences', [])
                                     if r.get('database') == 'Proteomes'})),
        'reviewed': record.get('entryType') == 'UniProtKB reviewed (Swiss-Prot)',
        'fragment': bool(description.get('flag', '').lower().startswith('fragment')),
    }
