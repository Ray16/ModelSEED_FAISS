"""Namespace-safe reaction annotation parsing and mapping snapshot selection."""
from pathlib import Path
import hashlib
import json
import re

EVIDENCE_ORDER = {'experimental_direct': 0, 'curated_direct': 1,
                  'homology_supported_direct': 2, 'computational_direct': 3}

def annotation_evidence(activity, requested):
    """Verify the returned annotation rather than trusting a search result."""
    wanted={str(i).removeprefix('RHEA:') for i in requested}
    rows=[r for r in reaction_statements(activity) if r['rhea_id'] in wanted]
    if not rows:
        return {'mapping_evidence_tier':'unverified_reaction_search',
                'mapping_rhea_ids':'', 'mapping_evidence':''}
    best=min(rows,key=lambda r:EVIDENCE_ORDER[r['evidence_tier']])
    return {'mapping_evidence_tier':best['evidence_tier'],
            'mapping_rhea_ids':';'.join(sorted({'RHEA:'+r['rhea_id'] for r in rows})),
            'mapping_evidence':' | '.join(sorted({r['evidence'] for r in rows}))}

def reaction_statements(text):
    """Return only reaction identifiers, with evidence from that statement.

    RHEA-COMP identifiers describe reaction participants. Their numbers occupy a
    different namespace, and must never be joined to reaction alias numbers.
    """
    rows=[]
    for block in (text or '').split('CATALYTIC ACTIVITY:')[1:]:
        ids=sorted(set(re.findall(r'(?<![\w-])Rhea:RHEA:(\d+)\b',block)))
        evidence=re.search(r'Evidence=(\{.*?\});',block)
        value=evidence.group(1) if evidence else ''
        tier=('experimental_direct' if 'ECO:0000269' in value else
              'homology_supported_direct' if 'ECO:0000250' in value else
              'computational_direct' if 'ECO:0000255' in value or 'ECO:0000256' in value else
              'curated_direct')
        rows.extend(dict(rhea_id=i,evidence=value,evidence_tier=tier) for i in ids)
    return rows

def default_mapping(project):
    """Honor an audited project policy; fail closed if its snapshot changes."""
    project=Path(project);policy=project/'mapping_policy.json'
    if not policy.exists():return project/'uniprot-msd-links.tsv'
    doc=json.loads(policy.read_text());path=(project/doc['default_links']).resolve()
    if not path.is_file():raise FileNotFoundError('Audited mapping snapshot is missing: '+str(path))
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=doc['sha256']:raise ValueError('Audited mapping snapshot hash mismatch: '+str(path))
    return path
