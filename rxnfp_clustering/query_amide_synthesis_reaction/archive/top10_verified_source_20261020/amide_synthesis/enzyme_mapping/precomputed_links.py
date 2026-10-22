#!/usr/bin/env python
"""Join screened reactions to a policy-pinned protein mapping with evidence.

Keep all accessions by default. Validate the evidence sidecar and sequence hash,
retain reaction/direction-specific tiers, and flag heavily populated reactions.
The explicit legacy override remains labeled unverified. A chemistry screen
validates reaction relevance, not the correctness of every protein assignment.
"""
from __future__ import annotations

import json
import os
import csv
import gzip
import hashlib
from collections import defaultdict
from pathlib import Path

import pandas as pd

from amide_synthesis.infrastructure.freshness import tracked_stage
from amide_synthesis.infrastructure.paths import RES, PROJECT
from amide_synthesis.enzyme_mapping.mapping_quality import default_mapping, EVIDENCE_ORDER
from amide_synthesis.enzyme_mapping.reaction_equivalence import validate_bridges, expand_mapping

LINKS = Path(os.environ.get('UNIPROT_MSD_LINKS') or
             default_mapping(PROJECT))

# A reaction annotated to more proteins than this is treated as over-annotated:
# the links are kept but flagged, and only the first MAX_PER_REACTION are taken.
PROMISCUOUS_PROTEINS = int(os.environ.get('PROMISCUOUS_PROTEINS', 500))
MAX_PER_REACTION = int(os.environ.get('MAX_LINKS_PER_REACTION', 0))

def load_evidence(links=LINKS, keep_reactions=None):
    """Join reaction-specific provenance only to the matching frozen mapping."""
    project=PROJECT
    policy=project/'mapping_policy.json'
    if not policy.exists():return {}
    doc=json.loads(policy.read_text())
    if Path(links).resolve()!=(project/doc['default_links']).resolve():return {}
    default_mapping(project)  # Check the policy-pinned evidence sidecar as well.
    if hashlib.sha256(Path(links).read_bytes()).hexdigest()!=doc['sha256']:
        raise ValueError('Mapping snapshot changed before evidence join')
    result={};keep=set(keep_reactions) if keep_reactions is not None else None
    with gzip.open(project/doc['evidence'],'rt') as handle:
        for row in csv.DictReader(handle,delimiter='\t'):
            rxn=row['modelseed_reaction']
            if keep is not None and rxn not in keep:continue
            key=(rxn,row['accession']);old=result.get(key)
            if old is None:
                result[key]=dict(row,mapping_rhea_ids=row['rhea_reaction'])
            else:
                ids=sorted(set(old['mapping_rhea_ids'].split(';'))|{row['rhea_reaction']})
                if EVIDENCE_ORDER[row['evidence_tier']]<EVIDENCE_ORDER[old['evidence_tier']]:
                    result[key]=dict(row,mapping_rhea_ids=';'.join(ids))
                else:old['mapping_rhea_ids']=';'.join(ids)
    return result


def load_links(path=LINKS, keep_reactions=None):
    """Read the link table, optionally restricted to a set of reactions.

    Returns (reaction -> {accession: sequence}, stats). Restricting while
    reading keeps the whole 262k-row file from being materialised.
    """
    path = Path(path)
    by_reaction = defaultdict(dict)
    stats = {'rows': 0, 'links': 0, 'with_sequence': 0}
    if not path.exists():
        return by_reaction, stats

    keep = set(keep_reactions) if keep_reactions is not None else None
    with path.open() as handle:
        for line in handle:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 2 or not parts[0]:
                continue
            stats['rows'] += 1
            accession = parts[0]
            sequence = parts[2] if len(parts) > 2 else ''
            for reaction in parts[1].split(','):
                reaction = reaction.strip()
                if not reaction or (keep is not None and reaction not in keep):
                    continue
                by_reaction[reaction][accession] = sequence
                stats['links'] += 1
                if sequence:
                    stats['with_sequence'] += 1
    return by_reaction, stats


def resolve_links(path=LINKS, keep_reactions=None):
    """Use audited direct aliases first, then explicitly validated equation bridges.

    A legacy override never receives evidence or bridges from another snapshot.
    Bridge source reactions are loaded even when they are outside the screen.
    """
    policy_path = PROJECT / 'mapping_policy.json'
    audited = (policy_path.exists() and Path(path).resolve() ==
               (PROJECT / json.loads(policy_path.read_text())['default_links']).resolve())
    bridges = validate_bridges(PROJECT, keep_reactions) if audited else []
    keep = None if keep_reactions is None else set(keep_reactions)
    expanded_keep = None if keep is None else keep | {b['source_reaction'] for b in bridges}
    by_reaction, stats = load_links(path, expanded_keep)
    evidence = load_evidence(path, expanded_keep)
    added = expand_mapping(by_reaction, evidence, bridges)
    if keep is not None:
        by_reaction = {r: members for r, members in by_reaction.items() if r in keep}
        evidence = {key: value for key, value in evidence.items() if key[0] in keep}
    stats['validated_bridge_pairs_added'] = len(added)
    stats['validated_bridges'] = bridges
    return by_reaction, evidence, stats, audited


def to_frame(by_reaction, hits=None,
             promiscuous=PROMISCUOUS_PROTEINS, cap=MAX_PER_REACTION, evidence=None,
             require_evidence=False):
    """Retain all links by default; an explicit cap prefers stronger evidence."""
    evidence=evidence or {}
    distance = {}
    chemistry={}
    if hits is not None and len(hits):
        column = ('substrate_distance' if 'substrate_distance' in hits.columns
                  else None)
        if column:
            distance = (hits.sort_values(column, ascending=False)
                            .drop_duplicates('rxn_id')
                            .set_index('rxn_id')[column].to_dict())
        chemistry=hits.sort_values('substrate_distance',ascending=False).drop_duplicates('rxn_id').set_index('rxn_id').to_dict('index') if 'substrate_distance' in hits else {}

    rows = []
    for reaction, members in by_reaction.items():
        over = len(members) > promiscuous
        ranked=sorted(members.items(),key=lambda pair:(
            EVIDENCE_ORDER.get(evidence.get((reaction,pair[0]),{}).get('evidence_tier'),99),pair[0]))
        if cap>0:ranked=ranked[:cap]
        for accession, sequence in ranked:
            ev=evidence.get((reaction,accession),{})
            if require_evidence and ev.get('evidence_tier') not in EVIDENCE_ORDER:
                raise ValueError('Missing or invalid audited reaction evidence: '+accession+' / '+reaction)
            if ev and hashlib.sha256(sequence.encode()).hexdigest()!=ev['sequence_sha256']:
                raise ValueError('Mapping/evidence sequence mismatch: '+accession)
            rows.append({
                **{k:v for k,v in chemistry.get(reaction,{}).items() if k in
                   ('reaction_name','query','cosine_similarity','self_condensation','acylates_free_diamine')},
                'uniprot': accession,
                'sequence': sequence,
                'length': len(sequence) or None,
                'rxn_id': reaction,
                'discovery_route': 'precomputed_link',
                'evidence_strength': ev.get('evidence_tier','precomputed_rhea_or_go'),
                'mapping_evidence_tier':ev.get('evidence_tier','legacy_unverified'),
                'mapping_rhea_ids':ev.get('mapping_rhea_ids',''),
                'mapping_evidence':ev.get('evidence',''),
                'mapping_source':ev.get('source','legacy_merged_mapping'),
                'mapping_relation':ev.get('mapping_relation','exact_rhea_alias' if ev else 'legacy_unverified'),
                'mapping_source_reaction':ev.get('mapping_source_reaction',reaction),
                'mapping_chemical_scope':ev.get('mapping_chemical_scope','alias_chemistry_not_independently_certified'),
                'mapping_bridge_audit':ev.get('mapping_bridge_audit',''),
                'mapping_sequence_sha256':ev.get('sequence_sha256',''),
                'mapping_alias_fanout':ev.get('alias_fanout',''),
                'protein_name':ev.get('protein_name',''),
                'reviewed':'reviewed' if ev else '',
                'taxon_id':ev.get('taxon_id',''),
                'discovery_tier': 'uniprot_msd_links',
                'substrate_distance_measured': distance.get(reaction, 0.0),
                'n_proteins_on_reaction': len(members),
                'over_annotated_reaction': over,
            })
    return pd.DataFrame(rows)


@tracked_stage('links')
def main():
    from amide_synthesis.infrastructure.paths import ensure_directories
    ensure_directories()
    hits = pd.read_csv(RES / 'amide_forming_neighbours.tsv', sep='\t')
    screened = sorted(set(hits.rxn_id))

    by_reaction, evidence, stats, audited = resolve_links(keep_reactions=screened)
    frame = to_frame(by_reaction, hits=hits,evidence=evidence,require_evidence=audited)
    frame.to_csv(RES / 'precomputed_link_candidates.tsv', sep='\t', index=False)

    existing = set()
    master = RES / 'master_candidates.tsv'
    if master.exists():
        existing = set(pd.read_csv(master, sep='\t', low_memory=False).uniprot)

    summary = {
        'link_table_rows': stats['rows'],
        'screened_reactions': len(screened),
        'reactions_resolved': len(by_reaction),
        'candidate_rows': int(len(frame)),
        'unique_proteins': int(frame.uniprot.nunique()) if len(frame) else 0,
        'new_vs_master': int(frame[~frame.uniprot.isin(existing)].uniprot.nunique())
                         if len(frame) else 0,
        'over_annotated_reactions': int(frame[frame.over_annotated_reaction]
                                        .rxn_id.nunique()) if len(frame) else 0,
        'with_sequence': int(frame.sequence.astype(bool).sum()) if len(frame) else 0,
        'by_evidence':frame.evidence_strength.value_counts().to_dict(),
        'mapping_path':str(LINKS),
        'default_cap':MAX_PER_REACTION,
        'validated_bridge_pairs_added':stats['validated_bridge_pairs_added'],
        'validated_bridges':stats['validated_bridges'],
    }
    print(json.dumps(summary, indent=2), flush=True)
    (RES / 'precomputed_link_summary.json').write_text(
        json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
