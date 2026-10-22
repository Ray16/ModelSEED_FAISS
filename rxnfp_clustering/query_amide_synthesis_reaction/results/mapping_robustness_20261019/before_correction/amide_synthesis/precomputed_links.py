#!/usr/bin/env python
"""Ingest a precomputed UniProt -> ModelSEED link table as a retrieval route.

Current default: the audited direct-reaction snapshot in mapping_policy.json.
The original merged Rhea/GO table is retained for comparison and can be explicitly
selected through UNIPROT_MSD_LINKS. The historical discussion below predates the
discovery of RHEA-COMP collisions and noncatalytic subunits; chemical screening of
a reaction alone does not validate its associated protein links.

ModelSEED ships no reaction -> enzyme mapping, which is why this project
reaches UniProt through Rhea, EC and product names, one reaction at a time.
A colleague's table (`uniprot-msd-links.tsv`, 262,380 proteins / 485,495 links
over 11,207 reactions) builds the same bridge globally and offline, by pairing
UniProt's curated Rhea annotations with ModelSEED's Rhea aliases, plus a second
route through GO.

MEASURED, on the 205 reactions this project's screen admits:

    reactions resolved   theirs 93    ours 74    union 109
    proteins             theirs 2518  ours 183   overlap 85

So it is far broader, and worth ingesting. The two are complementary rather
than ranked, and each covers the other's blind spot:

  * Theirs reaches reactions ours cannot, mostly because ours only queries
    UniProt for reactions that carry a usable identifier.
  * Ours reaches rxn21314 (BibC) and rxn21308 (PubC), where theirs returns
    NOTHING - those reactions have no Rhea alias in ModelSEED itself, and only
    the MetaNetX pivot recovers it. They are the two highest-scoring candidates
    in the project, so this is not a marginal gap.
  * Ours returns 8 proteins for rxn22343 (MurE) against their 1, because theirs
    is restricted to reviewed entries.

WHY IT IS FILTERED THROUGH THE SCREEN. The table's GO route maps any protein
annotated with ATPase activity onto the ATP-hydrolysis reaction: rxn00062
("ATP phosphohydrolase") carries 17,801 proteins, including DNA mismatch-repair
proteins, helicases and transporters. Chemically true, biologically useless.
The table carries no provenance column, so a precise Rhea-derived link and a
GO-derived one cannot be told apart from the file alone.

This matters here because the queries are ATP-dependent ligases. The protection
is structural rather than heuristic: links are only ever accepted for reactions
that the acylation screen has already PROVEN form an amide on a free amine, and
no cofactor-hydrolysis reaction can pass that screen. Promiscuous reactions are
additionally reported, and capped, so a single over-annotated reaction cannot
dominate the candidate list.
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

from .freshness import tracked_stage
from .paths import RES
from .mapping_quality import default_mapping, EVIDENCE_ORDER

LINKS = Path(os.environ.get('UNIPROT_MSD_LINKS') or
             default_mapping(Path(__file__).resolve().parent.parent))

# A reaction annotated to more proteins than this is treated as over-annotated:
# the links are kept but flagged, and only the first MAX_PER_REACTION are taken.
PROMISCUOUS_PROTEINS = int(os.environ.get('PROMISCUOUS_PROTEINS', 500))
MAX_PER_REACTION = int(os.environ.get('MAX_LINKS_PER_REACTION', 0))

def load_evidence(links=LINKS, keep_reactions=None):
    """Join reaction-specific provenance only to the matching frozen mapping."""
    project=Path(__file__).resolve().parent.parent
    policy=project/'mapping_policy.json'
    if not policy.exists():return {}
    doc=json.loads(policy.read_text())
    if Path(links).resolve()!=(project/doc['default_links']).resolve():return {}
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


def to_frame(by_reaction, hits=None,
             promiscuous=PROMISCUOUS_PROTEINS, cap=MAX_PER_REACTION, evidence=None):
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
    from .paths import ensure_directories
    ensure_directories()
    hits = pd.read_csv(RES / 'amide_forming_neighbours.tsv', sep='\t')
    screened = sorted(set(hits.rxn_id))

    by_reaction, stats = load_links(keep_reactions=screened)
    evidence=load_evidence(keep_reactions=screened)
    frame = to_frame(by_reaction, hits=hits,evidence=evidence)
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
    }
    print(json.dumps(summary, indent=2), flush=True)
    (RES / 'precomputed_link_summary.json').write_text(
        json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
