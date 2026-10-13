#!/usr/bin/env python
"""Ingest a precomputed UniProt -> ModelSEED link table as a retrieval route.

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
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .paths import RES

LINKS = Path(os.environ.get(
    'UNIPROT_MSD_LINKS',
    Path(__file__).resolve().parent.parent / 'uniprot-msd-links.tsv'))

# A reaction annotated to more proteins than this is treated as over-annotated:
# the links are kept but flagged, and only the first MAX_PER_REACTION are taken.
PROMISCUOUS_PROTEINS = int(os.environ.get('PROMISCUOUS_PROTEINS', 500))
MAX_PER_REACTION = int(os.environ.get('MAX_LINKS_PER_REACTION', 200))


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
             promiscuous=PROMISCUOUS_PROTEINS, cap=MAX_PER_REACTION):
    """Flatten into candidate rows, flagging and capping over-annotated reactions."""
    distance = {}
    if hits is not None and len(hits):
        column = ('substrate_distance' if 'substrate_distance' in hits.columns
                  else None)
        if column:
            distance = (hits.sort_values(column, ascending=False)
                            .drop_duplicates('rxn_id')
                            .set_index('rxn_id')[column].to_dict())

    rows = []
    for reaction, members in by_reaction.items():
        over = len(members) > promiscuous
        for accession, sequence in sorted(members.items())[:cap]:
            rows.append({
                'uniprot': accession,
                'sequence': sequence,
                'length': len(sequence) or None,
                'rxn_id': reaction,
                'discovery_route': 'precomputed_link',
                'evidence_strength': 'precomputed_rhea_or_go',
                'discovery_tier': 'uniprot_msd_links',
                'substrate_distance_measured': distance.get(reaction, 0.0),
                'n_proteins_on_reaction': len(members),
                'over_annotated_reaction': over,
            })
    return pd.DataFrame(rows)


def main():
    from .paths import ensure_directories
    ensure_directories()
    hits = pd.read_csv(RES / 'amide_forming_neighbours.tsv', sep='\t')
    screened = sorted(set(hits.rxn_id))

    by_reaction, stats = load_links(keep_reactions=screened)
    frame = to_frame(by_reaction, hits=hits)
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
    }
    print(json.dumps(summary, indent=2), flush=True)
    (RES / 'precomputed_link_summary.json').write_text(
        json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
