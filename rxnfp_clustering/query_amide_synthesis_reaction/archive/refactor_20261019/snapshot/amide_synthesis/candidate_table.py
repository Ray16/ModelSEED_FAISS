#!/usr/bin/env python
"""Collapse the enzyme hits into ONE prioritised candidate table for scoring.

Each row is a candidate enzyme to test, carrying the evidence behind it: the
ModelSEED reaction it came from, the cosine similarity to the nylon-6,6 query
that retrieved it, how the protein was resolved (exact Rhea reaction vs broad
EC), whether its native amine acceptor is a free diamine, and its sequence
length / structure availability.

`priority` is a transparent, deliberately simple ordering - not a model score:
exact-reaction evidence and nylon-like substrate chemistry are what make a
candidate worth bench time. It is meant to be REPLACED by the user's own
metric; the columns it is built from are all exposed.

Cosine no longer contributes. It is a retrieval coordinate, dominated by the
cofactor rather than the amide bond, and ranking on it is actively misleading:
AsbA is the third-highest cosine in the shortlist and ranks 32 of 36 on the
independent structural-geometry proxy, while PylC (cosine 0.554) ranks 11.
`substrate_distance` replaces it - same role, measured on the acyl donor and
amine acceptor the screen proved, so it describes the chemistry rather than the
cofactor. Cosine is retained as a column for provenance.

Output: results/candidates_ranked.tsv  (+ candidates_top.fasta for the top N)
"""
from __future__ import annotations


import pandas as pd

from .freshness import tracked_stage
from .paths import RES, ood_floor
from .enzyme_resolution import select_reactions
from . import families as _families
from .mapping_quality import annotation_evidence, EVIDENCE_ORDER
import re
TOP_FASTA = 25


def build_candidate_table(enzymes, hits):
    """Rank proteins against the current screen, excluding stale reaction rows.

    Family annotations must already be attached. This function does no file or
    network I/O; all scoring evidence remains exposed in the returned columns.
    """
    # best query per reaction, so each candidate states which framing found it
    rank_key = ('substrate_distance' if 'substrate_distance' in hits.columns
                else 'cosine_similarity')
    best = (hits.sort_values(rank_key, ascending=False)
                .drop_duplicates('rxn_id')
                .set_index('rxn_id')[[c for c in
                                      ['query', 'reaction_name', 'definition',
                                       'n_free_amines_on_donor', 'external_ids',
                                       'substrate_distance', 'found_via',
                                       'self_condensation']
                                      if c in hits.columns]])
    # A partial rerun (e.g. run_all.py --offline) re-screens reactions while
    # reusing the previous enzyme file. A left join would then keep enzymes whose
    # reaction the current screen rejected, silently resurrecting dropped
    # candidates. Restrict to reactions present in the CURRENT screen instead.
    enzymes = enzymes[enzymes.rxn_id.isin(best.index)]
    table = enzymes.join(best, on='rxn_id', rsuffix='_hit')

    claimed=table.resolved_via.fillna('').str.startswith('Rhea')
    for i in table.index[claimed]:
        requested=re.findall(r'\d+',str(table.at[i,'resolved_via']).split(':',1)[-1])
        activity=table.at[i,'catalytic_activity'] if 'catalytic_activity' in table else ''
        ev=annotation_evidence(activity if isinstance(activity,str) else '',requested)
        for key,value in ev.items():table.at[i,key]=value
        table.at[i,'evidence_strength']=ev['mapping_evidence_tier']
    table['exact_reaction_evidence'] = table.get('mapping_evidence_tier',pd.Series('',index=table.index)).isin(EVIDENCE_ORDER)
    if 'evidence_strength' not in table.columns:
        table['evidence_strength'] = table.exact_reaction_evidence.map(
            {True: 'exact_reaction', False: 'activity_ec'})
    weight = {'exact_reaction': 2, 'activity_ec': 1, 'name_match': 0,
              'experimental_direct':2.3,'curated_direct':2,
              'homology_supported_direct':1.7,'computational_direct':1.5,
              'unverified_reaction_search':0}
    # 3x on substrate distance so chemistry outweighs the convenience terms
    # (prokaryote, PDB) that were previously able to overturn it.
    substrate = table.get('substrate_distance')
    if substrate is None:
        substrate = pd.Series(0.0, index=table.index)
    table['substrate_distance'] = substrate.fillna(0.0)
    table['priority'] = (
        table.evidence_strength.map(weight).fillna(0)
        + 2 * table.acylates_free_diamine.astype(int)
        + table.is_prokaryote.astype(int)
        + table.has_pdb.astype(int)
        + 3 * table.substrate_distance)

    columns = ['priority', 'uniprot', 'gene', 'organism', 'family', 'protein_name',
               'substrate_distance', 'found_via', 'self_condensation',
               'cosine_similarity', 'query', 'rxn_id', 'reaction_name',
               'pfam', 'family_all',
               'acylates_free_diamine', 'n_free_amines_on_donor',
               'exact_reaction_evidence', 'evidence_strength', 'resolved_via',
               'mapping_evidence_tier','mapping_rhea_ids','mapping_evidence',
               'ec', 'is_prokaryote',
               'has_pdb', 'pdb', 'alphafold', 'length', 'reviewed', 'evidence',
               'cofactor', 'pathway', 'catalytic_activity', 'definition',
               'external_ids']
    columns = [c for c in columns if c in table.columns]
    table = (table[columns]
             .sort_values('priority', ascending=False)
             .drop_duplicates('uniprot')
             .reset_index(drop=True))
    return table


@tracked_stage('table')
def main():
    from .paths import ensure_directories
    ensure_directories()
    enzymes = pd.read_csv(RES / 'enzymes_uniprot.tsv', sep='\t')
    hits = pd.read_csv(RES / 'amide_forming_neighbours.tsv', sep='\t')

    # Family travels WITH every candidate row: the shortlist is only
    # interpretable by family (one family found sixteen times is convergence,
    # not sixteen hits), so it must not live in a separate file.
    annotations = pd.read_csv(RES / 'enzyme_annotations.tsv', sep='\t')
    pfam_by_uniprot = (annotations.drop_duplicates('uniprot')
                       .set_index('uniprot')['pfam'].to_dict()
                       if 'pfam' in annotations.columns else {})
    enzymes = _families.attach(enzymes, pfam_by_uniprot)

    hits = select_reactions(hits, ood_floor())
    stale = set(enzymes.rxn_id) - set(hits.rxn_id)
    if stale:
        print(f'dropping enzyme rows from {len(stale)} reactions outside the current '
              f'screen/OOD gate: {sorted(stale)[:5]}', flush=True)
    table = build_candidate_table(enzymes, hits)
    table.to_csv(RES / 'candidates_ranked.tsv', sep='\t', index=False)

    # sequences for the top slice, so the list is directly actionable
    fasta_source = {}
    for line in (RES / 'enzymes.fasta').read_text().split('>'):
        if not line.strip():
            continue
        header, *body = line.splitlines()
        fasta_source.setdefault(header.split('|')[0], ''.join(body))
    with open(RES / 'candidates_top.fasta', 'w') as handle:
        for row in table.head(TOP_FASTA).itertuples():
            seq = fasta_source.get(row.uniprot)
            if not seq:
                continue
            handle.write(f'>{row.uniprot}|{row.gene}|{row.rxn_id}|'
                         f'cos={row.cosine_similarity}|priority={row.priority:.3f}\n')
            for i in range(0, len(seq), 60):
                handle.write(seq[i:i + 60] + '\n')

    pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 40)
    print(f'{len(table)} distinct candidate enzymes '
          f'({int(table.acylates_free_diamine.sum())} on free-diamine reactions, '
          f'{int(table.exact_reaction_evidence.sum())} with exact-reaction evidence)')
    print(table.head(20)[['priority', 'uniprot', 'gene', 'family',
                          'cosine_similarity', 'acylates_free_diamine',
                          'evidence_strength']].to_string(index=False))
    assert len(enzymes) >= len(table), 'candidate table larger than its source'


if __name__ == '__main__':
    main()
