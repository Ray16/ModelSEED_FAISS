#!/usr/bin/env python
"""Resolve the shortlisted ModelSEED reactions to real, testable enzymes.

Reactions are only a guide; an experimentalist needs proteins. For each shortlisted
amide-forming reaction this script resolves enzymes from UniProt, preferring the
most specific identifier available:

  1. Rhea reaction ID  -> enzymes annotated as catalysing THAT exact reaction
  2. EC number         -> enzymes of that activity (broader)
  3. product name      -> name-matched synthase/ligase (fallback, weakest)

Routes 1 and 2 BOTH always run and are merged; only route 3 is a fallback, used
when neither structured route returns a hit. The whole of UniProtKB is searched
(Swiss-Prot and TrEMBL); reviewed status is a RANKING key, not a filter, so an
unreviewed enzyme annotated to the exact Rhea reaction still outranks a reviewed
entry matched only by EC.

Two admission rules keep the output honest:
  * reactions below the OOD floor from 01 (results/ood.json) are refused
    outright - the method has already declared them out-of-distribution, so
    carrying their enzymes would present non-candidates as candidates;
  * the per-reaction slots are taxonomy-stratified, since bacterial and archaeal
    sources are the practical starting point for expression in E. coli and an
    unstratified sort fills the slots with reviewed vertebrate homologues.

Outputs (results/):
  enzymes_uniprot.tsv    one row per enzyme, with reaction provenance
  enzymes.fasta          sequences, ready for ordering or modelling
"""
from __future__ import annotations

import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from .paths import DB, RES, ood_floor
from .metanetx import seed_to_rhea
from .uniprot import search

import csv

from .enzyme_resolution import (
    product_names, resolve_reaction, select_reactions, is_prokaryote, evidence_strength,
)


ENZYME_COLUMNS = [
    'rxn_id', 'reaction_name', 'cosine_similarity', 'acylates_free_diamine',
    'resolved_via', 'evidence_strength', 'uniprot', 'entry_name', 'protein_name',
    'gene', 'organism', 'lineage', 'is_prokaryote', 'ec', 'length', 'has_pdb',
    'pdb', 'alphafold',
    'reviewed', 'evidence', 'function', 'catalytic_activity', 'cofactor', 'pathway',
    'subunit', 'go_molecular_function', 'go_biological_process', 'pfam', 'interpro',
    'binding_sites', 'active_sites', 'keywords', 'sequence',
]
ANNOTATION_COLUMNS = [
    'uniprot', 'entry_name', 'organism', 'lineage', 'rxn_id', 'reviewed',
    'evidence', 'ec',
    'length', 'pdb', 'alphafold', 'pfam', 'interpro', 'cofactor', 'pathway', 'subunit',
    'catalytic_activity', 'function', 'go_molecular_function', 'go_biological_process',
    'binding_sites', 'active_sites', 'keywords',
]
UNRESOLVED_COLUMNS = [
    'rxn_id', 'cosine_similarity', 'reaction_name', 'has_rhea', 'ec', 'definition',
    'lookup_status',
]


def load_aliases():
    aliases = defaultdict(lambda: defaultdict(set))
    path = DB / 'Aliases/Unique_ModelSEED_Reaction_Aliases.txt'
    with path.open() as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            aliases[row['ModelSEED ID']][row.get('Source', '')].add(row['External ID'])
    return aliases


def main():
    from .paths import ensure_directories
    ensure_directories()
    hits = pd.read_csv(RES / 'amide_forming_neighbours.tsv', sep='\t')
    # 03 already applied the mechanism screens before ranking, so every row here
    # acylates a free small-molecule amine; just de-duplicate across queries.
    floor = ood_floor()
    useful = select_reactions(hits, floor)
    below_count = hits.rxn_id.nunique() - len(useful)
    print(f'{len(useful)} distinct small-molecule amide-forming reactions to resolve '
          f'({below_count} refused below the OOD floor {floor:.4f})', flush=True)

    aliases = load_aliases() if not useful.empty else {}
    # ModelSEED records a Rhea ID for only ~61% of ranked reactions. MetaNetX
    # reconciles identifiers across databases, so pivoting ModelSEED -> MNXR ->
    # Rhea recovers reactions ModelSEED does not cross-reference - including
    # rxn21314 (BibC), the top-scoring hit, which otherwise reaches UniProt only
    # through a string match on its product name.
    metanetx = seed_to_rhea() if not useful.empty else {}

    def resolve(hit):
        result = resolve_reaction(hit, aliases, metanetx, search_client=search)
        note = (f'  [TRUNCATED: {"+".join(result.truncated)} pool exceeds page budget]'
                if result.truncated else '')
        if result.failed:
            note += '  [LOOKUP FAILED: results may be incomplete]'
        print(f'  {hit.rxn_id} cos={hit.cosine_similarity:.3f} via {result.route}: '
              f'{result.pool_size} enzymes{note}', flush=True)
        return result

    workers = max(1, int(os.environ.get('UNIPROT_WORKERS', 6)))
    print(f'Resolving {len(useful)} reactions on {workers} UniProt workers', flush=True)
    hits_list = list(useful.itertuples())
    with ThreadPoolExecutor(max_workers=workers) as pool:
        resolved = list(pool.map(resolve, hits_list))

    records, seen = [], {}
    for hit, result in zip(hits_list, resolved):
        entries = result.entries
        for entry in entries:
            key = (entry['Entry'], hit.rxn_id)
            if key in seen:
                continue
            seen[key] = True
            records.append({
                'rxn_id': hit.rxn_id,
                'reaction_name': hit.reaction_name,
                'cosine_similarity': hit.cosine_similarity,
                'acylates_free_diamine': hit.acylates_free_diamine,
                'resolved_via': entry.get('_route', ''),
                'evidence_strength': evidence_strength(entry.get('_route', '')),
                'uniprot': entry['Entry'],
                'entry_name': entry.get('Entry Name', ''),
                'protein_name': entry.get('Protein names', ''),
                'gene': entry.get('Gene Names', ''),
                'organism': entry.get('Organism', ''),
                # Kept because it drives expression-host choice downstream; the
                # derived is_prokaryote flag discards the detail.
                'lineage': entry.get('Taxonomic lineage', ''),
                'is_prokaryote': is_prokaryote(entry),
                'ec': entry.get('EC number', ''),
                'length': entry.get('Length', ''),
                'has_pdb': bool(entry.get('PDB', '').strip()),
                'pdb': entry.get('PDB', ''),
                'alphafold': entry.get('AlphaFoldDB', ''),
                'reviewed': entry.get('Reviewed', ''),
                'evidence': entry.get('Protein existence', ''),
                'function': entry.get('Function [CC]', ''),
                'catalytic_activity': entry.get('Catalytic activity', ''),
                'cofactor': entry.get('Cofactor', ''),
                'pathway': entry.get('Pathway', ''),
                'subunit': entry.get('Subunit structure', ''),
                'go_molecular_function': entry.get('Gene Ontology (molecular function)', ''),
                'go_biological_process': entry.get('Gene Ontology (biological process)', ''),
                'pfam': entry.get('Pfam', ''),
                'interpro': entry.get('InterPro', ''),
                'binding_sites': entry.get('Binding site', ''),
                'active_sites': entry.get('Active site', ''),
                'keywords': entry.get('Keywords', ''),
                'sequence': entry.get('Sequence', ''),
            })

    # Make the recall gap visible: a reaction with no resolvable identifier
    # yields no candidate, which in the final table is indistinguishable from
    # "no enzyme exists". Writing the failures out separates the two.
    unresolved = [{'rxn_id': hit.rxn_id, 'cosine_similarity': hit.cosine_similarity,
                   'reaction_name': hit.reaction_name,
                   'has_rhea': bool(aliases.get(hit.rxn_id, {}).get('Rhea')
                                    or metanetx.get(hit.rxn_id)),
                   'ec': hit.ec, 'definition': getattr(hit, 'definition', ''),
                   'lookup_status': result.status}
                  for hit, result in zip(hits_list, resolved) if not result.entries]
    pd.DataFrame(unresolved, columns=UNRESOLVED_COLUMNS).to_csv(
        RES / 'reactions_without_enzymes.tsv', sep='\t', index=False)
    print(f'{len(unresolved)} ranked reactions produced no enzyme '
          f'(written to reactions_without_enzymes.tsv)', flush=True)

    enzymes = pd.DataFrame(records, columns=ENZYME_COLUMNS)
    if enzymes.empty:
        # Overwrite previous outputs even when nothing resolves. Otherwise the
        # next table stage would consume candidates left over from an old run.
        enzymes.drop(columns=['sequence']).to_csv(RES / 'enzymes_uniprot.tsv', sep='\t', index=False)
        enzymes[ANNOTATION_COLUMNS].to_csv(RES / 'enzyme_annotations.tsv', sep='\t', index=False)
        (RES / 'enzymes.fasta').write_text('')
        print('No enzymes resolved', flush=True)
        return 1 if any(result.failed for result in resolved) else 0
    enzymes = enzymes.sort_values(
        ['acylates_free_diamine', 'cosine_similarity', 'is_prokaryote'],
        ascending=[False, False, False]).reset_index(drop=True)
    enzymes.drop(columns=['sequence']).to_csv(RES / 'enzymes_uniprot.tsv',
                                              sep='\t', index=False)
    enzymes[ANNOTATION_COLUMNS].to_csv(RES / 'enzyme_annotations.tsv',
                                      sep='\t', index=False)

    with open(RES / 'enzymes.fasta', 'w') as handle:
        for row in enzymes.itertuples():
            if row.sequence:
                handle.write(f'>{row.uniprot}|{row.entry_name}|{row.rxn_id}|'
                             f'cos={row.cosine_similarity}|{row.organism}\n')
                for i in range(0, len(row.sequence), 60):
                    handle.write(row.sequence[i:i + 60] + '\n')

    pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 48)
    print(f'\n{len(enzymes)} enzymes across {enzymes.rxn_id.nunique()} reactions '
          f'({int(enzymes.is_prokaryote.sum())} prokaryotic, '
          f'{int(enzymes.has_pdb.sum())} with a PDB structure)')
    print(enzymes.head(25)[['rxn_id', 'cosine_similarity', 'uniprot', 'gene',
                            'organism', 'protein_name']].to_string(index=False))


if __name__ == '__main__':
    raise SystemExit(main())
