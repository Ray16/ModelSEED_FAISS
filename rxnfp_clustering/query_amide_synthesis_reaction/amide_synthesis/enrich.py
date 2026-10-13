#!/usr/bin/env python
"""Fill in annotation for candidates a route delivered with only a sequence.

Routes differ in how much they carry. The reaction routes request the full
UniProt record, but two do not:

  precomputed_link  the external link table is (accession, reactions,
                    sequence). 1,131 candidates arrived with no EC, no
                    organism, no Pfam, no catalytic activity - nothing to
                    judge them by beyond the reaction that found them.

  assayed           the 50-member panel is NCBI accessions from a separate
                    project. These are the MOST valuable rows in the table -
                    they are the only proteins anywhere here with measured
                    titre on the nylon-6,6 diad - and the least annotated.

                    Two routes, because accession matching is weak here:
                    NCBI -> UniProt reaches only 10 of 48, since most are
                    environmental sequences with no UniProtKB entry. EXACT
                    SEQUENCE lookup through UniParc reaches 48 of 50 and
                    supplies organism and protein name for every one. UniParc
                    archives every sequence ever deposited in any member
                    database, so this is identity, not homology - a sequence
                    absent from UniProtKB is usually still there under a
                    RefSeq or EMBL cross-reference.

Annotation is only ever ADDED. An existing value always wins, so re-running
this cannot overwrite a stronger route's record with a weaker lookup.

WRITES TO ITS OWN FILE, never back over `master_candidates.tsv`. Rewriting the
input in place would make the pipeline order-dependent in exactly the way
`reference_actives` already was: a later `master_table` run would silently
discard every enriched value, and nothing would report it. `master_table`
stays the composition step, this stays the annotation step, and `shortlist`
prefers the enriched file when it exists.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from . import families
from . import uniprot
from . import uniparc
from .paths import RES

BATCH = int(os.environ.get('ENRICH_BATCH', 20))
THREADS = 6

# Columns worth filling; keys are UniProt's names, values are ours.
FILLABLE = {
    'Protein names': 'protein_name', 'Gene Names': 'gene',
    'Organism': 'organism', 'Taxonomic lineage': 'lineage',
    'EC number': 'ec', 'Length': 'length',
    'Catalytic activity': 'catalytic_activity', 'PDB': 'pdb',
    'Protein existence': 'protein_existence', 'Reviewed': 'reviewed',
    'Cofactor': 'cofactor', 'Pathway': 'pathway', 'Subunit structure': 'subunit',
    'Function [CC]': 'function', 'Pfam': 'pfam', 'InterPro': 'interpro',
    'AlphaFoldDB': 'alphafold', 'Binding site': 'binding_site',
    'Active site': 'active_site', 'Keywords': 'keywords',
    'Gene Ontology (molecular function)': 'go_function',
    'Gene Ontology (biological process)': 'go_process',
}


def _column(frame, name, default=''):
    """Always a Series: DataFrame.get returns a scalar when the column is absent."""
    if name in frame.columns:
        return frame[name]
    return pd.Series([default] * len(frame), index=frame.index)


def blank(series):
    text = series.fillna('').astype(str).str.strip()
    return text.eq('') | text.str.lower().eq('nan')


def needs_enrichment(table, column='catalytic_activity'):
    """Accessions that look like UniProt entries and carry no annotation."""
    missing = blank(table.get(column, pd.Series('', index=table.index)))
    # A UniProt accession, as opposed to an NCBI one like WP_0835.1
    looks_uniprot = table.uniprot.astype(str).str.match(
        r'^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9]'
        r'([A-Z][A-Z0-9]{2}[0-9]){0,2}$')
    return table.uniprot[missing & looks_uniprot].dropna().unique().tolist()


def fetch(accessions, search=uniprot.search, batch=BATCH, threads=THREADS):
    """Full records for a list of accessions, batched into OR queries."""
    chunks = [accessions[i:i + batch] for i in range(0, len(accessions), batch)]

    def one(chunk):
        query = ' OR '.join(f'accession:{a}' for a in chunk)
        rows, failed, _ = search(query, max_pages=2)
        return rows, failed

    out, failures = [], 0
    if not chunks:
        return out, failures
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for rows, failed in pool.map(one, chunks):
            out.extend(rows)
            failures += int(failed)
    return out, failures


def resolve_ncbi(accessions, search=uniprot.search, threads=THREADS):
    """Best-effort NCBI accession -> UniProt entry. Partial by nature."""
    def one(accession):
        base = str(accession).split('.')[0]
        for query in (f'xref:embl-{base}', f'"{base}"'):
            rows, failed, _ = search(query, max_pages=1)
            if rows:
                return accession, rows[0]
        return accession, None

    mapped = {}
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for accession, row in pool.map(one, accessions):
            if row:
                mapped[accession] = row
    return mapped


def apply_records(table, records, key='uniprot'):
    """Merge fetched records in, never overwriting an existing value."""
    if not records:
        return table, 0
    frame = pd.DataFrame(records)
    if 'Entry' not in frame.columns:
        return table, 0
    frame = frame.drop_duplicates('Entry').set_index('Entry')
    filled = 0
    for source, target in FILLABLE.items():
        if source not in frame.columns:
            continue
        incoming = table[key].map(frame[source])
        if target not in table.columns:
            table[target] = incoming
            filled += int(incoming.notna().sum())
            continue
        gap = blank(table[target])
        table.loc[gap, target] = incoming[gap]
        filled += int((gap & incoming.notna()).sum())
    return table, filled


def main():
    from .paths import ensure_directories
    ensure_directories()
    table = pd.read_csv(RES / 'master_candidates.tsv', sep='\t', low_memory=False)

    pending = needs_enrichment(table)
    print(f'{len(pending)} UniProt accessions carry no annotation', flush=True)
    records, failures = fetch(pending)
    table, filled = apply_records(table, records)
    print(f'  retrieved {len(records)} records, filled {filled} values '
          f'({failures} failed batches)', flush=True)

    # The assayed panel: NCBI accessions, the most valuable and least
    # annotated rows in the table.
    assayed = table[table.evidence_strength.eq('measured_activity')]
    ncbi = assayed.uniprot.dropna().astype(str).tolist()
    mapped = resolve_ncbi(ncbi) if ncbi else {}
    print(f'  assayed panel: resolved {len(mapped)} of {len(ncbi)} '
          f'NCBI accessions to UniProt', flush=True)
    if mapped:
        table['uniprot_equivalent'] = table.uniprot.map(
            {k: v.get('Entry') for k, v in mapped.items()})
        records2 = [dict(row, Entry=accession)
                    for accession, row in mapped.items()]
        table, filled2 = apply_records(table, records2)
        print(f'  filled {filled2} values for the panel', flush=True)

    # Exact-sequence identity for the panel, where accessions fail.
    sequences = dict(zip(assayed.uniprot, assayed.get('sequence', '')))
    archived = uniparc.lookup_many(sequences) if sequences else {}
    print(f'  assayed panel: {len(archived)} of {len(sequences)} sequences '
          f'found in UniParc by exact match', flush=True)
    if archived:
        frame = pd.DataFrame.from_dict(archived, orient='index')
        for column in frame.columns:
            incoming = table.uniprot.map(frame[column])
            if column not in table.columns:
                table[column] = incoming
            else:
                gap = blank(table[column])
                table.loc[gap, column] = incoming[gap]
        # Fill the display fields from the archive where still empty.
        for target, source in (('organism', 'uniparc_organism'),
                               ('protein_name', 'uniparc_protein_name'),
                               ('gene', 'uniparc_gene')):
            if source in table.columns:
                gap = blank(table.get(target, pd.Series('', index=table.index)))
                if target not in table.columns:
                    table[target] = ''
                table.loc[gap, target] = table.loc[gap, source]

    # A Pfam accession is not a family name. 1,121 rows arrived from the
    # expansion routes with PF accessions and no readable label, which makes
    # them unreadable in the shortlist and uncountable by family.
    gap = blank(_column(table, 'family')) & ~blank(_column(table, 'pfam'))
    if gap.any():
        accessions = []
        for value in table.loc[gap, 'pfam']:
            accessions.extend(families.split_accessions(str(value)))
        names = families.pfam_names(accessions)
        resolved = 0
        primary, every = [], []
        for value in table.loc[gap, 'pfam']:
            accs = families.split_accessions(str(value))
            labels = [names.get(a) or a for a in accs]
            main = next((l for l in labels
                         if l not in families.SECONDARY_DOMAINS),
                        labels[0] if labels else '')
            primary.append(main)
            every.append(';'.join(labels))
            resolved += int(bool(main) and main not in accs)
        table.loc[gap, 'family'] = primary
        if 'family_all' in table.columns:
            still = blank(table.family_all) & gap
            table.loc[still, 'family_all'] = pd.Series(every, index=table.index[gap])[still]
        print(f'  family names: filled {int(gap.sum())} rows '
              f'({resolved} with a readable InterPro name)', flush=True)

    table.to_csv(RES / 'master_candidates_enriched.tsv', sep='\t', index=False)

    coverage = {column: round(float((~blank(table[column])).mean() * 100), 1)
                for column in ('ec', 'organism', 'lineage', 'pfam', 'interpro',
                               'catalytic_activity', 'function', 'keywords',
                               'alphafold', 'pdb')
                if column in table.columns}
    summary = {'accessions_enriched': len(pending),
               'records_retrieved': len(records),
               'values_filled': filled,
               'failed_batches': failures,
               'ncbi_requested': len(ncbi),
               'ncbi_resolved': len(mapped),
               'uniparc_exact_matches': len(archived),
               'uniparc_requested': len(sequences),
               'coverage_percent': coverage}
    print('\n' + json.dumps(summary, indent=2), flush=True)
    (RES / 'enrichment_summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
