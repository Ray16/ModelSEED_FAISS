#!/usr/bin/env python
"""Curated experimental properties UniProt holds and this pipeline never asked for.

WHAT THIS IS ACTUALLY FOR, since the first version of this docstring got the
emphasis wrong. Seven UniProt fields were absent from `uniprot.FIELDS`
entirely. This stage was started to chase KINETICS, because BRENDA is the
blocked source for it and a way around a blocker is tempting - but that was
reaching for the named obstacle rather than asking what would most help. The
measurement below settles it: kinetics is 0.5% of candidates, while UniProt's
curated FAMILY STATEMENT covers 75% and is the genuinely valuable field,
because it is assigned independently of the Pfam/InterPro grouping that the
domain and clade routes rest on. It is therefore a free cross-check on the
two routes that contribute the most candidates and carry the weakest evidence.
That check is implemented in `prose_contradicts` below and is the main reason
this module exists. Kinetics is kept because ~190 proteins with a measured KM
is still worth having, not because it carries the stage.

MEASURED COVERAGE, 2026-10-13, so the thin fields are not oversold. Sampled
300 reviewed and 200 unreviewed candidates:

    field                     reviewed   unreviewed
    kinetics (KM / kcat)          5.3%         0.0%
    pH dependence                 1.3%            -
    temperature dependence        1.3%            -
    activity regulation           6.0%            -
    subcellular location         58.0%        31.5%
    domain architecture          31.3%        79.0%
    sequence similarities        98.7%        72.0%

Two things follow, and they point in opposite directions:

1. KINETICS IS REAL BUT THIN. 5.3% of 4,302 reviewed candidates is roughly 230
   proteins with MEASURED KM or kcat, with an ECO:0000269 publication behind
   each. That is a genuine partial substitute for BRENDA - not a replacement,
   because BRENDA would cover the ~31,000 EC-bearing candidates rather than
   230 - and it is the only kinetic data obtainable here without credentials.
   It is 0% for unreviewed entries, which is a property of the source.

2. THE OTHER FIELDS ARE NOT THIN, AND DOMAIN ARCHITECTURE IS BETTER COVERED
   FOR UNREVIEWED ENTRIES (79%) THAN REVIEWED ONES (31%). Scoping this stage
   to reviewed proteins, which the kinetics number alone would have suggested,
   would have thrown away the better-covered half. So every candidate is
   queried, and the extra fields cost nothing per request.

`sequence_similarities` is worth singling out: at 98.7%/72% it is UniProt's own
curated family statement in prose ("Belongs to the D-alanine--D-alanine ligase
family"), which is independent of the Pfam/InterPro assignment this project
groups by, and so a free cross-check on the family routes.

Follows the `enrich` contract exactly: adds only, never overwrites, and writes
to its OWN file rather than back over the master table.
"""
from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from amide_synthesis.infrastructure import uniprot
from amide_synthesis.infrastructure.freshness import tracked_stage
from amide_synthesis.infrastructure.paths import RES

BATCH = int(os.environ.get('PROPERTIES_BATCH', 20))
THREADS = int(os.environ.get('PROPERTIES_THREADS', 6))

# UniProt's return-field names on the left, ours on the right. The correct
# field name for kinetics is `kinetics`; `cc_biophysicochemical_properties`,
# the name the documentation suggests, returns HTTP 400.
WANTED = {
    'kinetics': 'kinetics',
    'ph_dependence': 'ph_optimum',
    'temp_dependence': 'temperature_optimum',
    'cc_activity_regulation': 'activity_regulation',
    'cc_subcellular_location': 'subcellular_location',
    'ft_domain': 'domain_architecture',
    'cc_similarity': 'sequence_similarities',
}
RETURNED = {
    'Kinetics': 'kinetics',
    'pH dependence': 'ph_optimum',
    'Temperature dependence': 'temperature_optimum',
    'Activity regulation': 'activity_regulation',
    'Subcellular location [CC]': 'subcellular_location',
    'Domain [FT]': 'domain_architecture',
    'Sequence similarities': 'sequence_similarities',
}
FIELDS = 'accession,' + ','.join(WANTED)

# Both accession lengths UniProt issues. The assayed panel is NCBI and has no
# UniProtKB entry, so querying it would be guaranteed misses counted as
# failures.
ACCESSION = (r'^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]'
             r'|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})$')

KM = re.compile(r'KM=([0-9.]+)\s*(nM|uM|mM|M)\b', re.I)
UNIT_TO_MM = {'nm': 1e-6, 'um': 1e-3, 'mm': 1.0, 'm': 1e3}


def fetch(accessions, search=uniprot.search, batch=BATCH, threads=THREADS):
    """Curated property fields for a list of accessions, batched."""
    chunks = [accessions[i:i + batch]
              for i in range(0, len(accessions), batch)]
    if not chunks:
        return [], 0

    def one(chunk):
        query = ' OR '.join(f'accession:{a}' for a in chunk)
        rows, failed, _ = search(query, max_pages=2, fields=FIELDS)
        return rows, failed

    out, failures = [], 0
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for rows, failed in pool.map(one, chunks):
            out.extend(rows)
            failures += int(failed)
    return out, failures


def smallest_km_mm(text):
    """Tightest reported KM in mM, or None.

    A single entry often reports KM for several substrates. The smallest is
    the most informative single number for "does this enzyme bind anything
    tightly", and the full text is kept alongside so nothing is lost.
    """
    values = []
    for amount, unit in KM.findall(str(text or '')):
        try:
            values.append(float(amount) * UNIT_TO_MM[unit.lower()])
        except (ValueError, KeyError):
            continue
    return min(values) if values else None


def tidy(rows):
    """Returned records -> our column names, with KM parsed out."""
    if not rows:
        return pd.DataFrame(columns=['uniprot'] + list(RETURNED.values()))
    frame = pd.DataFrame(rows).rename(
        columns={'Entry': 'uniprot', **RETURNED})
    for column in RETURNED.values():
        if column not in frame.columns:
            frame[column] = pd.NA
    frame['km_min_mm'] = frame.kinetics.map(smallest_km_mm)
    frame['has_measured_kinetics'] = frame.kinetics.fillna('') \
        .astype(str).str.strip().ne('')
    keep = ['uniprot', 'km_min_mm', 'has_measured_kinetics'] \
        + list(RETURNED.values())
    return frame[keep].drop_duplicates('uniprot').reset_index(drop=True)


def prose_contradicts(similarities, protein_names):
    """UniProt's own family statement disagreeing with amide-forming chemistry.

    A THIRD independent axis. `master_table` challenges a candidate on its
    protein name and its EC; this is UniProt's curated "Belongs to the ...
    family" line, which is assigned separately from the Pfam/InterPro grouping
    the family routes rely on, so it is a free cross-check on them.

    The rescue spans the NAME as well as the prose, and that is load-bearing.
    Reading the prose alone flagged 148 rows of the 2-oxoacid dehydrogenase
    family, 59 of which are "Dihydrolipoyllysine-residue succinyltransferase"
    - the E2 components, genuine succinyl transferases, and succinyl is a C4
    DIACID donor, which is chemistry this project actively wants. The prose is
    true (they are part of that complex) but it describes the complex, not the
    subunit's reaction. Qualifying either way round therefore clears it, which
    leaves the E1 decarboxylase components correctly flagged and the E2
    transferases correctly kept.
    """
    from amide_synthesis.candidate_selection.master_table import DISQUALIFYING, QUALIFYING
    prose = similarities.fillna('').astype(str).str.lower()
    names = protein_names.fillna('').astype(str).str.lower()
    disqualified = prose.str.contains('|'.join(DISQUALIFYING), regex=True)
    qualified = (names + ' ' + prose).str.contains(
        '|'.join(QUALIFYING), regex=True)
    return disqualified & ~qualified


@tracked_stage('properties')
def main():
    from amide_synthesis.infrastructure.paths import ensure_directories
    ensure_directories()

    source = RES / 'master_candidates_enriched.tsv'
    if not source.exists():
        source = RES / 'master_candidates.tsv'
    table = pd.read_csv(source, sep='\t', low_memory=False)
    accessions = sorted(
        table[table.uniprot.astype(str).str.match(ACCESSION)]
        .uniprot.dropna().unique())
    print(f'{source.name}: {len(table):,} candidates, '
          f'{len(accessions):,} UniProt accessions to query', flush=True)

    previous=RES/'curated_properties.tsv'
    cached=pd.read_csv(previous,sep='\t',low_memory=False) if previous.exists() else pd.DataFrame()
    known=set(cached.uniprot) if len(cached) else set()
    pending=[a for a in accessions if a not in known]
    rows, failures = fetch(pending)
    properties=pd.concat([cached[cached.uniprot.isin(accessions)] if len(cached) else cached,tidy(rows)],ignore_index=True)
    print(f'reused {len(set(accessions)&known)} property snapshots; fetched {len(pending)} new records',flush=True)

    # The family-prose cross-check needs the protein's own name, which lives
    # in the candidate table, so it is joined here rather than re-fetched.
    # Reaction-anchored rows are exempt for the same reason they are exempt
    # from the name check: the screen proved amide formation on their own
    # reaction, which outranks a family label.
    names = (table.drop_duplicates('uniprot')
             .set_index('uniprot')
             .reindex(properties.uniprot))
    properties['family_prose_contradicts_chemistry'] = (
        prose_contradicts(properties.sequence_similarities,
                          names.protein_name.reset_index(drop=True))
        & names.discovery_route.reset_index(drop=True).ne('reaction').values)
    properties.to_csv(RES / 'curated_properties.tsv', sep='\t', index=False)

    reviewed = table.reviewed.fillna('').astype(str).str.lower() \
        .str.startswith('reviewed')
    reviewed_set = set(table[reviewed].uniprot.dropna())
    kinetic = properties[properties.has_measured_kinetics]

    def filled(column):
        return int(properties[column].fillna('').astype(str)
                   .str.strip().ne('').sum())

    summary = {
        'queried': len(accessions),
        'records_retrieved': int(len(properties)),
        'failed_batches': failures,
        'with_measured_kinetics': int(len(kinetic)),
        'with_measured_kinetics_reviewed': int(
            kinetic.uniprot.isin(reviewed_set).sum()),
        'with_parsed_km': int(properties.km_min_mm.notna().sum()),
        'family_prose_contradicts_chemistry': int(
            properties.family_prose_contradicts_chemistry.sum()),
        'coverage_percent': {
            name: round(filled(name) / max(len(properties), 1) * 100, 1)
            for name in RETURNED.values()
        },
        'note': 'Curated properties are REPORTED, not ranked on. Kinetics is '
                'a partial substitute for BRENDA: ~230 proteins here against '
                'the ~31,000 EC-bearing candidates BRENDA would cover.',
    }
    print(json.dumps(summary, indent=2), flush=True)
    (RES / 'curated_properties_summary.json').write_text(
        json.dumps(summary, indent=2) + '\n')

    if len(kinetic):
        print('\nCandidates with MEASURED kinetic parameters '
              '(tightest KM first):', flush=True)
        show = kinetic.dropna(subset=['km_min_mm']).nsmallest(12, 'km_min_mm')
        for row in show.itertuples():
            print(f'  {row.uniprot}  KM_min={row.km_min_mm:g} mM  '
                  f'{str(row.kinetics)[:88]}', flush=True)


if __name__ == '__main__':
    main()
