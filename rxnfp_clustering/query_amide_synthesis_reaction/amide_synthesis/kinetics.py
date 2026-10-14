#!/usr/bin/env python
"""Measured kinetics by EC, from SABIO-RK - which is not dead after all.

THE CORRECTION THIS RESTS ON. `INFORMATION_SOURCES.md` recorded SABIO-RK as
unusable: "its documented REST endpoint no longer serves data - 1182 bytes of
HTML, identical for every EC queried". That observation was right and the
conclusion drawn from it was wrong. The 1,182 bytes are a single-page
application shell (`<base href="/ui/">`), served for every unmatched path
including ones that never existed - which is why four different ECs gave
byte-identical replies. SABIO-RK was rewritten, not retired.

The current service documents its own export API inside that bundle:

    /export-api/sabio/kinlaw-entry/json?q=<solr>&page=<n>&pageSize=<n>

`q` is a Solr query, so `ECNumber:6.3.2.17` filters properly (17 entries
against 77,028 unfiltered), `pageSize` is honoured up to at least 1000, and
`Link` headers carry rel="next". No credentials. Only `kinlaw-entry` is
exported; the other entities 404.

WHAT IT IS WORTH, MEASURED. Our 40,628 EC-bearing candidates carry just 273
distinct complete EC numbers, so coverage is one cheap request each: 131 of
the 273 have kinetics, 5,643 entries between them, reaching 27,690 of 55,125
candidates. UniProt's own `kinetics` field reaches 189 proteins. This is the
source the project recorded as "the single highest-value unblock available"
and as needing BRENDA credentials; it needs none.

WHAT THIS IS NOT. Kinetics attach to an EC, not to a protein - a Km measured
on one family member is a property of the chemistry, not of our candidate.
It is recorded as `sabio_*` and is NOT folded into `master_priority` or the
convergence count, for the same reason curated ATP binding was rejected as an
axis: a property every member of a class shares cannot discriminate within
it. It is here because the goal is to resolve as much information per enzyme
as possible, and because a Km on the target chemistry is what a bench
scientist asks for next.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request

import pandas as pd

from .paths import RES

EXPORT = 'https://sabiork.h-its.org/export-api/sabio/kinlaw-entry/json'
PAGE_SIZE = 100
PAUSE = 0.35
RETRIES = 5
# Failures here are transient throttling, not missing data, and they landed
# on exactly the ECs that matter: a first run lost 21 of 273 including seven
# EC 6.3 amide ligases and carnosine synthase (6.3.2.11, 42 entries), all of
# which answered instantly on retry. So: exponential backoff, and a second
# pass over whatever still failed before anything is reported.
BACKOFF = 2.0
# The parameters a bench scientist asks for. SABIO names them exactly.
WANTED = ('Km', 'kcat', 'Ki', 'Vmax', 'kcat/Km')


def _fetch(query, page, size=PAGE_SIZE, timeout=90):
    """One page of kinetic-law entries, or None if the service refuses."""
    url = (f'{EXPORT}?q={urllib.parse.quote(query)}'
           f'&page={page}&pageSize={size}')
    for attempt in range(RETRIES):
        try:
            request = urllib.request.Request(
                url, headers={'Accept': 'application/json',
                              'User-Agent': 'amide-synthesis-pipeline'})
            with urllib.request.urlopen(request, timeout=timeout) as handle:
                return json.loads(handle.read().decode('utf-8', 'replace'))
        except Exception:
            if attempt == RETRIES - 1:
                return None
            time.sleep(BACKOFF ** attempt)
    return None


def entry_count(ec):
    """How many kinetic entries SABIO holds for an EC; None if the call failed."""
    payload = _fetch(f'ECNumber:{ec}', 1, size=1)
    if payload is None:
        return None
    return payload.get('meta', {}).get('total_count', 0)


def parameters(record):
    """The wanted kinetic parameters out of one entry, flattened."""
    law = record.get('kineticlaw') or {}
    description = record.get('enzyme_description') or {}
    proteins = description.get('proteins') or []
    accessions = ';'.join(sorted(
        {str(p.get('uniprot_id')) for p in proteins if p.get('uniprot_id')}))
    rows = []
    for parameter in law.get('parameter') or []:
        kind = ((parameter.get('parameter_type') or {}).get('name') or '')
        if kind not in WANTED:
            continue
        species = (parameter.get('species') or {}).get('species_key') or ''
        ec = description.get('ec_number')
        rows.append({
            'ec': ec,
            # SABIO labels multi-enzyme assemblies "Complex 6.3.4.16/2.1.3.3".
            # Such a record comes back because the COMPLEX contains an EC we
            # queried, but the measurement belongs to the assembly, not to
            # that single enzyme. The label is kept verbatim rather than
            # resolved to its parts, so it never joins to a candidate EC and
            # cannot inflate a per-EC median; this flag makes that explicit
            # instead of leaving a non-EC string in an `ec` column.
            'is_complex': str(ec or '').startswith('Complex '),
            'enzyme_name': description.get('enzyme_name'),
            'uniprot_ids': accessions,
            'parameter': kind,
            # `n_start_value` is SABIO's own unit-normalised value; the raw
            # value is meaningless without its unit, so both are kept.
            'value': parameter.get('start_value'),
            'unit': (parameter.get('unit') or {}).get('name'),
            'value_si': parameter.get('n_start_value'),
            'unit_si': (parameter.get('unit') or {}).get('n_name'),
            'species': species.split('|')[1].strip() if '|' in species else '',
            'wildtype': description.get('wildtype'),
            'pubmed': (record.get('publication') or {}).get('pubmed_id'),
            'entry_id': record.get('id'),
        })
    return rows


def collect(ec, limit=None):
    """Every wanted parameter SABIO holds for one EC."""
    rows, page = [], 1
    while True:
        payload = _fetch(f'ECNumber:{ec}', page)
        if payload is None:
            break
        for record in payload.get('data') or []:
            rows.extend(parameters(record))
        meta = payload.get('meta') or {}
        if page >= meta.get('total_pages', 1) or (
                limit and len(rows) >= limit):
            break
        page += 1
        time.sleep(PAUSE)
    return rows


def candidate_ecs(table):
    """The complete, numeric ECs our candidates carry."""
    found = set()
    for value in table.ec.dropna().astype(str):
        for part in value.split(';'):
            ec = part.strip()
            if ec.count('.') == 3 and '-' not in ec and all(
                    p.isdigit() for p in ec.split('.')):
                found.add(ec)
    return sorted(found)


def summarise(frame):
    """One row per (EC, parameter): how many, and the median in SI units."""
    if frame.empty:
        return pd.DataFrame(columns=['ec', 'parameter', 'n', 'median_si',
                                     'unit_si'])
    values = pd.to_numeric(frame.value_si, errors='coerce')
    grouped = frame.assign(value_si=values).dropna(subset=['value_si'])
    out = (grouped.groupby(['ec', 'parameter'])
           .agg(n=('value_si', 'size'), median_si=('value_si', 'median'),
                unit_si=('unit_si', 'first'))
           .reset_index())
    return out


def main():
    from .paths import ensure_directories
    ensure_directories()

    master = RES / 'master_candidates_enriched.tsv'
    if not master.exists():
        master = RES / 'master_candidates.tsv'
    table = pd.read_csv(master, sep='\t', low_memory=False,
                        usecols=lambda c: c in ('uniprot', 'ec'))
    ecs = candidate_ecs(table)
    print(f'{len(ecs)} distinct complete ECs among the candidates', flush=True)

    counts, rows = {}, []
    for index, ec in enumerate(ecs, 1):
        total = entry_count(ec)
        counts[ec] = total
        if total:
            rows.extend(collect(ec))
        time.sleep(PAUSE)
        if index % 50 == 0:
            print(f'  ...{index}/{len(ecs)} ECs, {len(rows):,} parameters',
                  flush=True)

    # Second pass over the transient failures, before anything is reported.
    retry = [ec for ec, n in counts.items() if n is None]
    if retry:
        print(f'retrying {len(retry)} ECs that failed', flush=True)
        for ec in retry:
            time.sleep(1.0)
            total = entry_count(ec)
            counts[ec] = total
            if total:
                rows.extend(collect(ec))
        still = [ec for ec in retry if counts.get(ec) is None]
        print(f'  recovered {len(retry) - len(still)}, still failing '
              f'{len(still)}', flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(RES / 'sabio_kinetics.tsv', sep='\t', index=False)
    digest = summarise(frame)
    digest.to_csv(RES / 'kinetics_by_ec.tsv', sep='\t', index=False)

    # Per candidate, so the shortlist and QUALIFIED can carry it.
    with_kinetics = {ec for ec, n in counts.items() if n}
    def covered(value):
        if pd.isna(value):
            return False
        return any(part.strip() in with_kinetics
                   for part in str(value).split(';'))
    table['sabio_has_kinetics'] = table.ec.apply(covered)
    km = digest[digest.parameter.eq('Km')].set_index('ec').median_si
    def best_km(value):
        if pd.isna(value):
            return None
        found = [km.get(p.strip()) for p in str(value).split(';')]
        found = [v for v in found if v is not None and pd.notna(v)]
        return min(found) if found else None
    table['sabio_median_km_M'] = table.ec.apply(best_km)
    table[['uniprot', 'sabio_has_kinetics', 'sabio_median_km_M']].to_csv(
        RES / 'kinetics_per_candidate.tsv', sep='\t', index=False)

    failed = sorted(ec for ec, n in counts.items() if n is None)
    summary = {
        'ecs_queried': len(ecs),
        'ecs_with_kinetics': int(len(with_kinetics)),
        'ecs_failed': failed,
        'kinetic_parameters': int(len(frame)),
        'candidates_covered': int(table.sabio_has_kinetics.sum()),
        'candidates_total': int(len(table)),
        'by_parameter': (frame.parameter.value_counts().to_dict()
                         if not frame.empty else {}),
        'note': 'Kinetics attach to an EC, not to a protein. Reported, never '
                'folded into master_priority or the convergence count: a '
                'property every member of a class shares cannot discriminate '
                'within it.',
    }
    print(json.dumps(summary, indent=2), flush=True)
    (RES / 'kinetics_summary.json').write_text(
        json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
