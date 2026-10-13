#!/usr/bin/env python
"""How good is the predicted structure of each shortlisted candidate?

The pipeline resolves an AlphaFold *identifier* for 71% of candidates and then
says nothing about whether the model is any good. For the project's stated next
step - cofolding adipate and HMDA against these proteins - that gap matters:
a mean pLDDT of 45 means the pocket geometry is not there to dock into, and
running an expensive cofold against it wastes the GPU the cluster shares.

So this fetches AlphaFold DB's own confidence numbers: `globalMetricValue`
(mean pLDDT over the chain) plus the fractions of residues at very high, high,
low and very low confidence. All four come from one request per protein.

WHAT THIS DELIBERATELY DOES NOT DO. pLDDT does not enter `master_priority` or
the shortlist selection. Model confidence is not evidence of catalysis - an
enzyme with a mediocre model is still an enzyme - and this project has removed
five score gates that confused ranking with retrieval. It is reported
alongside, exactly as `struct_S` is.

ORDERING. This runs AFTER `shortlist` and writes its OWN files, for the same
reason `enrich` runs after `master`: a stage that rewrites its input in place
makes the pipeline order-dependent, which silently dropped enrichment once
already. `shortlist.tsv` is never modified; the annotated copy is a separate
deliverable.

SCOPE. The shortlist only, which is ~300 requests. Fetching all 28,000
candidates carrying an AlphaFold identifier would be 28,000 requests against a
free public service for information that is only needed where a protein is
actually a bench or cofold target. Raise `MODEL_QUALITY_LIMIT` to widen it.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from .http import get_text
from .paths import RES

API = 'https://alphafold.ebi.ac.uk/api/prediction/{accession}'

# UniProt's official accession syntax, BOTH lengths: the 6-character form and
# the 10-character form (A0A0E1RWL5). Anchored, so the panel's NCBI accessions
# and the proxy table's "-F1" suffixed identifiers are correctly skipped.
ACCESSION = (r'^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]'
             r'|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})$')

# Polite: a small pool against a free public service, with a pause per call.
THREADS = int(os.environ.get('MODEL_QUALITY_THREADS', 4))
PAUSE = float(os.environ.get('MODEL_QUALITY_PAUSE', 0.15))
LIMIT = int(os.environ.get('MODEL_QUALITY_LIMIT', 0))     # 0 = whole shortlist

# AlphaFold's own banding. Below 50 the backbone placement is unreliable, so a
# cofold against it is not worth a GPU slot.
PLDDT_CONFIDENT = 70.0
PLDDT_UNUSABLE = 50.0

FIELDS = ('globalMetricValue', 'fractionPlddtVeryHigh', 'fractionPlddtConfident',
          'fractionPlddtLow', 'fractionPlddtVeryLow')


def fetch_one(accession, fetch=get_text):
    """AlphaFold confidence for one accession, or None if it has no model.

    A missing model and a network failure are different things and are kept
    apart: the first returns None, the second sets `failed` so the caller can
    report it rather than recording a silent zero.
    """
    text, _, failed = fetch(API.format(accession=accession), timeout=45)
    if failed:
        return None, True
    try:
        payload = json.loads(text)
    except Exception:
        return None, False
    if not isinstance(payload, list) or not payload:
        return None, False
    entry = payload[0]
    row = {'uniprot': accession}
    for field in FIELDS:
        value = entry.get(field)
        row[_snake(field)] = (float(value) if isinstance(value, (int, float))
                              else None)
    row['alphafold_entry'] = entry.get('entryId')
    row['alphafold_version'] = entry.get('latestVersion')
    return row, False


def _snake(name):
    out = []
    for char in name:
        if char.isupper():
            out.append('_')
        out.append(char.lower())
    return ''.join(out).lstrip('_').replace('global_metric_value', 'plddt_mean')


def fetch_many(accessions, fetch=get_text, threads=THREADS):
    def one(accession):
        time.sleep(PAUSE)
        return fetch_one(accession, fetch=fetch)

    rows, failures = [], 0
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for row, failed in pool.map(one, accessions):
            if failed:
                failures += 1
            elif row:
                rows.append(row)
    return pd.DataFrame(rows), failures


def classify(frame):
    """Label each model by whether it is worth a cofold slot."""
    if 'plddt_mean' not in frame.columns:
        frame['plddt_mean'] = pd.NA
    plddt = pd.to_numeric(frame.plddt_mean, errors='coerce')
    frame['model_usable_for_cofold'] = plddt >= PLDDT_UNUSABLE
    frame['model_confident'] = plddt >= PLDDT_CONFIDENT
    return frame


def main():
    from .paths import ensure_directories
    ensure_directories()

    source = RES / 'shortlist.tsv'
    if not source.exists():
        raise SystemExit(f'{source.name} not found; run the shortlist stage')
    short = pd.read_csv(source, sep='\t', low_memory=False)

    # Only real UniProt accessions have an AlphaFold entry. The assayed panel
    # is NCBI, and asking for those would be 48 guaranteed misses reported as
    # failures.
    #
    # This must match BOTH accession lengths UniProt issues. A first version
    # allowed only the 6-character form and silently skipped every 10-character
    # one, querying 75 of 301 while reporting no failures - the same silent
    # under-retrieval this project has had to remove five times.
    pattern = ACCESSION
    eligible = short[short.uniprot.astype(str).str.match(pattern)]
    accessions = sorted(eligible.uniprot.dropna().unique())
    if LIMIT:
        accessions = accessions[:LIMIT]
    print(f'shortlist {len(short)}; {len(accessions)} UniProt accessions to query',
          flush=True)

    quality, failures = fetch_many(accessions)
    quality = classify(quality) if len(quality) else pd.DataFrame(
        columns=['uniprot', 'plddt_mean'])
    quality.to_csv(RES / 'alphafold_quality.tsv', sep='\t', index=False)

    # One annotated deliverable, assembled from the sidecar files the later
    # stages write. Each sidecar is keyed by accession and optional: a missing
    # one leaves its columns absent rather than failing the stage.
    annotated = short.merge(quality, on='uniprot', how='left')
    for sidecar in ('curated_properties.tsv',):
        path = RES / sidecar
        if path.exists():
            extra = pd.read_csv(path, sep='\t', low_memory=False)
            overlap = [c for c in extra.columns
                       if c != 'uniprot' and c in annotated.columns]
            annotated = annotated.merge(extra.drop(columns=overlap),
                                        on='uniprot', how='left')
            print(f'merged {sidecar}: '
                  f'{len(extra.columns) - 1 - len(overlap)} columns', flush=True)
    annotated.to_csv(RES / 'shortlist_annotated.tsv', sep='\t', index=False)

    plddt = pd.to_numeric(quality.get('plddt_mean'), errors='coerce') \
        if len(quality) else pd.Series(dtype=float)
    summary = {
        'shortlist': int(len(short)),
        'queried': len(accessions),
        'models_found': int(len(quality)),
        'network_failures': failures,
        'no_model': len(accessions) - int(len(quality)),
        'plddt_mean_median': round(float(plddt.median()), 2) if len(plddt) else None,
        'confident_ge_70': int((plddt >= PLDDT_CONFIDENT).sum()) if len(plddt) else 0,
        'unusable_lt_50': int((plddt < PLDDT_UNUSABLE).sum()) if len(plddt) else 0,
        'thresholds': {'confident': PLDDT_CONFIDENT, 'unusable': PLDDT_UNUSABLE},
        'note': 'pLDDT is reported, never ranked on: model confidence is not '
                'evidence of catalysis.',
    }
    print(json.dumps(summary, indent=2), flush=True)
    (RES / 'alphafold_quality_summary.json').write_text(
        json.dumps(summary, indent=2) + '\n')

    if len(quality):
        show = ['uniprot', 'plddt_mean', 'model_confident',
                'model_usable_for_cofold']
        worst = quality.assign(_p=plddt).sort_values('_p').head(10)[show]
        print('\nLowest-confidence models on the shortlist '
              '(poor cofold targets):', flush=True)
        print(worst.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
