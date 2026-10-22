#!/usr/bin/env python
"""Candidates that are known from outside ModelSEED, including measured actives.

The reaction, domain and clade routes all begin from ModelSEED. Two sets of
proteins are relevant to nylon-6,6 amide synthesis and cannot be reached that
way at all:

  assayed    a 50-member amide-synthetase diversity panel with MEASURED
             product titre on the nylon-6,6 L1 diad. These in-house measurements
             belong in the candidate table by definition. They are NCBI
             accessions, invisible to a UniProt reaction lookup. Separate
             literature percent-yield measurements for DdaG/SfaB are documented
             in results/colleague_review_20261017 and must not be pooled with
             this ug/ml readout.

  proxied    proteins carrying an independently computed geometric score.
             Most are already found by the reaction routes; the remainder were
             curated from the literature and are missing here purely because
             nothing links them to a ModelSEED reaction.

Both are read from a separate project, READ-ONLY. Nothing in this module writes
outside this project's results directory.

Provenance is kept sharp. `activity_ugml` is a measurement; everything else in
this pipeline is a prediction, and the two must never be confused. A protein
with measured activity is not scored by substrate distance at all - there is
nothing to predict about it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from . import uniprot
from .paths import RES

SOURCE = Path(os.environ.get(
    'AMIDE_SYNTHETASE_REFERENCE',
    '/homes/rzhu/Enzyme_agent/exp_data/amide_synthetase'))
PANEL = SOURCE / 'assay_data' / 'panel_activity.csv'
PROXY = SOURCE / 'structural_proxies' / 'tables' / 'selected_score_discovery.csv'

# Titre separating the panel's standout producers from the rest.
STRONG_PRODUCER_UGML = float(os.environ.get('STRONG_PRODUCER_UGML', 0.5))


def load_panel(path=PANEL):
    """The assayed diversity panel, with measured nylon-6,6 L1 titre."""
    if not Path(path).exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    out = pd.DataFrame({
        'uniprot': frame.accession,          # NCBI accession; no UniProt mapping
        'gene': frame.id,
        'protein_name': 'amide synthetase (assayed diversity panel)',
        'sequence': frame.sequence,
        'length': frame.length_aa,
        'activity_ugml': frame.activity_N66L1_ugml_avg,
        'activity_std': frame.activity_N66L1_ugml_std,
        'discovery_route': 'assayed',
        'evidence_strength': 'measured_activity',
        'discovery_tier': 'assay_panel',
    })
    out['is_strong_producer'] = out.activity_ugml >= STRONG_PRODUCER_UGML
    return out


def missing_proxy_accessions(path=PROXY, known=()):
    """Structurally scored proteins, minus any caller-supplied exclusions.

    `known` defaults to empty on purpose: see `main`. Excluding proteins that
    another route already found is de-duplication, and de-duplication happens
    once, in `master_table.build`.
    """
    if not Path(path).exists():
        return []
    frame = pd.read_csv(path)
    return sorted(set(frame['id']) - set(known))


def fetch_accessions(accessions, search=uniprot.search):
    """Full UniProt records for a list of accessions."""
    rows = []
    batch = 20
    for start in range(0, len(accessions), batch):
        chunk = accessions[start:start + batch]
        query = ' OR '.join(f'accession:{a}' for a in chunk)
        got, failed, _ = search(query, max_pages=2)
        rows.extend(got)
    return rows


def main():
    from .paths import ensure_directories
    from .master_table import RENAME
    ensure_directories()

    # Deliberately NOT gated on master_candidates.tsv. That file is written
    # AFTER this stage, so reading it here made the output order-dependent: on
    # a second run the proteins looked "already known", were skipped, and the
    # merge then had nothing to add - structural-score coverage silently fell
    # from 36 to 21. Every scored protein is fetched every time; master_table
    # de-duplicates, which is where that belongs.
    panel = load_panel()
    missing = missing_proxy_accessions()
    print(f'assay panel: {len(panel)} proteins '
          f'({int(panel.is_strong_producer.sum()) if len(panel) else 0} strong)',
          flush=True)
    print(f'structurally scored but unreached: {len(missing)}', flush=True)

    proxied = pd.DataFrame()
    if missing:
        rows = fetch_accessions(missing)
        proxied = pd.DataFrame(rows).rename(columns=RENAME)
        if len(proxied):
            proxied['discovery_route'] = 'proxied'
            proxied['evidence_strength'] = 'literature_curated'
            proxied['discovery_tier'] = 'structural_proxy'

    frame = pd.concat([panel, proxied], ignore_index=True)
    frame.to_csv(RES / 'reference_actives.tsv', sep='\t', index=False)

    summary = {
        'assayed': int(len(panel)),
        'strong_producers': int(panel.is_strong_producer.sum()) if len(panel) else 0,
        'proxied_requested': len(missing),
        'proxied_resolved': int(len(proxied)),
        'total': int(len(frame)),
    }
    print(json.dumps(summary, indent=2), flush=True)
    (RES / 'reference_actives_summary.json').write_text(
        json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
