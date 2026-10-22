#!/usr/bin/env python
"""Structural evidence from EXPERIMENTAL structures, no folding required.

The structural stage proper - cofolding the shortlist against adipate, HMDA and
ATP - needs shared-cluster GPUs and was deliberately sequenced after retrieval.
But 82 shortlisted candidates (179 across the whole table) already have
experimental structures: 297 and 825 distinct PDB entries respectively. For
those, crystallography has already answered questions a predicted pose only
estimates, and reading them costs nothing but HTTP.

What this resolves per candidate:

  method / resolution     is the structure good enough to measure geometry in
  bound cofactor          ATP, ADP, AMP, or a non-hydrolysable analogue
                          (AMP-PNP, AMP-PCP) actually present in the model
  bound metal             Mg/Mn, required by every ATP-dependent ligase here
  ligand inventory        what else is in the site

WHY THE COFACTOR MATTERS. The geometric proxy this project corroborates against
scores the carboxylate oxygen attacking ATP's phosphorus - it needs an ATP
position. A structure with ATP or an analogue bound gives that position
experimentally, so those candidates can be scored WITHOUT cofolding, and they
are also the right templates for cofolding the rest.

This is annotation, not prediction: every field is a fact about a deposited
model. Nothing here is a score and nothing enters the ranking.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from amide_synthesis.infrastructure.http import get_text
from amide_synthesis.infrastructure.freshness import tracked_stage
from amide_synthesis.infrastructure.paths import RES

ENTRY = 'https://data.rcsb.org/rest/v1/core/entry/'
THREADS = int(os.environ.get('PDB_THREADS', 6))
TIMEOUT = 60

# Adenosine phosphates and the analogues used to trap a pre-transfer state.
ATP_LIKE = {'ATP', 'ADP', 'AMP', 'ANP', 'ACP', 'AGS', 'APC', 'ADX', 'AP5'}
METALS = {'MG', 'MN', 'ZN', 'CA', 'CO', 'NI'}


def entry_facts(pdb_id, fetch=get_text):
    """Method, resolution and bound components for one PDB entry."""
    text, _, failed = fetch(f'{ENTRY}{pdb_id}', timeout=TIMEOUT)
    if failed or not text:
        return {'pdb_id': pdb_id, 'lookup_failed': True}
    try:
        data = json.loads(text)
    except ValueError:
        return {'pdb_id': pdb_id, 'lookup_failed': True}

    info = data.get('rcsb_entry_info', {}) or {}
    methods = [m.get('method') for m in (data.get('exptl') or []) if m.get('method')]
    resolutions = info.get('resolution_combined') or []
    bound = info.get('nonpolymer_bound_components') or []
    components = {str(c).upper() for c in bound}
    return {
        'pdb_id': pdb_id,
        'method': ';'.join(methods),
        'resolution_a': resolutions[0] if resolutions else None,
        'bound_components': ';'.join(sorted(components)),
        'has_atp_like': bool(components & ATP_LIKE),
        'atp_like_ligand': ';'.join(sorted(components & ATP_LIKE)),
        'has_metal': bool(components & METALS),
        'lookup_failed': False,
    }


def collect(pdb_ids, fetch=get_text, threads=THREADS):
    with ThreadPoolExecutor(max_workers=threads) as pool:
        return pd.DataFrame(list(pool.map(
            lambda p: entry_facts(p, fetch=fetch), pdb_ids)))


def pdb_ids_of(table):
    """{accession: [pdb ids]} for every candidate carrying a structure."""
    out = {}
    column = table.get('pdb')
    if column is None:
        return out
    for accession, value in zip(table.uniprot, column.fillna('')):
        ids = [x.strip().upper() for x in str(value).split(';')
               if x.strip() and x.strip().lower() != 'nan']
        if ids:
            out[accession] = ids
    return out


def summarise_per_protein(by_protein, facts):
    """One row per candidate: its best structure and whether ATP is present."""
    indexed = facts.set_index('pdb_id') if len(facts) else facts
    rows = []
    for accession, ids in by_protein.items():
        present = [i for i in ids if len(indexed) and i in indexed.index]
        if not present:
            continue
        block = indexed.loc[present]
        if isinstance(block, pd.Series):
            block = block.to_frame().T
        with_atp = block[block.has_atp_like.fillna(False).astype(bool)]
        # Prefer a cofactor-bound structure; among those, the best resolution.
        pool = with_atp if len(with_atp) else block
        resolution = pd.to_numeric(pool.resolution_a, errors='coerce')
        best = pool.index[resolution.values.argmin()] if resolution.notna().any() \
            else pool.index[0]
        rows.append({
            'uniprot': accession,
            'n_pdb_entries': len(present),
            'best_pdb': best,
            'best_pdb_resolution_a': pd.to_numeric(
                pool.loc[best, 'resolution_a'], errors='coerce'),
            'best_pdb_method': pool.loc[best, 'method'],
            'pdb_has_atp_like': bool(len(with_atp)),
            'pdb_atp_like_ligand': ';'.join(sorted(
                {x for v in with_atp.atp_like_ligand.fillna('')
                 for x in str(v).split(';') if x})),
            'pdb_has_metal': bool(block.has_metal.fillna(False).astype(bool).any()),
            'pdb_bound_components': ';'.join(sorted(
                {x for v in block.bound_components.fillna('')
                 for x in str(v).split(';') if x}))[:300],
        })
    return pd.DataFrame(rows)


@tracked_stage('pdb')
def main():
    from amide_synthesis.infrastructure.paths import ensure_directories
    ensure_directories()
    enriched = RES / 'master_candidates_enriched.tsv'
    source = enriched if enriched.exists() else RES / 'master_candidates.tsv'
    table = pd.read_csv(source, sep='\t', low_memory=False)

    by_protein = pdb_ids_of(table)
    every_id = sorted({i for ids in by_protein.values() for i in ids})
    print(f'{len(by_protein)} candidates carry {len(every_id)} distinct PDB '
          f'entries', flush=True)

    previous=RES/'pdb_entry_facts.tsv'
    cached=pd.read_csv(previous,sep='\t') if previous.exists() else pd.DataFrame()
    known=set(cached.loc[~cached.lookup_failed.fillna(True).astype(bool),'pdb_id']) if len(cached) else set()
    missing=[i for i in every_id if i not in known]
    fetched=collect(missing) if missing else pd.DataFrame()
    facts=pd.concat([cached[cached.pdb_id.isin(set(every_id)&known)] if len(cached) else cached,fetched],ignore_index=True)
    facts.to_csv(RES / 'pdb_entry_facts.tsv', sep='\t', index=False)
    per_protein = summarise_per_protein(by_protein, facts)
    per_protein.to_csv(RES / 'pdb_evidence.tsv', sep='\t', index=False)

    summary = {
        'candidates_with_structure': int(len(per_protein)),
        'pdb_entries_queried': len(every_id),
        'cached_entry_snapshots_reused':len(set(every_id)&known),
        'entries_fetched_this_run':len(missing),
        'lookups_failed': int(facts.lookup_failed.sum()) if len(facts) else 0,
        'entries_with_atp_like': int(facts.has_atp_like.fillna(False).sum())
                                 if len(facts) else 0,
        'candidates_with_cofactor_bound': int(
            per_protein.pdb_has_atp_like.sum()) if len(per_protein) else 0,
        'candidates_with_metal_bound': int(
            per_protein.pdb_has_metal.sum()) if len(per_protein) else 0,
        'median_best_resolution_a': (
            float(pd.to_numeric(per_protein.best_pdb_resolution_a,
                                errors='coerce').median())
            if len(per_protein) else None),
    }
    print(json.dumps(summary, indent=2), flush=True)
    (RES / 'pdb_evidence_summary.json').write_text(
        json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
