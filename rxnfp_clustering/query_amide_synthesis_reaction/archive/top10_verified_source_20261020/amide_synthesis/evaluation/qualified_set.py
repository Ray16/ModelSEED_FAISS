#!/usr/bin/env python
"""Every candidate with independent evidence, uncapped - not just the bench 250.

WHY THIS EXISTS. `shortlist` answers "which proteins should the next bench
round test", and it answers it well: `TARGET` is 250, and per-family,
per-clade and per-genus caps stop one cluster of orthologs from spending the
whole round on a single question. Both are right for a bench round and wrong
for the question actually being asked here, which is how many candidates the
evidence supports at all. Those are different questions and the project only
ever shipped an answer to the first. 40 proteins carrying two independent
lines of evidence appeared in no deliverable, purely because a sibling from
the same family was already in the 250.

THE BAR. `n_discriminating_lines >= 2` from `converged.py`, after the
correction that stopped a curated pocket ligand from re-counting the curated
reaction it came from - the bar is meaningless without it, since it was
1,823 candidates before and 84 after. Family membership
(`lead_family_verdict`) is excluded from the count as a weak axis, so every
line here is a discriminating one. Bench eligibility is applied, because a
candidate nobody can express is not a candidate; the diversity caps and the
tier quotas are NOT.

WHAT TWO LINES IS AND IS NOT. It means two methods that fail differently
agree. It is not validated against nylon-6,6 activity, and with three
measured positives it cannot be - see `validate_against_activity.py`. Ranked
by `master_priority` so the reader can take as many as they have capacity
for, which is the point of not capping it.

Usage:
    python tools/qualified_set.py        # writes results/QUALIFIED.{tsv,md}
"""
from __future__ import annotations
from amide_synthesis.infrastructure.paths import PROJECT

import sys
from pathlib import Path

import pandas as pd


from amide_synthesis.infrastructure.paths import RES
from amide_synthesis.candidate_selection.shortlist import eligible

MIN_LINES = 2

# Resolved per enzyme, in the order a reader wants them.
REPORT = [
    'uniprot', 'gene', 'organism', 'protein_name', 'ec', 'length',
    'n_discriminating_lines', 'evidence_lines', 'best_substrate_distance',
    'substrate_distance_source', 'discovery_route', 'evidence_strength',
    'pfam', 'interpro', 'alphafold', 'pdb', 'catalytic_activity',
    'kinetics', 'on_bench_shortlist', 'master_priority',
]


def qualified(table, evidence, shortlisted):
    """The candidates meeting the bar, bench-eligible, ranked, uncapped."""
    merged = table.merge(
        evidence[['uniprot', 'n_discriminating_lines', 'evidence_lines']],
        on='uniprot', how='left', suffixes=('', '_evidence'))
    lines = pd.to_numeric(merged.get('n_discriminating_lines'),
                          errors='coerce').fillna(0)
    passing = merged[lines >= MIN_LINES]
    checked = eligible(passing)
    out = checked[checked.excluded_because.fillna('').eq('')].copy()
    out['on_bench_shortlist'] = out.uniprot.isin(shortlisted)
    # `best_substrate_distance` is written by `converged`, into its own file;
    # recompute it here from the same three columns rather than depending on
    # that file's presence.
    distances = [pd.to_numeric(out.get(c), errors='coerce')
                 for c in ('substrate_distance_measured',
                           'substrate_distance_own_rhea',
                           'substrate_distance_inherited')
                 if c in out.columns]
    out['best_substrate_distance'] = (
        pd.concat(distances, axis=1).max(axis=1) if distances else pd.NA)
    # Evidence first, then priority: a 4-line candidate outranks a 2-line one
    # whatever the tier weights say, since lines are the bar this file applies.
    return out.sort_values(['n_discriminating_lines', 'master_priority'],
                           ascending=False)


def _gene_breakdown(frame):
    """One line naming the families the off-shortlist rows belong to."""
    genes = (frame.gene.fillna('?').astype(str).str.split().str[0]
             .value_counts())
    shown = ', '.join(f'`{g}` x{n}' if n > 1 else f'`{g}`'
                      for g, n in genes.head(10).items())
    return f'Off-shortlist by gene: {shown}.' if len(genes) else ''


def main():
    table = pd.read_csv(RES / 'master_candidates_enriched.tsv', sep='\t',
                        low_memory=False)
    evidence = pd.read_csv(RES / 'evidence_lines.tsv', sep='\t',
                           low_memory=False)
    shortlist_path = RES / 'shortlist.tsv'
    shortlisted = set(pd.read_csv(shortlist_path, sep='\t',
                                  low_memory=False).uniprot) \
        if shortlist_path.exists() else set()

    out = qualified(table, evidence, shortlisted)
    columns = [c for c in REPORT if c in out.columns]
    out[columns].to_csv(RES / 'QUALIFIED.tsv', sep='\t', index=False)

    unseen = out[~out.on_bench_shortlist]
    lines = [
        '# Candidates with independent evidence, uncapped',
        '',
        f'**{len(out)}** candidates carry at least {MIN_LINES} '
        'DISCRIMINATING lines of evidence and pass bench eligibility. '
        f'**{len(unseen)}** of them are not on the 250-slot bench shortlist, '
        'which caps per family, clade and genus so that one cluster of '
        'orthologs does not spend a whole round on one question. That cap is '
        'right for choosing a bench round and wrong for counting what the '
        'evidence supports, so this file applies the bar and not the cap.',
        '',
        'Two lines means two methods that fail differently agree. Nothing '
        'here is validated against nylon-6,6 activity, and with three '
        'measured positives nothing can be.',
        '',
        '**Read the off-shortlist rows as an ortholog tail, not as new '
        'chemistry.** They are overwhelmingly additional members of families '
        'the shortlist already represents, which is exactly what the '
        'diversity caps are for; what they offer is more targets inside a '
        'family that already looks good, not a new hypothesis. The gene '
        'breakdown below says which families, so that is checkable rather '
        'than a claim.',
        '',
        _gene_breakdown(unseen),
        '',
        f'- {int(out.n_discriminating_lines.ge(3).sum())} at 3 or more lines '
        '(the converged set)',
        f'- families represented: {out.pfam.nunique()}',
        f'- organisms represented: {out.organism.nunique()}',
        '',
        '| # | protein | gene | organism | lines | evidence | distance | '
        'on shortlist |',
        '|---:|---|---|---|---:|---|---:|---|',
    ]
    for n, (_, row) in enumerate(out.iterrows(), 1):
        gene = str(row.get('gene') or '')
        gene = gene.split()[0] if gene and gene != 'nan' else ''
        organism = str(row.get('organism') or '')[:38]
        evidence_text = str(row.get('evidence_lines') or '').replace('+', ', ')
        distance = pd.to_numeric(pd.Series([row.get('best_substrate_distance')]),
                                 errors='coerce').iloc[0]
        shown = f'{distance:.3f}' if pd.notna(distance) else '--'
        seen = 'yes' if row.on_bench_shortlist else '**no**'
        lines.append(
            f'| {n} | `{row.uniprot}` | {gene} | {organism} '
            f'| {int(row.n_discriminating_lines)} | {evidence_text} '
            f'| {shown} | {seen} |')

    (RES / 'QUALIFIED.md').write_text('\n'.join(lines) + '\n')

    print(f'{len(out)} qualified at >= {MIN_LINES} discriminating lines; '
          f'{len(unseen)} of them not on the bench shortlist')
    print(f'written {RES / "QUALIFIED.tsv"} and QUALIFIED.md')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
