#!/usr/bin/env python
"""Does every per-protein analysis still cover every candidate?

THE DEFECT THIS CATCHES. A stage that ran BEFORE a route existed never covers
that route's proteins, and nothing says so. The `actives` route added 822
homologs of the three measured producers; every per-protein artifact in
`results/` had been written earlier, so each one silently stopped one route
short:

  pocket_capacity.tsv       54,199 rows - 0 of the 822
  ligand_evidence.tsv       54,199 rows - 0 of the 822
  curated_properties.tsv    54,150 rows - 0 of the 822
  member_similarity.tsv     46,254 rows - 0 of the 822

Each file looked healthy on its own: tens of thousands of rows, no blanks, no
error. The gap is only visible against the CURRENT candidate set, which is
why this is a check and not a lesson. The consequence was not cosmetic - those
are the structural axes, so the entire measured-data hypothesis sat at one
evidence line each partly BY CONSTRUCTION, because the analyses that could
have given it a second line had never been run on it.

This is the same root cause as the stale-input guard in `shortlist`: an
artifact out of date with respect to its input. That guard compares two files;
this compares every per-protein file against the candidate set.

MISSING ROWS ARE NOT ALWAYS A BUG. A stage that legitimately applies to a
subset - `pdb_evidence` only to proteins with a crystal structure,
`adenylation_geometry` only to those with a bound nucleotide - is expected to
be short, and padding it would be worse. A share alone cannot tell the two
apart, and the first draft of this tool got that wrong: it read
`member_similarity` at 84% as a stage that had missed a route, when that stage
is a within-family tie-break and correctly has nothing to say about a protein
admitted by its own reaction or name.

So a file is either declared SCOPED, with the scope written down and checked
for completeness inside it, or it is expected to cover everything. The scope
is the stronger statement: `member_similarity` is not merely allowed to be
short, it must cover all 46,254 family-route candidates and does. Files too
small to be either are reported for the reader and never fail.

Usage:
    python tools/audit_coverage.py       # exits non-zero if a route was missed
"""
from __future__ import annotations
from amide_synthesis.infrastructure.paths import PROJECT

import sys
from pathlib import Path

import pandas as pd


from amide_synthesis.candidate_selection.curated_properties import ACCESSION
from amide_synthesis.infrastructure.paths import RES

# Above this share of its expected scope, a file is understood to be an
# analysis that should be complete. Below it, the stage is selective by
# design and is only reported.
BROAD_COVERAGE = 0.5

# Stages that apply to part of the candidate set on purpose. The scope is
# declared here so it can be CHECKED rather than inferred from a share: each
# of these must cover its scope completely.
SCOPED = {
    # A within-family tie-break. A protein admitted by its own reaction,
    # name, or homology to a measured active has no family members to be
    # discriminated against, so the family routes are the whole scope.
    'member_similarity.tsv': (
        'candidates admitted by family membership',
        lambda frame: frame.discovery_route.isin(('domain', 'clade')),
    ),
    # Queries UniProt by accession. The 48-member assayed panel is keyed by
    # NCBI accessions (OPY85079.1) and AlphaFold model ids (A0A2D9ZTX9-F1),
    # which have no UniProt record to curate; all 48 of the shortfall are
    # that panel and none is a valid accession.
    'curated_properties.tsv': (
        'candidates with a real UniProt accession',
        lambda frame: frame.uniprot.astype(str).str.fullmatch(ACCESSION),
    ),
}

# Not per-protein analyses: inputs, per-reaction tables, or summaries whose
# accession column means something other than "a candidate we analysed".
SKIP = {
    'master_candidates.tsv',
    'master_candidates_enriched.tsv',
    'reactions_without_enzymes.tsv',
    'queries.tsv',
    'query_tsne_coords.tsv',
    'enzyme_families.tsv',
    'family_evaluation.tsv',
    'family_evaluation_by_signature.tsv',
    'family_expansion.tsv',
    'family_expansion_report.tsv',
    'family_evaluation_by_signature.tsv',
    'pdb_entry_facts.tsv',
    'ec_vote.tsv',
}


def candidates_frame(res):
    """The table every per-protein stage is expected to have seen."""
    enriched = res / 'master_candidates_enriched.tsv'
    master = res / 'master_candidates.tsv'
    source = enriched if enriched.exists() else master
    frame = pd.read_csv(source, sep='\t', low_memory=False,
                        usecols=['uniprot', 'discovery_route'])
    return frame.dropna(subset=['uniprot']), source.name


def expected(frame, name):
    """The accessions `name` should cover, and what that scope is called."""
    if name in SCOPED:
        label, predicate = SCOPED[name]
        return set(frame[predicate(frame)].uniprot.astype(str)), label
    return set(frame.uniprot.astype(str)), 'every candidate'


def audit(res):
    frame, source = candidates_frame(res)
    broad, selective = [], []
    for path in sorted(res.glob('*.tsv')):
        if path.name in SKIP:
            continue
        try:
            head = pd.read_csv(path, sep='\t', low_memory=False, nrows=0)
        except Exception:
            continue
        if 'uniprot' not in head.columns:
            continue
        rows = pd.read_csv(path, sep='\t', low_memory=False,
                           usecols=['uniprot'])
        scope, label = expected(frame, path.name)
        covered = set(rows.uniprot.dropna().astype(str)) & scope
        share = len(covered) / max(len(scope), 1)
        row = (path.name, len(covered), len(scope) - len(covered), share,
               label)
        (broad if share >= BROAD_COVERAGE else selective).append(row)
    return frame, source, broad, selective


def main():
    frame, source, broad, selective = audit(RES)
    print(f'{len(frame):,} candidates in {source}\n')

    if selective:
        print('SELECTIVE BY DESIGN (<50% of scope) - reported, never failed:')
        for name, covered, _, share, _ in sorted(selective,
                                                 key=lambda r: -r[3]):
            print(f'   {name:34s} {covered:>7,}  {share:6.1%}')

    print('\nANALYSES EXPECTED TO BE COMPLETE:')
    for name, covered, missing, share, label in sorted(broad,
                                                       key=lambda r: r[3]):
        flag = f'  MISSING {missing:,}' if missing else ''
        scope = '' if label == 'every candidate' else f'   [{label}]'
        print(f'   {name:34s} {covered:>7,}  {share:6.1%}{flag}{scope}')

    incomplete = [row for row in broad if row[2] > 0]
    if incomplete:
        print('\nAn analysis short of its scope is a stage that ran before a '
              'route existed. Re-run it:')
        for name, _, missing, _, label in incomplete:
            print(f'   {name:34s} {missing:,} of {label} never analysed')
        return 1
    print('\nevery analysis covers its full scope.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
