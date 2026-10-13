#!/usr/bin/env python
"""Turn 34,134 candidates into a shortlist someone can actually act on.

A 34k-row table is a search result, not a recommendation. Two things make it
unusable as it stands:

REDUNDANCY. The ranking is evidence-weighted, and evidence is correlated with
how well-studied a family is, so the top of the table collapses onto a few
families: 300 rows hold only 29 families and 27% of them are one family. Most
of that is near-identical orthologs retrieved from the same UniRef50 cluster.
Testing 10 members of one cluster answers one question, not ten.

UNTESTABILITY. Plenty of high-ranking rows are fragments, fusion proteins, or
2,000-residue multi-domain monsters. They are correct retrievals and bad
experiments.

So this stage applies bench constraints and then diversifies, capping how many
slots any one family or UniRef50 clade may take. Diversity is enforced AFTER
ranking and the caps are explicit, so nothing is silently discarded - the full
table remains the source of truth and every exclusion reason is written to its
own column.

The output deliberately includes the 49 assayed proteins regardless of caps.
They carry MEASURED activity on the nylon-6,6 diad, which is not a prediction
and should never be crowded out by one.
"""
from __future__ import annotations

import json
import os
import re

import pandas as pd

from .paths import RES

TARGET = int(os.environ.get('SHORTLIST_SIZE', 250))
# Soluble, single-domain-ish, expressible in a standard host.
MIN_LENGTH = int(os.environ.get('SHORTLIST_MIN_LENGTH', 200))
MAX_LENGTH = int(os.environ.get('SHORTLIST_MAX_LENGTH', 1200))
# Diversity caps: one cluster of orthologs answers one question.
MAX_PER_FAMILY = int(os.environ.get('SHORTLIST_MAX_PER_FAMILY', 12))
MAX_PER_CLADE = int(os.environ.get('SHORTLIST_MAX_PER_CLADE', 3))
MAX_PER_GENUS = int(os.environ.get('SHORTLIST_MAX_PER_GENUS', 8))

ALWAYS_KEEP = ('measured_activity',)

# Slots are allocated per evidence tier, strongest first, and the diversity
# caps apply WITHIN a tier rather than across the whole list.
#
# Applying the caps globally was actively harmful: the strongest evidence is
# concentrated in a few well-studied families, while clade homologs are spread
# thinly across many, so a global family cap spent the list on homologs and
# admitted only 3 of 249 `exact_reaction` proteins. Diversity must not be
# allowed to outrank evidence - it exists to stop ONE family monopolising a
# tier, not to penalise a tier for being concentrated.
TIER_SHARE = (
    ('measured_activity', 1.00),   # uncapped; it is measurement, not prediction
    ('exact_reaction', 0.35),
    ('activity_ec', 0.15),
    ('precomputed_rhea_or_go', 0.10),
    ('literature_curated', 0.05),
    ('name_match', 0.05),
    ('reaction_name_match', 0.03),
    ('family_homolog', 0.15),
    ('clade_homolog', 0.10),
    ('chemistry_neighbour', 0.02),
)


def _text(value):
    """Group key for a possibly-missing field.

    `str(nan or '')` is the STRING "nan", which is truthy, so every row with a
    missing value collapses onto one group key. That silently guillotined the
    strongest tier: all 215 eligible `exact_reaction` rows have no
    `clade_seed`, shared the key "nan", and the per-clade cap of 3 admitted
    three of them.
    """
    if value is None:
        return ''
    if isinstance(value, float) and pd.isna(value):
        return ''
    text = str(value).strip()
    return '' if text.lower() in ('nan', 'none', '<na>') else text


def _genus(organism):
    name = _text(organism)
    return re.split(r'[\s(]', name)[0] if name else ''


def eligible(table):
    """Bench constraints, with the reason for every exclusion recorded."""
    out = table.copy()
    length = pd.to_numeric(out.get('length'), errors='coerce')
    sequence = out.get('sequence').fillna('').astype(str)
    contradicted = out.get('annotation_contradicts_chemistry').fillna(False)

    reasons = pd.Series('', index=out.index)
    reasons = reasons.mask(sequence.str.len() < 1, 'no_sequence')
    reasons = reasons.mask(reasons.eq('') & length.isna(), 'no_length')
    reasons = reasons.mask(reasons.eq('') & (length < MIN_LENGTH), 'too_short')
    reasons = reasons.mask(reasons.eq('') & (length > MAX_LENGTH), 'too_long')
    reasons = reasons.mask(
        reasons.eq('') & contradicted.astype(bool),
        'annotation_contradicts_chemistry')
    # Measured activity overrides every predicted objection except a missing
    # sequence, which would make the row untestable.
    measured = out.get('evidence_strength').isin(ALWAYS_KEEP)
    reasons = reasons.mask(measured & sequence.str.len().gt(0), '')

    out['excluded_because'] = reasons
    return out


def diversify(frame, limit, per_family=MAX_PER_FAMILY,
              per_clade=MAX_PER_CLADE, per_genus=MAX_PER_GENUS):
    """Greedy pass down one tier's ranking, honouring per-group caps.

    Deterministic: the input order is the ranking, and ties were already broken
    by accession upstream.
    """
    family_count, clade_count, genus_count = {}, {}, {}
    chosen = []
    for row in frame.itertuples():
        if len(chosen) >= limit:
            break
        if getattr(row, 'evidence_strength', '') in ALWAYS_KEEP:
            chosen.append(row.Index)
            continue
        family = _text(getattr(row, 'family', ''))
        clade = _text(getattr(row, 'clade_seed', '')) or f'_{row.Index}'
        genus = _genus(getattr(row, 'organism', ''))
        if family and family_count.get(family, 0) >= per_family:
            continue
        if clade_count.get(clade, 0) >= per_clade:
            continue
        if genus and genus_count.get(genus, 0) >= per_genus:
            continue
        family_count[family] = family_count.get(family, 0) + 1
        clade_count[clade] = clade_count.get(clade, 0) + 1
        genus_count[genus] = genus_count.get(genus, 0) + 1
        chosen.append(row.Index)
    return frame.loc[chosen]


def build(table, target=TARGET, shares=TIER_SHARE):
    """Stratify by evidence tier, diversify within each, then backfill."""
    screened = eligible(table)
    pool = screened[screened.excluded_because.eq('')]
    picked = []
    for tier, share in shares:
        tier_rows = pool[pool.evidence_strength.eq(tier)]
        if tier_rows.empty:
            continue
        limit = len(tier_rows) if tier in ALWAYS_KEEP else max(
            1, round(target * share))
        picked.append(diversify(tier_rows, limit))
    out = (pd.concat(picked) if picked else pool.head(0))

    # Backfill any slots the quotas left unused, so a thin tier does not
    # shrink the deliverable.
    if len(out) < target:
        rest = pool.drop(index=out.index)
        out = pd.concat([out, diversify(rest, target - len(out))])
    return out.sort_values('master_priority', ascending=False)


def write_fasta(frame, path):
    with open(path, 'w') as handle:
        for row in frame.itertuples():
            sequence = str(getattr(row, 'sequence', '') or '')
            if not sequence:
                continue
            header = '|'.join(str(x) for x in (
                row.uniprot, getattr(row, 'gene', '') or '-',
                getattr(row, 'evidence_strength', '') or '-',
                f'priority={getattr(row, "master_priority", 0)}'))
            handle.write(f'>{header}\n')
            for start in range(0, len(sequence), 60):
                handle.write(sequence[start:start + 60] + '\n')


def main():
    from .paths import ensure_directories
    ensure_directories()
    table = pd.read_csv(RES / 'master_candidates.tsv', sep='\t', low_memory=False)
    screened = eligible(table)
    short = build(table)

    short.to_csv(RES / 'shortlist.tsv', sep='\t', index=False)
    write_fasta(short, RES / 'shortlist.fasta')

    summary = {
        'candidates_considered': int(len(table)),
        'excluded': screened.excluded_because.value_counts().to_dict(),
        'eligible': int(screened.excluded_because.eq('').sum()),
        'shortlisted': int(len(short)),
        'distinct_families': int(short.family.nunique()),
        'distinct_organisms': int(short.organism.nunique()),
        'by_evidence': short.evidence_strength.value_counts().to_dict(),
        'with_measured_activity': int(
            short.evidence_strength.eq('measured_activity').sum()),
        'with_pdb': int(short.pdb.notna().sum()),
        'self_condensing': int(short.self_condensation.fillna(False).sum()),
        'caps': {'family': MAX_PER_FAMILY, 'clade': MAX_PER_CLADE,
                 'genus': MAX_PER_GENUS},
    }
    print(json.dumps(summary, indent=2), flush=True)
    (RES / 'shortlist_summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    show = ['uniprot', 'gene', 'organism', 'family', 'evidence_strength',
            'substrate_distance_measured', 'struct_S', 'master_priority']
    show = [c for c in show if c in short.columns]
    print('\nTop 20:', flush=True)
    print(short.head(20)[show].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
