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

from .freshness import tracked_stage
from .paths import RES

TARGET = int(os.environ.get('SHORTLIST_SIZE', 250))
# Soluble, single-domain-ish, expressible in a standard host.
#
# THE FLOOR WAS 200 AND THAT WAS MEASURABLY THE WRONG PLACE. Its stated
# purpose is expressibility, but a 170-residue single-domain protein is
# MORE expressible than a 600-residue one, not less, so a floor cannot be
# measuring what the comment claims. Nor was it excluding fragments: of
# 55,125 candidates exactly ONE is flagged `(Fragment)`.
#
# What it was doing is visible in the length histogram. There is a dense
# mode at 170-179 (2,885 candidates) and 200 cuts through its descending
# shoulder - 925 candidates in the 190-199 bin against 364 in 200-209. It
# sliced a coherent population of small single-domain enzymes, most of them
# GNAT-fold acyltransferases, which is the family holding the measured
# acceptor chemistry. This project defended its 0.75 chemistry floor on the
# grounds that it sat in a REAL gap in a discrete distribution; by the same
# standard 200 does not, and 120 does - 39 candidates in 120-129, 162 below
# 100 across the whole tail.
#
# Measured effect of the move: bench-eligible 41,457 -> 49,069, the
# acceptor cohort 309 -> 1,751 (prokaryotic 205 -> 1,104), and the
# evidence-backed QUALIFIED set 82 -> 84. It matters enormously for the
# cohort and almost not at all for the converged work, because there
# evidence binds rather than length.
MIN_LENGTH = int(os.environ.get('SHORTLIST_MIN_LENGTH', 120))
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
    ('experimental_direct', 0.20),
    ('curated_direct', 0.08),
    ('homology_supported_direct', 0.05),
    ('computational_direct', 0.07),
    ('unverified_reaction_search', 0.0),
    ('exact_reaction', 0.35),
    ('activity_ec', 0.15),
    # The other hypothesis. Given a real share rather than a token one: it is
    # the only tier anchored in measured activity, and a bench round that
    # tests only annotation-derived candidates tests half the question.
    ('homolog_of_measured_active', 0.15),
    ('precomputed_rhea_or_go', 0.10),
    ('literature_curated', 0.05),
    ('name_match', 0.05),
    # UniProt names the protein for the target function but no reaction is
    # known for it. Same evidence weight as `name_match`, so the same share.
    ('enzyme_name_match', 0.05),
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


def _column(frame, name, default=''):
    """Always a Series aligned to `frame`.

    `DataFrame.get(name)` returns None when the column is absent, so every
    `.fillna`/`.isin` call below raises AttributeError on a table that happens
    not to carry it. The same trap was already fixed in `master_table` and
    `enrich`; this copy was still live.
    """
    if name in frame.columns:
        return frame[name]
    return pd.Series([default] * len(frame), index=frame.index)


def eligible(table):
    """Bench constraints, with the reason for every exclusion recorded."""
    out = table.copy()
    length = pd.to_numeric(_column(out, 'length', None), errors='coerce')
    sequence = _column(out, 'sequence').fillna('').astype(str)
    contradicted = _column(
        out, 'annotation_contradicts_chemistry', False).fillna(False)

    # A transmembrane protein is a poor bench target for the same reason a
    # 3,000-residue one is: it will not express solubly in a standard host.
    # This module claimed "soluble, single-domain-ish, expressible" from the
    # start but only ever checked LENGTH, so 2 of 301 shortlisted proteins
    # carried a Transmembrane keyword - including a membrane-anchored
    # fatty-acid amide HYDROLASE.
    #
    # Only the transmembrane keyword excludes. A signal peptide does not: a
    # secreted soluble protein expresses perfectly well, often better. The
    # bare "Membrane" keyword can mean peripheral association, which is not a
    # solubility problem, so it is left alone too.
    keywords = _column(out, 'keywords').fillna('').astype(str)
    transmembrane = keywords.str.contains('Transmembrane', case=False)
    out['has_signal_peptide'] = keywords.str.contains('Signal', case=False)

    reasons = pd.Series('', index=out.index)
    reasons = reasons.mask(_column(out,'evidence_strength').eq('unverified_reaction_search'),
                           'unverified_reaction_mapping')
    reasons = reasons.mask(sequence.str.len() < 1, 'no_sequence')
    reasons = reasons.mask(reasons.eq('') & transmembrane, 'transmembrane')
    reasons = reasons.mask(reasons.eq('') & length.isna(), 'no_length')
    # A SOLVED STRUCTURE OVERRIDES THE LENGTH HEURISTIC, both ends. The
    # length bounds are a proxy for "expressible in a standard host"; a
    # crystal structure is direct evidence of the thing the proxy
    # estimates, because somebody expressed, purified and crystallised the
    # protein. Direct evidence beats a proxy, which is the same rule this
    # function already applies when measured activity overrides every
    # predicted objection.
    #
    # It is a narrow exemption, not a relaxation: 48 candidates are
    # excluded as too_short while holding a structure, and they include the
    # GNAT acetyltransferases of EC 2.3.1.57 - 161-172 residues, with up to
    # four structures each - which are the acceptor chemistry's best
    # expressible representatives. The bound itself is unchanged for
    # everything without one.
    # `.notna()` is wrong here: this column holds '' as often as NaN, and
    # an empty string is not a structure. The existing bench-filter fixture
    # caught exactly that.
    solved = (_column(out, 'pdb').fillna('').astype(str).str.strip()
              .ne('').fillna(False))
    reasons = reasons.mask(
        reasons.eq('') & (length < MIN_LENGTH) & ~solved, 'too_short')
    reasons = reasons.mask(
        reasons.eq('') & (length > MAX_LENGTH) & ~solved, 'too_long')
    reasons = reasons.mask(
        reasons.eq('') & contradicted.astype(bool),
        'annotation_contradicts_chemistry')
    # UniProt's own curated family statement, a third axis independent of the
    # protein name and the EC that `master_table` already checks, and
    # independent of the Pfam/InterPro grouping the family routes rely on.
    # Computed by the `properties` stage; absent if that stage has not run.
    prose = _column(out, 'family_prose_contradicts_chemistry', False)
    reasons = reasons.mask(
        reasons.eq('') & pd.to_numeric(prose, errors='coerce')
        .fillna(0).astype(bool),
        'family_prose_contradicts_chemistry')
    # The family was judged on its own validated seeds' chemistry and failed -
    # see family_evaluation.py. A reaction-anchored protein is exempt: its own
    # reaction was screened, which outranks its family's verdict.
    in_rejected = _column(out, 'in_rejected_family', False)
    reasons = reasons.mask(
        reasons.eq('') & pd.to_numeric(in_rejected, errors='coerce')
        .fillna(0).astype(bool),
        'family_judged_reject')
    # Measured activity overrides every predicted objection except a missing
    # sequence, which would make the row untestable.
    measured = _column(out, 'evidence_strength').isin(ALWAYS_KEEP)
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
    """Stratify by evidence tier, diversify within each, then backfill.

    A tier absent from `shares` gets no slots, and the backfill below cannot
    rescue it because that only fires when the quotas UNDERfill the target.
    When the protein-name route was added its tier was missing here, so all
    3,469 of its candidates were silently unshortlistable - the quotas already
    overfilled the target, the backfill never ran, and the summary looked
    normal. An unallocated tier is a curation mistake, not a ranking decision,
    so it fails loudly rather than disappearing.
    """
    screened = eligible(table)
    pool = screened[screened.excluded_because.eq('')]
    unallocated = (set(pool.evidence_strength.dropna().unique())
                   - {tier for tier, _ in shares})
    if unallocated:
        raise ValueError(
            f'evidence tiers with no slot allocation: {sorted(unallocated)}. '
            'Add them to TIER_SHARE with a deliberate share; leaving one out '
            'silently drops every candidate it carries.')
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


def choose_source(res):
    """The table to shortlist from, refusing a stale enriched copy.

    The enriched table is preferred when the annotation stage has run; it is
    a superset, never a replacement. But only when it was built from the
    CURRENT master. `enrich` needs UniProt, so `--offline` drops it, and
    re-running `master` alone then leaves the enriched table one generation
    behind - whereupon this stage would silently shortlist from superseded
    flags. That happened: a disqualifier fix cleared 38 rows in `master` and
    the shortlist did not move, because it was still reading the pre-fix
    enriched copy. Fail instead of guessing; the alternative is a wrong
    shortlist that looks right.
    """
    enriched = res / 'master_candidates_enriched.tsv'
    master = res / 'master_candidates.tsv'
    if enriched.exists() and master.exists() and (
            enriched.stat().st_mtime < master.stat().st_mtime):
        raise SystemExit(
            f'{enriched.name} is older than {master.name}: it was built from '
            'a superseded master table. Re-run the `enrich` stage (it needs '
            'UniProt, so not under --offline), or delete the enriched file '
            'to shortlist from the master directly.')
    return enriched if enriched.exists() else master


@tracked_stage('shortlist')
def main():
    from .paths import ensure_directories
    ensure_directories()
    source = choose_source(RES)
    table = pd.read_csv(source, sep='\t', low_memory=False)
    print(f'shortlisting from {source.name}', flush=True)

    # Experimental structural facts, where crystallography has already
    # answered what a predicted pose only estimates.
    pdb_facts = RES / 'pdb_evidence.tsv'
    if pdb_facts.exists():
        facts = pd.read_csv(pdb_facts, sep='\t')
        table = table.merge(facts, on='uniprot', how='left')

    # Family verdicts. A protein admitted by family membership inherits its
    # family's score, so a family judged `reject` on its own seeds' chemistry
    # should not be contributing bench candidates at all. This PRUNES rather
    # than broadens: it is the lever that makes the table more targeted.
    # Member-specific similarity, used ONLY to order rows that are already
    # tied. 8,593 of 9,365 family homologs sit in a tie of >= 100 rows and
    # 36,274 of 36,889 clade homologs do, so without this the rows that reach
    # the bench list are chosen by accession string. It cannot change which
    # tier or which score a row has - see member_similarity.py.
    similarity = RES / 'member_similarity.tsv'
    if similarity.exists():
        scored = pd.read_csv(similarity, sep='\t', low_memory=False)
        if 'member_similarity' in scored.columns:
            table = table.merge(scored[['uniprot', 'member_similarity']],
                                on='uniprot', how='left')

    verdicts = RES / 'family_evaluation.tsv'
    if verdicts.exists():
        judged = pd.read_csv(verdicts, sep='\t', low_memory=False)
        rejected = set()
        for row in judged[judged.verdict.eq('reject')].itertuples():
            for accession in str(getattr(row, 'signatures', '') or '').split(';'):
                accession = accession.strip()
                if accession:
                    rejected.add(accession)
        if rejected:
            pattern = '|'.join(re.escape(a) for a in sorted(rejected))
            signature_text = (_column(table, 'pfam').fillna('').astype(str)
                              + ';' + _column(table, 'interpro')
                              .fillna('').astype(str))
            # Exempt, for the same reason `reaction` is exempt: the route's
            # evidence does not come from the family's annotated chemistry.
            #
            # 819 of the 822 measured-active homologs carry PF13193 /
            # IPR045851, which belong to "ATP-dependent AMP-binding enzyme" -
            # a family `famval` rejects, correctly, because its product is a
            # thioester that cannot polycondense. Applying that verdict here
            # would suppress the ENTIRE measured-data hypothesis on the
            # strength of the chemistry hypothesis, and the two are in direct
            # conflict: three members of that rejected family are the only
            # proteins known to make the nylon bond.
            #
            # A family verdict is a statement about annotated chemistry. This
            # route's evidence is measured activity by descent, which is
            # exactly the evidence that contradicts the annotation, so the
            # verdict has no authority over it.
            exempt = _column(table, 'discovery_route').isin(
                ('reaction', 'active_homolog'))
            table['in_rejected_family'] = (
                signature_text.str.contains(pattern, regex=True) & ~exempt)
            print(f'family verdicts: {len(rejected)} signatures in rejected '
                  f'families; {int(table.in_rejected_family.sum())} candidates '
                  'flagged', flush=True)

    # Curated properties, for the family-prose cross-check above. Only the
    # flag is joined: the prose itself belongs in the annotated deliverable,
    # not in the selection step.
    properties = RES / 'curated_properties.tsv'
    if properties.exists():
        extra = pd.read_csv(properties, sep='\t', low_memory=False)
        column = 'family_prose_contradicts_chemistry'
        if column in extra.columns:
            table = table.merge(extra[['uniprot', column]],
                                on='uniprot', how='left')
    # Re-sort so the tie-break applies. The primary key is unchanged, so no
    # row moves relative to a row with a different priority.
    if 'member_similarity' in table.columns:
        table = table.sort_values(
            ['master_priority', 'member_similarity', 'uniprot'],
            ascending=[False, False, True]).reset_index(drop=True)
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
        'with_experimental_structure': int(short.get(
            'best_pdb', pd.Series(dtype=object)).notna().sum()),
        'with_cofactor_bound_structure': int(short.get(
            'pdb_has_atp_like', pd.Series(dtype=bool)).fillna(False).sum()),
        'self_condensing': int(short.self_condensation.fillna(False).sum()),
        'caps': {'family': MAX_PER_FAMILY, 'clade': MAX_PER_CLADE,
                 'genus': MAX_PER_GENUS},
    }
    print(json.dumps(summary, indent=2), flush=True)
    (RES / 'shortlist_summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    show = ['uniprot', 'gene', 'organism', 'family', 'evidence_strength',
            'substrate_distance_measured', 'struct_S', 'best_pdb',
            'pdb_has_atp_like', 'master_priority']
    show = [c for c in show if c in short.columns]
    print('\nTop 20:', flush=True)
    print(short.head(20)[show].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
