#!/usr/bin/env python
"""Expand each chemistry-validated family into all of its informative members.

WHY. The reaction-anchored routes (Rhea / EC / product name) can only return a
protein that someone has already annotated to a ModelSEED-mappable reaction.
That is a tiny, arbitrary slice of each family: ModelSEED has no reaction ->
enzyme mapping of its own, so coverage reflects curation effort, not biology.
Having established that a FAMILY performs nylon-like chemistry - the screen
proved the bond, and `substrate_distance` scored the substrates - every other
member of that family is a candidate on the same evidence.

This stage therefore inverts the question. Instead of "which protein is
annotated to this reaction", it asks "which proteins belong to a family whose
chemistry we have already validated", and resolves each one as fully as the
UniProt schema allows.

TWO ROUTES, because "the family" means different things at different scales.
Expanding every Pfam that touches a validated protein is wrong in two opposite
ways, and the first version of this stage made both mistakes:

  * CO-OCCURRENCE. A multi-domain protein credits ALL of its domains with its
    chemistry. That is how a peptidoglycan amidase domain (CHAP, a hydrolase)
    and a calcium-channel domain (RyR/IP3R homology) came to be "validated
    amide-forming families". They are passengers on the same polypeptide.

  * SCALE. Enrichment - validated seeds per 1000 family members - separates a
    family whose chemistry is characteristic from one where it is incidental:

        PF27279  DdaG ligase        47.2 / 1000   (127 members)   specific
        PF04183  IucA/IucC NIS       3.5 / 1000   (10k)           specific
        PF00583  GNAT acetyltransf.  0.03 / 1000  (411k)          incidental
        PF00501  AMP-binding         0.01 / 1000  (450k)          incidental

    But low enrichment does NOT mean irrelevant. Mur ligase (0.07) and
    AMP-binding (0.01) are ancient and universal - every bacterium has murC/D/E/F
    and every acyl-CoA synthetase has an adenylation domain - so only a CLADE
    inside them does the chemistry. Expanding the domain would bury the signal;
    the right neighbourhood is sequence homology around the validated protein
    itself.

So: specific families expand by DOMAIN, broad ones expand by CLADE (the seed's
own UniRef50 cluster), and co-occurrence passengers are dropped.

TIERS, not a flat dump. A Pfam family can hold hundreds of thousands of TrEMBL
entries, almost all of them unannotated and untestable. Members are collected in
tiers that each state a reason the entry is worth a slot, and the tier is
recorded so a reader can cut wherever they like:

    reviewed     Swiss-Prot: curated function, catalytic activity, evidence codes
    structure    has a PDB entry: directly usable for the structural stage
    protein_evidence  protein-level existence (PE=1): it has been observed
    diverse      remaining members, capped, for taxonomic breadth

INHERITED, NOT MEASURED. `substrate_distance` is a property of a REACTION. A
family member retrieved here has no ModelSEED reaction of its own, so it
inherits its family's best score. The column is named
`family_substrate_distance` and `evidence_strength` is `family_homolog`
precisely so this is never confused with a per-reaction measurement.
"""
from __future__ import annotations

import json
import functools
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from . import uniprot
from .paths import RES

# Only expand families whose own chemistry cleared the bar.
FAMILY_SD_FLOOR = float(os.environ.get('FAMILY_SD_FLOOR', 0.40))
# Validated seeds per 1000 family members, above which the chemistry is taken to
# be characteristic of the family rather than incidental to it.
MIN_ENRICHMENT_PER_1K = float(os.environ.get('MIN_ENRICHMENT_PER_1K', 1.0))
# A single seed cannot distinguish a catalytic domain from a passenger domain.
MIN_SEEDS = int(os.environ.get('MIN_FAMILY_SEEDS', 2))
# Pages per clade query. Most UniRef50 clusters fit in one, but the large ones
# were truncating at 3, which silently let UniProt's relevance order decide
# which members of a validated protein's own cluster we got to see.
CLADE_PAGES = int(os.environ.get('CLADE_PAGES', 8))
# Polite to a public API; the same cap the reaction routes use.
THREADS = 6

# The signature axis to expand along. Pfam was the only one for a long time,
# and that was a silent limit rather than a decision: EVERY family route here
# grouped proteins by Pfam, so anything Pfam splits differently from the
# curated signature databases was unreachable. InterPro aggregates TIGRFAM/
# NCBIfam, PANTHER, PIRSF and HAMAP, which classify by curated function
# rather than by domain match, and it is enriched far harder - IPR048223
# carries 3 validated seeds among 6 total members (500 per 1k) against the
# best Pfam entry's 47 per 1k.
#
# The enrichment, tiering and passenger-removal logic is identical for both,
# so it is parameterised rather than copied: a fix to the shared logic then
# applies to both axes instead of one.
SIGNATURES = {
    'pfam': {'prefix': 'PF', 'column': 'pfam', 'xref': 'xref:pfam-'},
    'interpro': {'prefix': 'IPR', 'column': 'interpro', 'xref': 'xref:interpro-'},
}

# Page budgets per tier. Raised after 45 queries reported saturation: a
# saturated query means the family had MORE informative members than we
# returned, and which ones you got was decided by UniProt's relevance order
# inside a truncation. The curated tiers are worth paging further; `diverse`
# is deliberately capped because it is breadth, not evidence.
TIERS = (
    ('reviewed', 'reviewed:true', 25),
    ('structure', 'database:pdb', 15),
    ('protein_evidence', 'existence:1', 15),
    ('diverse', 'fragment:false', 3),
)

# EVALUATE THE WHOLE FAMILY, do not sample it.
#
# The tiers above are a sampling scheme: they spend a page budget on the most
# informative members and accept that the rest are never seen. Measured, that
# returned 26,074 of 257,297 members of the validated families - 10.1%. Which
# 90% went missing was decided by UniProt's relevance ordering inside a
# truncation, which is not a criterion at all.
#
# It was defended on the grounds that the shortlist caps any family at 12
# members, so the unseen members could not reach the deliverable anyway. That
# confuses the deliverable with the evaluation: a member that is never
# retrieved is never scored, never annotated, and cannot be found again by
# anyone reading the outputs. For a family whose chemistry the enrichment test
# has already VALIDATED, there is no reason to see a tenth of it.
#
# IMPLEMENTED, THEN TURNED OFF ON PURPOSE - default '0'.
#
# Enumerating the 61 validated family groups in full is 257,297 members. It
# was built, and then the arithmetic settled it against itself: every
# enumerated member INHERITS its family's substrate distance, so 257,297 rows
# would carry about 61 distinct scores between them. That is a membership
# list, not an evaluation. It cannot rank, it cannot prioritise, and it buries
# the 816 reaction-anchored candidates - the only ones whose chemistry was
# actually measured - under three hundred times their number of homologues.
#
# This pipeline is a TARGETED search: chemistry first, then the enzymes that
# do it. Completeness of a homology list is not the same goal and trades
# against it.
#
# Kept behind the flag because the capability is real and the right answer
# depends on the question. If a specific family needs exhaustive coverage -
# for a phylogenetic analysis, or to pick assay targets within one family -
# set ENUMERATE_DOMAIN=1, ideally with the family set narrowed first.
ENUMERATE_DOMAIN = os.environ.get('ENUMERATE_DOMAIN', '0') == '1'
ENUMERATE_PAGE_SIZE = int(os.environ.get('ENUMERATE_PAGE_SIZE', 500))
# Slack over ceil(size/page) so a family that grew since it was sized is still
# finished rather than silently truncated.
ENUMERATE_SLACK_PAGES = int(os.environ.get('ENUMERATE_SLACK_PAGES', 4))


def member_tier(row):
    """Label a member from its OWN record, not from the query that found it.

    Under the tiered scheme the label said which filter retrieved a protein,
    which is an artifact of the sampling order. Enumerating removes the query
    as a variable, so the tier becomes a property of the member.
    """
    if str(row.get('Reviewed', '')).lower().startswith('reviewed'):
        return 'reviewed'
    if str(row.get('PDB', '') or '').strip():
        return 'structure'
    if 'protein level' in str(row.get('Protein existence', '')).lower():
        return 'protein_evidence'
    return 'diverse'


def _enumerate_family(job, search=uniprot.search):
    """Every member of one family, in as few requests as the API allows."""
    pfam, size, xref = job
    pages = max(1, -(-int(size or 0) // ENUMERATE_PAGE_SIZE)) + ENUMERATE_SLACK_PAGES
    rows, failed, saturated = search(f'{xref}{pfam}',
                                     size=ENUMERATE_PAGE_SIZE,
                                     max_pages=pages)
    for row in rows:
        row['discovery_tier'] = member_tier(row)
        row['seed_pfam'] = pfam
    return pfam, 'enumerated', rows, failed, saturated


def seed_families(candidates, floor=FAMILY_SD_FLOOR, signature='pfam'):
    """Pfam accessions worth expanding, with the evidence that justifies each.

    A candidate's `pfam` column lists every domain on the protein and any of
    them could be the catalytic one, so all are credited here and the
    passengers are removed afterwards by `classify_families`, which has the
    family sizes needed to tell a catalytic domain from a co-occurring one.
    """
    config = SIGNATURES[signature]
    out = {}
    for row in candidates.itertuples():
        distance = float(getattr(row, 'substrate_distance', 0.0) or 0.0)
        raw = str(getattr(row, config['column'], '') or '')
        for accession in raw.replace(',', ';').split(';'):
            accession = accession.strip()
            if not accession.startswith(config['prefix']):
                continue
            slot = out.setdefault(accession, {
                'pfam': accession, 'family': getattr(row, 'family', ''),
                'family_substrate_distance': 0.0, 'n_seeds': 0,
                'seed_accessions': [], 'self_condensing': False})
            slot['family_substrate_distance'] = max(
                slot['family_substrate_distance'], distance)
            slot['n_seeds'] += 1
            slot['seed_accessions'].append(row.uniprot)
            slot['self_condensing'] |= bool(
                getattr(row, 'self_condensation', False))
    return {k: v for k, v in out.items()
            if v['family_substrate_distance'] >= floor}


def family_size(pfam, count=None, signature='pfam'):
    """Total UniProtKB entries carrying this signature."""
    if count is None:
        from .uniprot import total_results
        count = total_results
    return count(f"{SIGNATURES[signature]['xref']}{pfam}")


def classify_families(families, sizes,
                      min_enrichment=MIN_ENRICHMENT_PER_1K,
                      min_seeds=MIN_SEEDS):
    """Split seed domains into domain-expandable, clade-only, and passengers.

    `sizes` maps Pfam accession -> total UniProtKB members.

    * `domain`    the chemistry is characteristic of the family; expand it all.
    * `clade`     a real catalytic domain, but so large and ancient that only a
                  sub-clade does this chemistry (Mur ligase, AMP-binding).
                  Expanding the domain would bury the signal, so these are
                  served by sequence homology around the seed instead.
    * `passenger` too few seeds to separate catalysis from co-occurrence
                  (CHAP, RyR/IP3R): dropped.
    """
    verdict = {}
    for pfam, meta in families.items():
        size = sizes.get(pfam, 0) or 0
        enrichment = 1000.0 * meta['n_seeds'] / size if size else 0.0
        record = dict(meta, family_size=size,
                      enrichment_per_1k=round(enrichment, 4))
        if meta['n_seeds'] < min_seeds:
            record['route'] = 'passenger'
        elif enrichment >= min_enrichment:
            record['route'] = 'domain'
        else:
            record['route'] = 'clade'
        verdict[pfam] = record
    return verdict


def clade_seeds(families):
    """Accessions whose own homologues should be fetched, not their domain.

    A protein in a `clade` family is the only reliable description of its own
    functional neighbourhood, so it is expanded individually.
    """
    out = {}
    for meta in families.values():
        if meta.get('route') != 'clade':
            continue
        for accession in meta['seed_accessions']:
            slot = out.setdefault(accession, {
                'seed': accession, 'family': meta['family'],
                'pfam': meta['pfam'],
                'family_substrate_distance': 0.0})
            slot['family_substrate_distance'] = max(
                slot['family_substrate_distance'],
                meta['family_substrate_distance'])
    return out


def _fetch_clade(job, search=uniprot.search):
    """Members of the seed's UniRef50 cluster - its functional neighbourhood."""
    accession, meta, pages = job
    rows, failed, saturated = search(
        f'uniref_cluster_50:UniRef50_{accession}', max_pages=pages)
    for row in rows:
        row['discovery_tier'] = 'clade_uniref50'
        row['seed_pfam'] = meta['pfam']
        row['clade_seed'] = accession
    return accession, 'clade_uniref50', rows, failed, saturated


def _fetch_tier(job, search=uniprot.search):
    pfam, tier, filt, pages, xref = job
    query = f'{xref}{pfam} AND {filt}'
    rows, failed, saturated = search(query, max_pages=pages)
    for row in rows:
        row['discovery_tier'] = tier
        row['seed_pfam'] = pfam
    return pfam, tier, rows, failed, saturated


def expand(families, search=uniprot.search, threads=THREADS,
           signature='pfam', enumerate_all=None):
    """Retrieve members for every family. Returns (frame, report).

    `enumerate_all` takes the whole family; the tiered path samples it under a
    page budget and is kept for comparison. See ENUMERATE_DOMAIN.
    """
    xref = SIGNATURES[signature]['xref']
    if enumerate_all is None:
        enumerate_all = ENUMERATE_DOMAIN
    if enumerate_all:
        jobs = [(pfam, meta.get('family_size', 0), xref)
                for pfam, meta in families.items()]
        fetch = _enumerate_family
    else:
        jobs = [(pfam, tier, filt, pages, xref)
                for pfam in families
                for tier, filt, pages in TIERS]
        fetch = _fetch_tier
    collected, report = {}, []
    with ThreadPoolExecutor(max_workers=threads) as pool:
        results = pool.map(lambda j: fetch(j, search), jobs)
        for pfam, tier, rows, failed, saturated in results:
            report.append({'pfam': pfam, 'tier': tier, 'returned': len(rows),
                           'failed': failed, 'saturated': saturated})
            for row in rows:
                accession = row.get('Entry')
                if not accession:
                    continue
                # First tier to claim an accession keeps it: tiers are ordered
                # by how much they tell you about the protein.
                collected.setdefault(accession, row)

    frame = pd.DataFrame(list(collected.values()))
    if frame.empty:
        return frame, pd.DataFrame(report)

    meta = frame.seed_pfam.map(lambda p: families.get(p, {}))
    # Label a member by the SIGNATURE THAT SEEDED IT, resolved to its own
    # curated name - not by the seed protein's label.
    #
    # Taking the seed's label was wrong in a way that looked fine: a seed
    # protein is labelled by the first name in its own signature list, so
    # P0ADT5 and P33222 - "Putative acid--amine ligase YgiC/YjfC (EC 6.3.1.-)",
    # seeded from PF03738 and genuinely on-chemistry - were presented as "CHAP
    # domain", a peptidoglycan hydrolase, because IPR016185 carried that name
    # historically. The chemistry and the ranking were unaffected (both key on
    # the signature) but every family NAME and every count of distinct
    # families read off the column was suspect.
    from .families import pfam_names
    resolved = pfam_names(sorted({p for p in frame.seed_pfam.dropna().unique()}))
    frame['family'] = [
        resolved.get(signature) or m.get('family', '')
        for signature, m in zip(frame.seed_pfam, meta)]
    frame['family_substrate_distance'] = [
        m.get('family_substrate_distance', 0.0) for m in meta]
    frame['family_self_condensing'] = [
        m.get('self_condensing', False) for m in meta]
    frame['evidence_strength'] = 'family_homolog'
    return frame, pd.DataFrame(report)


def main():
    from .paths import ensure_directories
    ensure_directories()
    candidates = pd.read_csv(RES / 'candidates_ranked.tsv', sep='\t')

    # `candidates_ranked.tsv` keeps only a display subset and has no InterPro
    # column, so seeding the second axis from it silently produced nothing -
    # the stage ran, reported "0 signatures seeded" for InterPro, and looked
    # fine. The full stage-04 record has it, and that stage runs before this
    # one, so it is joined rather than re-fetched.
    details = RES / 'enzymes_uniprot.tsv'
    if details.exists() and 'interpro' not in candidates.columns:
        full = (pd.read_csv(details, sep='\t', low_memory=False)
                .drop_duplicates('uniprot').set_index('uniprot'))
        if 'interpro' in full.columns:
            candidates['interpro'] = candidates.uniprot.map(full.interpro)
    missing = ('interpro' not in candidates.columns
               or not candidates.interpro.notna().any())
    if missing:
        print('WARNING: no InterPro column available; the second signature '
              'axis will seed nothing', flush=True)

    # Both signature axes, scored by the same enrichment test. Pfam groups by
    # domain match; InterPro aggregates the curated databases (TIGRFAM/NCBIfam,
    # PANTHER, PIRSF, HAMAP) that group by function, and splits families Pfam
    # merges. Expanding only Pfam was a silent limit, not a decision.
    families = {}
    for signature in SIGNATURES:
        seeded = seed_families(candidates, signature=signature)
        if not seeded:
            continue
        sizer = functools.partial(family_size, signature=signature)
        with ThreadPoolExecutor(max_workers=THREADS) as pool:
            sizes = dict(zip(seeded, pool.map(sizer, seeded)))
        seeded = classify_families(seeded, sizes)
        for accession, meta in seeded.items():
            meta['signature'] = signature
        print(f'{signature}: {len(seeded)} signatures seeded', flush=True)
        families.update(seeded)

    # The second axis contributes to the DOMAIN route only, and that is a
    # measured decision rather than caution. The clade route is keyed on the
    # seed PROTEIN's UniRef50 cluster, not on the signature, and 580 of the
    # 614 seed proteins are shared between the axes - so letting InterPro feed
    # it re-expanded the same clusters and took clade_homolog from 28,029 to
    # 44,061 rows. That is ortholog count on the weakest evidence tier, not
    # new chemistry, and the shortlist caps any family at 12 members so it
    # cannot reach the deliverable either. The domain route is different: it
    # is keyed on the signature, and InterPro splits families Pfam merges, so
    # it genuinely adds families (23 -> 81).
    CLADE_AXIS = 'pfam'
    for accession, meta in families.items():
        if meta.get('signature') != CLADE_AXIS and meta.get('route') == 'clade':
            meta['route'] = 'clade_duplicate_axis'

    for route in ('domain', 'clade', 'passenger', 'clade_duplicate_axis'):
        chosen = {k: v for k, v in families.items() if v['route'] == route}
        print(f'\n{route}: {len(chosen)} families', flush=True)
        for pfam, meta in sorted(chosen.items(),
                                 key=lambda kv: -kv[1]['enrichment_per_1k']):
            print(f'  {pfam}  sd={meta["family_substrate_distance"]:.3f}  '
                  f'seeds={meta["n_seeds"]:>3}  size={meta["family_size"]:>7}  '
                  f'enrich={meta["enrichment_per_1k"]:>7.3f}/1k  '
                  f'{meta["family"][:40]}', flush=True)

    frames, reports = [], []
    for signature in SIGNATURES:
        chosen = {k: v for k, v in families.items()
                  if v['route'] == 'domain' and v.get('signature') == signature}
        if not chosen:
            continue
        part, part_report = expand(chosen, signature=signature)
        if len(part):
            part['signature_db'] = signature
            frames.append(part)
        if len(part_report):
            part_report['signature_db'] = signature
            reports.append(part_report)
    frame = (pd.concat(frames, ignore_index=True) if frames
             else pd.DataFrame(columns=['Entry']))
    report = (pd.concat(reports, ignore_index=True) if reports
              else pd.DataFrame())
    # One protein can be reached by both axes; the first keeps it, and
    # `signature_db` records which.
    if len(frame) and 'Entry' in frame.columns:
        frame = frame.drop_duplicates('Entry').reset_index(drop=True)

    clades = clade_seeds(families)
    print(f'\nclade route: {len(clades)} seed proteins', flush=True)
    clade_rows, clade_report = {}, []
    if clades:
        jobs = [(a, m, CLADE_PAGES) for a, m in clades.items()]
        with ThreadPoolExecutor(max_workers=THREADS) as pool:
            for accession, tier, rows, failed, saturated in pool.map(
                    _fetch_clade, jobs):
                clade_report.append({'pfam': clades[accession]['pfam'],
                                     'tier': f'clade:{accession}',
                                     'returned': len(rows), 'failed': failed,
                                     'saturated': saturated})
                for row in rows:
                    entry = row.get('Entry')
                    if entry:
                        row['family'] = clades[accession]['family']
                        row['family_substrate_distance'] = clades[
                            accession]['family_substrate_distance']
                        row['family_self_condensing'] = False
                        row['evidence_strength'] = 'clade_homolog'
                        clade_rows.setdefault(entry, row)

    if clade_rows:
        clade_frame = pd.DataFrame(list(clade_rows.values()))
        # A protein already retrieved by the domain route keeps that record.
        if not frame.empty:
            clade_frame = clade_frame[~clade_frame.Entry.isin(set(frame.Entry))]
        frame = pd.concat([frame, clade_frame], ignore_index=True)
        report = pd.concat([report, pd.DataFrame(clade_report)],
                           ignore_index=True)

    known = set(candidates.uniprot)
    if not frame.empty:
        frame['already_a_candidate'] = frame.Entry.isin(known)
        frame = frame.sort_values(
            ['family_substrate_distance', 'discovery_tier', 'Entry'],
            ascending=[False, True, True])
    frame.to_csv(RES / 'family_expansion.tsv', sep='\t', index=False)
    report.to_csv(RES / 'family_expansion_report.tsv', sep='\t', index=False)

    summary = {
        'families_seeded': len(families),
        'route_domain': sum(v['route'] == 'domain' for v in families.values()),
        'route_clade': sum(v['route'] == 'clade' for v in families.values()),
        'route_passenger_dropped': sum(v['route'] == 'passenger'
                                       for v in families.values()),
        'clade_seed_proteins': len(clades),
        'members_retrieved': int(len(frame)),
        'new_accessions': int((~frame.already_a_candidate).sum()) if len(frame) else 0,
        'by_tier': (frame.discovery_tier.value_counts().to_dict()
                    if len(frame) else {}),
        'by_evidence': (frame.evidence_strength.value_counts().to_dict()
                        if len(frame) else {}),
        'saturated_queries': int(report.saturated.sum()) if len(report) else 0,
        'failed_queries': int(report.failed.sum()) if len(report) else 0,
    }
    print('\n' + json.dumps(summary, indent=2), flush=True)
    (RES / 'family_expansion_summary.json').write_text(
        json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
