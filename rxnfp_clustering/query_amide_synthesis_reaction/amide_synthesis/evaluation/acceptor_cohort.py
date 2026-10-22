#!/usr/bin/env python
"""Candidates whose EC has MEASURED kinetics on the acceptor half-reaction.

HMDA is an acceptor. Every chemistry score in this project reasons about how
much a native substrate resembles adipate + HMDA, and `substrate_walk`
narrowed that to enzymes measured across a chain-length series. Both ask more
than the evidence needs to be useful: a SINGLE published parameter for
acylating a free primary amine on an aliphatic polyamine is already the
target chemistry, measured. Requiring a walk on top of it demanded
demonstrated chain tolerance as well, and that extra demand is what kept this
cohort out of sight.

Three ECs qualify out of the 273 our candidates carry, so the signal is
selective rather than a property everything shares - and two of the three are
EC 6.3, ATP-dependent amide ligases, the same bond-forming chemistry as the
nylon diad:

    6.3.1.8   glutathionylspermidine synthase   Km spermidine 7.4 mM
    6.3.1.9   trypanothione synthase            Km spermidine 92 uM - 1.4 mM
    2.3.1.57  diamine N-acetyltransferase       Km spermidine 50-130 uM

WHY THIS IS A SEPARATE FILE AND NOT AN EVIDENCE AXIS. The signal attaches to
an EC, so every member of EC 6.3.1.8 earns it equally and it cannot rank
within the class - the same objection that kept curated ATP binding out of
the convergence count and keeps `sabio_*` out of `master_priority`. What it
can do is define a cohort, which is what this writes.

THE HONEST SHAPE OF THE COHORT. 1,776 candidates carry one of the three ECs
and 1,751 survive bench eligibility, 1,104 of them prokaryotic, against 15 on
the bench shortlist. That near-total survival is recent: the bench length
floor used to sit at 200 residues and excluded 1,479 of them, because EC
2.3.1.57's GNAT acetyltransferases run 150-200 residues. The floor was
measured against the length histogram, found to be cutting through a dense
mode at 170-179 rather than sitting in a gap, and moved to 120.

They are mostly orthologs of each other: the evidence is per EC, so read
this as three chemistries with deep membership, not 1,751 independent
leads. The full ranking ships as TSV so the number taken is the reader's
decision; the table below is a readable slice.

Usage:
    python tools/acceptor_cohort.py     # writes results/ACCEPTOR_COHORT.{tsv,md}
"""
from __future__ import annotations
from amide_synthesis.infrastructure.paths import PROJECT

import sys
from pathlib import Path

import pandas as pd


from amide_synthesis.infrastructure.paths import RES
from amide_synthesis.candidate_selection.shortlist import eligible

# Free primary-amine acceptors on an aliphatic backbone. Exact names: both
# regex directions failed on this data (see `substrate_walk`).
ACCEPTORS = (
    'Spermidine', 'Spermine', 'Putrescine', 'Cadaverine', 'Agmatine',
    '1,3-Diaminopropane', 'Ethylenediamine', 'Hexamethylenediamine',
    '1,6-Diaminohexane', 'Hexane-1,6-diamine',
)
SHOWN = 25


def acceptor_ecs(kinetics):
    """ECs with at least one measured parameter on a free polyamine amine.

    Ki values are excluded. A Ki says a molecule BINDS, which for a
    bisubstrate analogue says nothing about turnover - the EC 2.3.1.87 trap
    recorded in `substrate_walk`, where an apparent C2-C6 acyl series is
    entirely transition-state-mimic inhibition constants.
    """
    hit = kinetics[kinetics.species.isin(ACCEPTORS)
                   & kinetics.parameter.ne('Ki')]
    return hit.groupby('ec').agg(
        enzyme_name=('enzyme_name', lambda s: next(iter(s.dropna()), '')),
        n_parameters=('value', 'size'),
        acceptors=('species', lambda s: ','.join(sorted(set(s)))),
    ).reset_index()


def with_pocket(found, capacity):
    """Attach pocket capacity: 168 of 1,751 have room for both monomers.

    1,458 have room for HMDA; only 168 also have room for ADIPATE, and
    nylon-6,6 needs both. Every one of the 168 is an ATP-dependent ligase
    (166 EC 6.3.1.8, 2 EC 6.3.1.9) and not one is a GNAT
    acetyltransferase.

    TWO THINGS I FIRST CLAIMED ABOUT THIS AND HAD TO WALK BACK, both
    checked by looking at the spans rather than the booleans.

    It is NOT a per-protein discriminator, which is how it was first
    described. All 166 passing members of EC 6.3.1.8 carry the SAME spans
    - amine 19, acid 9 - because the span is inherited from a ligand bound
    in the family's structures, not measured on each protein. It separates
    families, and so mostly separates ECs: 0% of EC 2.3.1.57 passes, 87%
    of 6.3.1.8, 11% of 6.3.1.9. The only within-EC split is 24 of 190 in
    6.3.1.8, and those 24 have spans of ZERO - no data, not a small pocket.

    And the GNAT exclusion is weaker evidence than it looks. Those 1,542
    proteins do have real amine-side data (spans 3, 5, 8, 12), but
    `acid_span` is 0 for every single one. That is CONSISTENT with a
    CoA-dependent acetyltransferase having no free-acid subsite - its
    donor arrives as a thioester - but it is not proof of one, because it
    is equally consistent with no acid-bearing ligand ever having been
    crystallised in that family. Absence of a measurement is being read as
    absence of a pocket. The conclusion survives on the mechanism; it is
    not independently confirmed by the geometry, which is what I first
    wrote.
    """
    if capacity is None or capacity.empty:
        found['room_for_both'] = False
        return found
    columns = [c for c in ('uniprot', 'room_for_hmda', 'room_for_adipate',
                           'amine_span', 'acid_span') if c in capacity.columns]
    merged = found.merge(capacity[columns].drop_duplicates('uniprot'),
                         on='uniprot', how='left')
    def _flag(name):
        return (pd.to_numeric(merged.get(name), errors='coerce')
                .fillna(0).astype(bool))
    merged['room_for_both'] = _flag('room_for_hmda') & _flag('room_for_adipate')
    return merged


def cohort(master, ecs):
    """Bench-eligible candidates in those ECs, ranked for a bench."""
    wanted = set(ecs)
    rows = master.ec.fillna('').astype(str).str.split(';').apply(
        lambda parts: any(p.strip() in wanted for p in parts))
    found = eligible(master[rows])
    found = found[found.excluded_because.fillna('').eq('')].copy()
    if found.empty:
        return found
    lineage = found.lineage.fillna('').astype(str)
    found['prokaryote'] = lineage.str.contains('Bacteria|Archaea', case=False)
    found['is_reviewed'] = (found.reviewed.fillna('').astype(str).str.lower()
                            .str.startswith('reviewed'))
    found['has_pdb'] = found.pdb.notna()
    found['has_model'] = found.alphafold.notna()
    found['matched_ec'] = found.ec.fillna('').astype(str).apply(
        lambda v: next((p.strip() for p in v.split(';')
                        if p.strip() in wanted), ''))
    # Expressibility first, then structural evidence, then resemblance as a
    # tie-break only - resemblance is the axis this whole cohort fails.
    # Pocket room for BOTH monomers outranks everything else: it is the
    # only per-protein discriminator available inside a cohort defined by
    # a per-EC property.
    capacity_path = RES / 'pocket_capacity.tsv'
    capacity = (pd.read_csv(capacity_path, sep='\t', low_memory=False)
                if capacity_path.exists() else None)
    found = with_pocket(found, capacity)
    return found.sort_values(
        ['room_for_both', 'is_reviewed', 'has_pdb', 'prokaryote',
         'has_model'], ascending=False)


def main():
    kinetics_path = RES / 'sabio_kinetics.tsv'
    if not kinetics_path.exists():
        raise SystemExit('run the `kinetics` stage first')
    kinetics = pd.read_csv(kinetics_path, sep='\t', low_memory=False)
    kinetics['species'] = kinetics.species.fillna('').astype(str)
    ecs = acceptor_ecs(kinetics)
    master = pd.read_csv(RES / 'master_candidates_enriched.tsv', sep='\t',
                         low_memory=False)
    found = cohort(master, ecs.ec)

    columns = [c for c in ('uniprot', 'matched_ec', 'gene', 'organism',
                           'length', 'prokaryote', 'is_reviewed', 'has_pdb',
                           'pdb', 'alphafold', 'room_for_both',
                           'room_for_hmda', 'room_for_adipate',
                           'amine_span', 'acid_span',
                           'substrate_distance_measured',
                           'master_priority') if c in found.columns]
    found[columns].to_csv(RES / 'ACCEPTOR_COHORT.tsv', sep='\t', index=False)

    shortlisted = set(pd.read_csv(RES / 'shortlist.tsv', sep='\t',
                                  low_memory=False).uniprot) \
        if (RES / 'shortlist.tsv').exists() else set()
    pro = int(found.prokaryote.sum())
    on_list = int(sum(u in shortlisted for u in found.uniprot))
    both = int(found.room_for_both.sum()) if 'room_for_both' in found else 0
    lines = [
        '# The acceptor-chemistry cohort', '',
        'HMDA is an **acceptor**. These are the candidates whose EC has at '
        'least one *measured* kinetic parameter for acylating a free primary '
        'amine on an aliphatic polyamine — the target half-reaction, '
        'established experimentally rather than inferred from annotation.',
        '', '| EC | enzyme | parameters | acceptors measured |',
        '|---|---|---:|---|',
    ]
    for _, row in ecs.iterrows():
        lines.append(f'| {row.ec} | {row.enzyme_name} | {row.n_parameters} '
                     f'| {row.acceptors} |')
    lines += [
        '', f'**{len(found):,} candidates**, {pro:,} of them prokaryotic and '
        f'of expressible length, against **{on_list}** on the bench '
        'shortlist.', '',
        'Two things a reader should hold onto. The **kinetic** evidence is '
        'per EC, so every member earns it equally and it alone cannot rank '
        'one above another — which is why this is a cohort file and not an '
        'evidence axis, the same objection that keeps curated ATP binding '
        'out of the convergence count. Pocket capacity, below, cuts the '
        'cohort hard and is what the ordering uses, but it turns out to '
        'be family-inherited rather than per-protein, so it mostly '
        'separates ECs too. And the members are largely orthologs of one '
        'another: treat this '
        'as three chemistries with deep membership, not as '
        f'{len(found):,} independent leads.', '',
        f'**{both} of these have pocket room for BOTH monomers.** 1,458 '
        f'have room for HMDA; only {both} also have room for adipate, and '
        'nylon-6,6 needs both. This is the sharpest cut available on the '
        'cohort, and the two paragraphs after it say exactly how much '
        'weight it will bear — less than it first appears.', '',
        'Every one of them is an **ATP-dependent ligase** (166 EC 6.3.1.8, '
        '2 EC 6.3.1.9) and not one is a GNAT acetyltransferase. Read that '
        'carefully, because it is weaker than it looks: the GNATs have '
        'real amine-side spans (3, 5, 8, 12) but `acid_span` is **0 for '
        'all 1,542**, and absence of a measurement is not absence of a '
        'pocket. It is consistent with a CoA-dependent acetyltransferase '
        'having no free-acid subsite — its donor arrives as a thioester — '
        'but no acid-bearing ligand may simply have been crystallised in '
        'that family. The conclusion rests on the mechanism; the geometry '
        'does not independently confirm it.', '',
        'Nor is this a per-protein signal, though it is computed per '
        'protein. All 166 passing members of EC 6.3.1.8 carry identical '
        'spans (amine 19, acid 9) because the span is inherited from a '
        'ligand bound in the family\'s structures. It separates families, '
        'hence mostly ECs — 0% of 2.3.1.57, 87% of 6.3.1.8, 11% of '
        '6.3.1.9 — and the 24 that fail inside 6.3.1.8 have spans of zero, '
        'which is missing data rather than a small pocket. They are sorted '
        'first below regardless, since a family-level reason is still a '
        'reason.', '',
        'Ki values are excluded when selecting the ECs. A Ki says a molecule '
        'binds, which for a bisubstrate analogue says nothing about turnover '
        '— see the EC 2.3.1.87 near-miss in `SUBSTRATE_WALK.md`.', '',
        f'Full ranking in `ACCEPTOR_COHORT.tsv`; the first {SHOWN} follow.',
        '', '| protein | EC | gene | organism | aa | both monomers | reviewed '
        '| PDB | prok |',
        '|---|---|---|---|---:|---|---|---|---|',
    ]
    for _, row in found.head(SHOWN).iterrows():
        gene = str(row.get('gene') or '').split(' ')[0]
        gene = '' if gene == 'nan' else gene
        lines.append(
            f'| `{row.uniprot}` | {row.matched_ec} | {gene} '
            f'| {str(row.get("organism") or "")[:34]} '
            f'| {int(row.length) if pd.notna(row.get("length")) else "?"} '
            f'| {"**yes**" if row.get("room_for_both") else "no"} '
            f'| {"yes" if row.is_reviewed else "no"} '
            f'| {"yes" if row.has_pdb else "no"} '
            f'| {"yes" if row.prokaryote else "no"} |')
    (RES / 'ACCEPTOR_COHORT.md').write_text('\n'.join(lines) + '\n')

    print(f'{len(ecs)} ECs with measured acceptor kinetics: '
          f'{", ".join(ecs.ec)}')
    print(f'{len(found):,} bench-eligible candidates, {pro:,} prokaryotic; '
          f'{sum(u in shortlisted for u in found.uniprot)} shortlisted')
    print(f'written {RES / "ACCEPTOR_COHORT.tsv"} and ACCEPTOR_COHORT.md')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
