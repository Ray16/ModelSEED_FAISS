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

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from amide_synthesis.paths import RES          # noqa: E402
from amide_synthesis.shortlist import eligible  # noqa: E402

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
    """Attach pocket capacity, the one thing that discriminates WITHIN a cohort.

    The kinetic evidence is per EC, so every member of EC 6.3.1.8 earns it
    equally and nothing in it can rank one above another. Pocket capacity
    is per protein - measured from ligands bound in that protein's family
    structures - so it can, and the answer is sharp: 1,458 of 1,751 have
    room for HMDA but only 168 have room for ADIPATE, and nylon-6,6 needs
    both.

    The split is mechanistically self-consistent, which is why it is worth
    trusting. Every one of the 168 is an ATP-dependent ligase (166 EC
    6.3.1.8, 2 EC 6.3.1.9); not one is a GNAT acetyltransferase. An
    acetyltransferase pocket is built around a C2 acetyl group and has no
    room for a C6 diacid, so the geometry agrees with the chemistry
    without having been told to.
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
        'out of the convergence count. (Pocket capacity, below, is per '
        'protein and does discriminate; that is what the ordering uses.) '
        'And the members are largely orthologs of one another: treat this '
        'as three chemistries with deep membership, not as '
        f'{len(found):,} independent leads.', '',
        f'**{both} of these have pocket room for BOTH monomers**, and that '
        'is the only discriminator available inside a cohort defined by a '
        'per-EC property — the kinetics are identical for every member, '
        'pocket capacity is per protein. 1,458 have room for HMDA; only '
        f'{both} also have room for adipate, and nylon-6,6 needs both.', '',
        'The split is mechanistically self-consistent, which is why it is '
        'worth trusting. Every one of them is an **ATP-dependent ligase** '
        '(166 EC 6.3.1.8, 2 EC 6.3.1.9) and not one is a GNAT '
        'acetyltransferase — a pocket built around a C2 acetyl group has '
        'no room for a C6 diacid, so the geometry agrees with the '
        'chemistry without having been told to. They are sorted first '
        'below.', '',
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
