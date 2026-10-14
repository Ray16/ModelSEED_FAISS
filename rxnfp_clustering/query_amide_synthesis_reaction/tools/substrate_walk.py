#!/usr/bin/env python
"""Which enzymes have been MEASURED on a chain-length series, not just a substrate?

THE BLIND SPOT THIS ADDRESSES. Every chemistry score in this project asks how
much an enzyme's native substrate resembles adipate + HMDA. That is the right
question and it has a hole in it: an enzyme whose native substrate is distant
but which has published activity on progressively LONGER homologues is
exactly what an engineering campaign wants, and nothing here can see it.

Pantoate--beta-alanine ligase (PanC, EC 6.3.2.1) is the case that exposed it.
It forms an amide between a carboxylate and the free amine of an omega-amino
acid - mechanistically the nylon bond - and SABIO-RK holds kinetics for it
across the donor series:

    C3  beta-alanine          Km   0.8 mM     kcat/Km 4.3e6 M^-1 s^-1
    C4  4-aminobutanoate      Km 335   mM     kcat/Km 0.81  M^-1 s^-1
    C5  5-aminovaleric acid   Km  72   mM     kcat/Km 0.42  M^-1 s^-1
    C6  6-aminohexanoate      never measured

Our own scoring puts PanC at substrate distance 0.425, far below the 0.75
floor, and it reaches none of the shortlist's evidence axes. It is correct
about resemblance and blind to the walk.

WHAT THE NUMBERS DO AND DO NOT SUPPORT. The cliff is C3 -> C4: five orders of
magnitude. C4 -> C5 is nearly flat, and Km actually IMPROVES (335 -> 72 mM).
A flat C4->C5 segment is consistent with C6 being comparable to C5, but that
is an extrapolation from two points and this tool does not pretend otherwise;
it reports the series and the gap. Note also that 0.42 M^-1 s^-1 is a feeble
catalyst - this is a starting scaffold for engineering, not a producer, and
the honest claim is "measured, weak, non-zero activity on a C5 omega-amino
acid by an enzyme that makes the right bond".

REGEXES LIE IN BOTH DIRECTIONS HERE, so the series are matched on exact
names. Searching `adipate` returned 2-oxoadipate and L-2-aminoadipate, which
are alpha-substituted lysine-pathway metabolites and not adipate at all;
searching `5-?aminovalerate` returned nothing because SABIO spells it
"5-Aminovaleric acid". The first error invents a hit, the second hides one.

Usage:
    python tools/substrate_walk.py      # writes results/SUBSTRATE_WALK.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from amide_synthesis.paths import RES   # noqa: E402

# Exact SABIO substrate names, by carbon count. Exact because both regex
# failure modes were observed on this very data - see the module docstring.
OMEGA_AMINO = {
    2: ['Glycine'],
    3: ['beta-Alanine', 'Beta-Alanine'],
    4: ['4-Aminobutanoate', 'gamma-Aminobutyric acid'],
    5: ['5-Aminovaleric acid', '5-Aminopentanoate'],
    6: ['6-Aminohexanoate', '6-Aminohexanoic acid', 'epsilon-Aminocaproate'],
    7: ['7-Aminoheptanoate'],
}
DIACID = {
    4: ['Succinate'],
    5: ['Glutarate'],
    6: ['Adipate', 'Hexanedioate'],
    7: ['Pimelate'],
    8: ['Suberate'],
}
SERIES = {'omega-amino acid donor': OMEGA_AMINO, 'linear dicarboxylate': DIACID}
# The target: nylon-6,6 needs C6 on both sides; nylon-6 needs a C6 omega-amino
# acid. Anything measured at C5 or beyond is one step away.
TARGET_CARBONS = 6


def walks(kinetics, series):
    """Per EC, which rungs of a series carry a measured parameter."""
    names = {name: n for n, values in series.items() for name in values}
    frame = kinetics.copy()
    frame['carbons'] = frame.species.map(names)
    hit = frame.dropna(subset=['carbons'])
    if hit.empty:
        return pd.DataFrame(columns=['ec', 'enzyme_name', 'rungs',
                                     'longest', 'reaches_target'])
    rows = []
    for ec, group in hit.groupby('ec'):
        rungs = sorted({int(c) for c in group.carbons})
        rows.append({
            'ec': ec,
            'enzyme_name': next((n for n in group.enzyme_name.dropna()), ''),
            'rungs': ','.join(f'C{c}' for c in rungs),
            'n_rungs': len(rungs),
            'longest': max(rungs),
            'reaches_target': max(rungs) >= TARGET_CARBONS,
            'uniprot_ids': ';'.join(sorted(
                {a for v in group.uniprot_ids.dropna().astype(str)
                 for a in v.replace('(', '').replace(')', '').split(';')
                 if a and a[0].isalpha() and len(a) >= 6})),
        })
    out = pd.DataFrame(rows)
    return out.sort_values(['n_rungs', 'longest'], ascending=False)


# How many representatives to name per walk-EC. The point is a handful a
# bench can act on, not the whole class: EC 6.3.2.1 alone holds 1,390
# candidates, and admitting them wholesale is the mass-expansion this
# project refuses everywhere else.
PER_EC = 6


def representatives(master, ec, limit=PER_EC):
    """The best handful of candidates in a walk-EC, with the rule explicit.

    Ranked on expressibility and evidence a bench can use rather than on
    `master_priority`, which is tuned for substrate resemblance - the very
    axis these proteins fail. Reviewed entries first (curated, so the EC
    assignment is trustworthy), then a solved structure, then an AlphaFold
    model, then substrate distance as the tie-break.
    """
    def _col(name, default=None):
        if name in master.columns:
            return master[name]
        return pd.Series([default] * len(master), index=master.index)

    rows = _col('ec', '').fillna('').astype(str).str.split(';').apply(
        lambda parts: ec in [p.strip() for p in parts])
    found = master[rows].copy()
    if found.empty:
        return found
    reviewed = _col('reviewed', '').fillna('').astype(str).str.lower()
    found['is_reviewed'] = reviewed[rows].str.startswith('reviewed')
    found['has_pdb'] = _col('pdb')[rows].notna()
    found['has_model'] = _col('alphafold')[rows].notna()
    lineage = _col('lineage', '').fillna('').astype(str)
    found['prokaryote'] = lineage[rows].str.contains('Bacteria|Archaea',
                                                     case=False)
    found['distance'] = pd.to_numeric(
        _col('substrate_distance_measured', 0.0)[rows],
        errors='coerce').fillna(0.0)
    # A protein nobody can express is not a representative.
    length = pd.to_numeric(_col('length', None)[rows], errors='coerce')
    found = found[length.between(100, 1200, inclusive='both').fillna(False)]
    return found.sort_values(
        ['is_reviewed', 'has_pdb', 'has_model', 'distance'],
        ascending=False).head(limit)


def main():
    path = RES / 'sabio_kinetics.tsv'
    if not path.exists():
        raise SystemExit('run the `kinetics` stage first')
    kinetics = pd.read_csv(path, sep='\t', low_memory=False)
    kinetics['species'] = kinetics.species.fillna('').astype(str)

    lines = ['# Measured substrate walks, not substrate resemblance', '',
             'Every chemistry score here asks how much an enzyme\'s native '
             'substrate resembles adipate + HMDA. This asks a different '
             'question: which enzymes have **published kinetics on more than '
             'one rung of a chain-length series**, and how far up it they '
             'have been taken. An enzyme whose native substrate is distant '
             'but whose activity has been walked toward C6 is what an '
             'engineering campaign wants, and no axis in this project can '
             'see it.', '']
    for label, series in SERIES.items():
        found = walks(kinetics, series)
        multi = found[found.n_rungs >= 2] if not found.empty else found
        lines += [f'## {label}', '']
        if multi.empty:
            lines += ['No enzyme has measured parameters on two or more '
                      'rungs of this series.', '']
            continue
        lines += ['| EC | enzyme | rungs measured | longest | reaches C6 |',
                  '|---|---|---|---:|---|']
        for _, row in multi.iterrows():
            lines.append(
                f'| {row.ec} | {row.enzyme_name or "—"} | {row.rungs} '
                f'| C{row.longest} | {"yes" if row.reaches_target else "**no**"} |')
        lines.append('')
        # The rungs alone read as a success story. The VALUES are what stop
        # a reader concluding these enzymes already work.
        best = multi.iloc[0]
        detail = kinetics[kinetics.ec.eq(best.ec)
                          & kinetics.species.isin(
                              [n for v in series.values() for n in v])]
        if not detail.empty:
            lines += [f'Measured values for the longest walk, EC {best.ec}:',
                      '', '| rung | substrate | parameter | value | unit |',
                      '|---|---|---|---:|---|']
            rung = {name: n for n, v in series.items() for name in v}
            for _, r in (detail.sort_values('species')
                         [['species', 'parameter', 'value', 'unit']]
                         .drop_duplicates().iterrows()):
                lines.append(f'| C{rung.get(r.species, "?")} | {r.species} '
                             f'| {r.parameter} | {r.value} | {r.unit} |')
            lines += ['', '**Read the values, not the rungs - they say the '
                      'opposite of what the rung count suggests.** The '
                      'tempting story is that activity declines with chain '
                      'length and flattens by C5, so C6 is worth measuring. '
                      'The C2 row kills it: glycine is 0.6 M^-1 s^-1, '
                      'against C4 at 0.81 and C5 at 0.42. Everything that is '
                      'not beta-alanine sits at the same ~0.5 M^-1 s^-1 '
                      'floor whether it is SHORTER or longer, so the flat '
                      'C4-C5 segment is non-specific background, not '
                      'tolerance extending along the chain. PanC is sharply '
                      'specific for C3; it is a four-rung walk that '
                      'demonstrates specificity, not flexibility, and it is '
                      'not evidence that C6 would work.', '',
                      'Carnosine synthase is the better lead ON THIS AXIS '
                      'even with fewer rungs: C3 Km 1.49-1.8 mM against C4 '
                      'Km 8.8 mM is a five-fold degradation, not five '
                      'orders of magnitude. No kcat is published for it, so '
                      'catalytic efficiency cannot be compared and this is '
                      'a Km argument only.', '',
                      'It is worse on every other axis a bench cares about, '
                      'and the trade-off should be read whole. **All 326 '
                      'carnosine synthases held here are eukaryotic** - '
                      'there is not one bacterial or archaeal member - and '
                      'they run 827-951 residues. PanC is 283-309 residues, '
                      'bacterial, reviewed, and has solved structures. So '
                      'the enzyme with the better substrate tolerance is '
                      'the harder expression target by a wide margin, and '
                      'the one that is trivial to express is the one whose '
                      'walk shows specificity rather than flexibility.', '']
    # Who, concretely, a bench could order. Named per walk-EC rather than
    # merged into the converged set: this is a DIFFERENT axis - measured
    # chain-series kinetics - and folding it in would imply agreement
    # between methods that have not agreed.
    master_path = RES / 'master_candidates_enriched.tsv'
    if master_path.exists():
        master = pd.read_csv(master_path, sep='\t', low_memory=False)
        walk_ecs = []
        for series in SERIES.values():
            found = walks(kinetics, series)
            if not found.empty:
                walk_ecs += list(found[found.n_rungs >= 2].ec)
        lines += ['## Candidates in these ECs', '',
                  'All three walk-ECs are near-invisible in the existing '
                  'deliverables: 1,720 candidates between them, 8 on the '
                  'bench shortlist, **0 in QUALIFIED.md** - because every '
                  'evidence axis scores substrate RESEMBLANCE, which is '
                  'exactly what these proteins fail (carnosine synthase '
                  '0.667 and PanC 0.425, against a 0.75 floor).',
                  '',
                  f'Listing the best {PER_EC} per EC rather than the class: '
                  'EC 6.3.2.1 alone holds 1,390 candidates and admitting '
                  'them wholesale is the mass-expansion refused everywhere '
                  'else here. Ranked on what a bench needs - reviewed, then '
                  'a solved structure, then a model, then substrate '
                  'distance - not on `master_priority`, which is tuned for '
                  'the resemblance axis these proteins fail.', '']
        for ec in dict.fromkeys(walk_ecs):
            chosen = representatives(master, ec)
            if chosen.empty:
                continue
            lines += [f'### EC {ec}', '',
                      '| protein | gene | organism | len | reviewed | PDB '
                      '| prokaryote |',
                      '|---|---|---|---:|---|---|---|']
            for _, row in chosen.iterrows():
                gene = str(row.get('gene') or '').split(' ')[0]
                gene = '' if gene == 'nan' else gene
                lines.append(
                    f'| `{row.uniprot}` | {gene} '
                    f'| {str(row.get("organism") or "")[:36]} '
                    f'| {int(row.length) if pd.notna(row.get("length")) else "?"} '
                    f'| {"yes" if row.is_reviewed else "no"} '
                    f'| {"yes" if row.has_pdb else "no"} '
                    f'| {"yes" if row.prokaryote else "**no**"} |')
            lines.append('')

    (RES / 'SUBSTRATE_WALK.md').write_text('\n'.join(lines) + '\n')

    for label, series in SERIES.items():
        found = walks(kinetics, series)
        multi = found[found.n_rungs >= 2] if not found.empty else found
        print(f'{label}: {len(multi)} enzymes measured on >= 2 rungs')
        for _, row in multi.iterrows():
            print(f'   EC {row.ec:10s} {row.rungs:16s} '
                  f'longest C{row.longest}  {row.enzyme_name[:40]}')
    print(f'written {RES / "SUBSTRATE_WALK.md"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
