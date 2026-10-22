#!/usr/bin/env python
"""Measure recall against lists defined OUTSIDE this pipeline.

WHY A SEPARATE TOOL. Every control in `tests/` encodes an assumption this
project already held, so the suite can be entirely green while a whole class
of candidate is missing. On 2026-10-13, 192 controls passed and every reported
number reproduced from the output files - and the pipeline had found 0 of the
26 proteins UniProt literally names "Amide synthetase". Internal tests cannot
find that; an external list can.

Two audits, because they fail differently.

1. NAMED LISTS. Proteins UniProt names for the target function, with nylon
   hydrolases as a negative control. Low recall here is a straightforward hole:
   the pipeline should contain anything an external curator calls an amide
   synthetase. This is the audit that found the gap, which turned out to be
   the clade route sampling PF00501 (~450,000 members) at 767.

2. EC CLASSES. Recall against EC 6.3.1/6.3.2/6.3.3, which DEFINE the
   carboxylate + amine bond. The percentage here is NOT a quality measure, and
   reading it as one would repeat the mistake this project rejected elsewhere
   ("chemically true, biologically meaningless"). Measured 2026-10-13, the
   aggregate looks alarming and is not:

       EC 6.3.1.-   1,440 of 74,714    1.9%
       EC 6.3.2.-  13,127 of 240,141   5.5%

   Decomposing by sub-EC explains all of it, and the explanation has three
   parts:

   a. HIGH recall where the sub-EC is small and specific - 6.3.2.11 carnosine
      synthase 88%, 6.3.1.8 glutathionylspermidine 69%, 6.3.1.17 100%. Where
      a ModelSEED reaction maps to the EC, the pipeline enumerates it.
   b. LOW recall on huge sub-ECs - 6.3.2.17 folylpolyglutamate synthetase is
      577 of 24,413. This is sampling, not loss: those are near-identical
      orthologs, the shortlist caps any family at 12, so enumerating all
      24,413 would change nothing downstream while inflating the count. The
      same argument rejected `cyanophycin synthetase` as a name query.
   c. ZERO recall on classes the screen CORRECTLY refuses. Verified one by one
      against UniProt's own reaction text rather than from memory:

          6.3.1.1  6.3.1.5  6.3.1.14   acceptor is NH4+ -> terminal amide,
                                       scored 0.0 by the hard gate because a
                                       terminal amide cannot polycondense
          6.3.1.19 6.3.1.20            acceptor is a carrier PROTEIN
          6.3.2.26 6.3.2.40            NRPS; carrier-bound thioester
          6.3.1.15                     NovL-type aryl amide synthetase; a real
                                       amide synthetase, but its product is a
                                       terminal amide

   So there is no actionable EC gap. That is a negative result, and it is
   recorded here so the aggregate percentage is not rediscovered and
   misreported as a hole.

A JUDGEMENT THIS EXPOSES, for the record. (c) rests on the polycondensation
hard gate, which is correct for the nylon-6,6 target as stated. If the goal
ever broadens to amide bonds generally, 6.3.1.15 and its relatives become
candidates rather than exclusions, and that is a decision to make explicitly
rather than by changing a weight.

Usage:
    python tools/recall_audit.py            # both audits
    python tools/recall_audit.py names      # named lists only
    python tools/recall_audit.py ec         # EC classes only
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from amide_synthesis.paths import RES                      # noqa: E402
from amide_synthesis.uniprot import search, total_results  # noqa: E402

# Externally defined positives: UniProt's own name for the target function.
POSITIVE_NAMES = ('amide synthetase', 'amide bond synthetase',
                  'amide synthase', 'amide ligase')

# Externally defined negatives: nylon-DEGRADING hydrolases. Finding these
# would mean the chemistry screen is admitting bond breaking as bond forming.
NEGATIVE_ACCESSIONS = ('Q14T92', 'Q1EPR4', 'Q1EPR5', 'Q79F77')

EC_PREFIXES = ('6.3.1', '6.3.2', '6.3.3')
SUB_EC_LIMIT = 50


def load_table():
    for name in ('master_candidates_enriched.tsv', 'master_candidates.tsv'):
        path = RES / name
        if path.exists():
            return pd.read_csv(path, sep='\t', low_memory=False), name
    raise SystemExit('no master candidate table found; run the pipeline first')


def audit_names(table):
    have = set(table.uniprot.dropna())
    print('EXTERNAL NAMED LISTS')
    print(f'{"name":28s} {"external":>9s} {"ours":>6s} {"recall":>8s}')
    worst = []
    for name in POSITIVE_NAMES:
        rows, _, _ = search(f'protein_name:"{name}"', max_pages=5,
                            fields='accession')
        accs = [r['Entry'] for r in rows]
        found = sum(1 for a in accs if a in have)
        pct = found / len(accs) * 100 if accs else 0.0
        print(f'{name:28s} {len(accs):>9,} {found:>6,} {pct:>7.1f}%')
        if accs and pct < 90:
            worst.append((name, found, len(accs)))

    present = [a for a in NEGATIVE_ACCESSIONS if a in have]
    print(f'\nnegative control (NylC nylon hydrolases) present: '
          f'{len(present)} of {len(NEGATIVE_ACCESSIONS)}'
          f'{"  <- should be 0" if present else "  ok"}')
    if worst:
        print('\nACTIONABLE - an external curator calls these amide '
              'synthetases and we do not have them:')
        for name, found, total in worst:
            print(f'  {name}: {found} of {total}')
    else:
        print('\nno named list is under-covered.')
    return worst


def audit_ec(table):
    ec = table.ec.fillna('').astype(str)
    print('\nEC CLASSES THAT DEFINE THE BOND')
    print('the percentage is NOT a quality measure; read the module docstring')
    print(f'{"EC":12s} {"UniProt":>9s} {"ours":>7s} {"recall":>7s}')
    rows = []
    for prefix in EC_PREFIXES:
        for index in range(1, SUB_EC_LIMIT + 1):
            sub = f'{prefix}.{index}'
            external = total_results(f'ec:{sub}')
            if not external:
                continue
            ours = int(ec.str.contains(re.escape(sub) + r'(?!\d)',
                                       regex=True).sum())
            rows.append((sub, external, ours))
    rows.sort(key=lambda row: -row[1])
    for sub, external, ours in rows:
        print(f'{sub:12s} {external:>9,} {ours:>7,} '
              f'{ours / external * 100:>6.1f}%')

    frame = pd.DataFrame(rows, columns=['sub_ec', 'uniprot_total', 'ours'])
    frame['recall_percent'] = (frame.ours / frame.uniprot_total * 100).round(2)
    frame.to_csv(RES / 'recall_by_sub_ec.tsv', sep='\t', index=False)
    print(f'\nwritten: {RES / "recall_by_sub_ec.tsv"}')
    print('A zero here is only a gap if the chemistry is polycondensation-'
          'capable.\nEvery zero measured so far is a terminal amide, a protein'
          ' acceptor or an\nNRPS - all correctly refused. Check the reaction '
          'text before acting on one.')
    return frame


def main(which='all'):
    table, source = load_table()
    print(f'source {source}  candidates {len(table):,}\n')
    if which in ('all', 'names'):
        audit_names(table)
    if which in ('all', 'ec'):
        audit_ec(table)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'all')
