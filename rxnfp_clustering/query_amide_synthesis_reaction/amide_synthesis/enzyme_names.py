#!/usr/bin/env python
"""Candidates found by what UniProt CALLS the enzyme, not by what reaction it does.

WHY THIS EXISTS. Every other route starts from a ModelSEED reaction, so a
protein is reachable only if something links it to one: a Rhea alias, an EC
number, a product name, or a validated relative. An external recall test
measured on 2026-10-13 showed what that misses. Of the 26 proteins UniProt
literally names "Amide synthetase", this project had found ZERO - while
correctly excluding all 4 NylC nylon hydrolases used as negative controls.

The cause is not a missing route but a sampling budget. All 26 carry
PF00501 + PF13193, i.e. AMP-binding/ANL - the assayed panel's OWN family, and
one this pipeline does cover. PF00501 has ~450,000 members, far too broad for
the domain route, so it is handled by the clade route at 8 pages per seed. That
samples 767 of 450,000, and the 26 members whose name states the target
function are not among them. This is the same defect as the five score gates
removed earlier: a budget deciding RETRIEVAL, and discarding precisely the
highest-value members.

An enzyme's name is cheap, specific evidence that no reaction pivot can supply,
so it is asked for directly.

PRECISION, AND WHY COUNT IS NOT THE GATE. A sibling route was rejected on
precision: searching the catalytic-activity field for SUBSTRATE names returned
the 100-result page cap on every ubiquitous metabolite, which would have added
~12,000 meaningless associations (see INFORMATION_SOURCES.md). The obvious
defence - admit a query only if it is small - was tested here and is WRONG.
Measured totals:

    amide synthetase               26     specific
    acid--amine ligase            166     specific (YgiC/YjfC, EC 6.3.1.-)
    carboxylate--amine ligase   3,243     specific (EC 6.3.1.-)
    cyanophycin synthetase      3,819     specific, and the closest biological
                                          analogue of nylon there is
    N-acyltransferase          59,383     a SUPERFAMILY label
    glutamine synthetase       64,888     specific, but the WRONG chemistry

Size tracks how widespread an enzyme is, not how precise its name is. So names
are curated on chemistry, each with the reason recorded, and `NAMES` is the
whole of the claim - there is no inference from one name to another. The count
is kept only as a sanity ceiling: a name matching more than SANITY_CEILING
entries is refused loudly rather than ingested quietly, which is the guard that
would catch a superfamily label added here later by mistake.

WHAT A HIT IS WORTH. A name is not a reaction. These rows carry no
`substrate_distance` at all - nothing here knows which reaction the protein
runs, and inventing an inherited value would be fabrication. They rank on the
`enzyme_name_match` weight alone and therefore sit below every
reaction-anchored candidate, which is correct. `master_table` also challenges
them with `annotation_contradicts_chemistry`, as it does every non-reaction
route, so a "reductase" arriving under a qualifying name is still penalised.
"""
from __future__ import annotations

import json
import os

import pandas as pd

from . import uniprot
from .paths import RES

# Protein names denoting formation of an amide bond between a carboxylate and
# an amine - the nylon-6,6 bond. Curated on chemistry; the note is the claim.
NAMES = (
    ('amide synthetase',
     'states the target function; 26 entries, all PF00501+PF13193 ANL'),
    ('amide bond synthetase',
     'the engineered-biocatalysis name for acid + amine coupling'),
    ('amide synthase',
     'the -synthase spelling of the same function'),
    ('amide ligase',
     'the -ligase spelling of the same function'),
    ('acid--amine ligase',
     'EC 6.3.1.- systematic name; YgiC/YjfC ATP-grasp amide synthetases'),
    ('carboxylate--amine ligase',
     'EC 6.3.1.- systematic name for carboxylate + amine -> amide'),
    ('beta-lactam synthetase',
     'intramolecular amide closure by an ANL-family adenylating enzyme'),
)

# Tested and deliberately NOT used. Recorded so they are not retried: the
# measured total is given, but the reason is always chemistry or specificity.
REJECTED = (
    ('N-acyltransferase', 59383,
     'a superfamily label, not a function: spans acetyl-, succinyl-, '
     'myristoyl- and every other acyl donor'),
    ('peptide synthetase', 18124,
     'NRPS modules; the amide is formed on a carrier-bound thioester, not '
     'between two free monomers, and the screen rejects that chemistry'),
    ('aminoacyltransferase', 930,
     'aminoacyl-tRNA and ribosomal machinery; acceptor is not a free amine'),
    ('glutamine synthetase', 64888,
     'forms a TERMINAL amide from ammonia. The substrate screen scores this '
     '0.0 because a terminal amide cannot polycondense. Genuine members are '
     'already present via the EC route where a reaction maps to them'),
    ('cyanophycin synthetase', 3819,
     'on-chemistry and the closest biological nylon analogue, but CphA is '
     'ALREADY captured by activity_ec and family_homolog. Adding ~3,800 '
     'near-identical orthologs would inflate the candidate count without '
     'adding one new family'),
)

# A curated name should never match a superfamily. If one does, the curation is
# wrong and that must be visible, not absorbed.
SANITY_CEILING = int(os.environ.get('ENZYME_NAME_CEILING', 10000))

# Generous: the point of the route is recall, and the largest admitted name has
# ~3,243 members. 40 pages of 100 covers every admitted name completely.
MAX_PAGES = int(os.environ.get('ENZYME_NAME_PAGES', 40))


def query_for(name):
    return f'protein_name:"{name}"'


def fetch_name(name, search=uniprot.search, total=uniprot.total_results):
    """Rows for one curated name, or an empty list if it fails the ceiling.

    Returns (rows, info). `info` always records what happened, including the
    refusals, so the summary cannot silently omit a name.
    """
    query = query_for(name)
    size = total(query)
    if size > SANITY_CEILING:
        return [], {'name': name, 'total': size, 'fetched': 0,
                    'admitted': False,
                    'reason': f'exceeds sanity ceiling {SANITY_CEILING}; '
                              'this name looks like a superfamily label'}
    rows, failed, saturated = search(query, max_pages=MAX_PAGES)
    return rows, {'name': name, 'total': size, 'fetched': len(rows),
                  'admitted': True, 'failed': bool(failed),
                  'saturated': bool(saturated)}


def collect(names=NAMES, search=uniprot.search, total=uniprot.total_results):
    """Every admitted name, de-duplicated, with provenance per protein."""
    from .master_table import RENAME

    frames, report = [], []
    for name, note in names:
        rows, info = fetch_name(name, search=search, total=total)
        info['note'] = note
        report.append(info)
        print(f'  {name:28s} total={info["total"]:>7,} '
              f'fetched={info["fetched"]:>5} '
              f'{"" if info["admitted"] else "REFUSED: " + info["reason"]}',
              flush=True)
        if not rows:
            continue
        frame = pd.DataFrame(rows).rename(columns=RENAME)
        frame['matched_enzyme_name'] = name
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=['uniprot']), report

    table = pd.concat(frames, ignore_index=True)
    # One protein can match two spellings; keep the first and record it.
    table = table.drop_duplicates('uniprot').reset_index(drop=True)
    table['discovery_route'] = 'enzyme_name'
    table['evidence_strength'] = 'enzyme_name_match'
    table['discovery_tier'] = 'protein_name'
    return table, report


def main():
    from .paths import ensure_directories
    ensure_directories()

    print('enzyme-name route: querying UniProt by protein name', flush=True)
    table, report = collect()

    # Reported for context only. This stage does NOT subtract what other routes
    # already found - de-duplication happens once, in master_table.build. The
    # refs stage was order-dependent for exactly this reason.
    known = set()
    for name in ('master_candidates_enriched.tsv', 'master_candidates.tsv'):
        path = RES / name
        if path.exists():
            known = set(pd.read_csv(path, sep='\t', low_memory=False)
                        .uniprot.dropna())
            break
    novel = int((~table.uniprot.isin(known)).sum()) if len(table) else 0

    table.to_csv(RES / 'enzyme_name_candidates.tsv', sep='\t', index=False)
    summary = {
        'names_admitted': sum(1 for r in report if r['admitted']),
        'names_refused': [r['name'] for r in report if not r['admitted']],
        'proteins': int(len(table)),
        'novel_against_existing_master': novel,
        'per_name': report,
        'rejected_names': [{'name': n, 'total': t, 'reason': why}
                           for n, t, why in REJECTED],
    }
    print(json.dumps(summary, indent=2), flush=True)
    (RES / 'enzyme_name_summary.json').write_text(
        json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
