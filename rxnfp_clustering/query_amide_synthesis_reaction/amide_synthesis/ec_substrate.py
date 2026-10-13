#!/usr/bin/env python
"""What is the best nylon-like chemistry anyone in this EC class performs?

WHERE THIS COMES FROM. Screening all of Rhea was tried as a RETRIEVAL route
and failed: 417 amide-forming reactions are unreachable from ModelSEED and
they yield 3 proteins, because almost none has an enzyme annotated to it (see
INFORMATION_SOURCES.md). The conclusion was that the route was dead. That was
the wrong conclusion about the right data - Rhea is poor at naming enzymes and
excellent at describing reactions, so it is useless for retrieval and valuable
for ANNOTATION.

WHAT IT ADDS. A candidate's `substrate_distance` is measured on whichever
ModelSEED reaction happened to match it, and a homologue's is inherited from
its family. Neither says what the protein's own EC CLASS is capable of. Rhea
maps reactions to EC numbers, so scoring every Rhea reaction with this
project's own screen gives, per EC, the best amide chemistry anyone in that
class is curated to do - independent of whether ModelSEED contains the
reaction at all.

Measured 2026-10-13: 26,410 of 38,542 candidates carrying a full EC gain a
substrate-level score they did not have, median 0.557, max 0.869, with 13,063
at 0.6 or better.

HOW TO READ IT, AND HOW NOT TO. `ec_substrate_ceiling` is a CEILING over an EC
class, not this protein's own substrate. Many unrelated proteins share one
value, and a protein is not entitled to its class's best member. So it is
reported, and used only as a tie-break; it does not enter `master_priority`
and it gates nothing. What it is genuinely good for is the opposite direction:
flagging a candidate whose EC class demonstrably does nylon-like chemistry
even though the ModelSEED reaction that retrieved it could not
(`ec_beats_own_reaction`), which is a discovery signal no other column carries.
"""
from __future__ import annotations

import json
import os
import re
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from .http import get
from .paths import CACHE, RES
from .substrate_distance import score_reaction

FTP = 'https://ftp.expasy.org/databases/rhea/tsv/{name}'
SMILES_FILE = CACHE / 'rhea-reaction-smiles.tsv'
EC_FILE = CACHE / 'rhea2ec.tsv'
SCORED = CACHE / 'rhea_substrate_scores.tsv'

FULL_EC = re.compile(r'^\d+\.\d+\.\d+\.\d+$')
# Rhea publishes each reaction as a group of four consecutive ids: the master
# plus its left-to-right, right-to-left and bidirectional forms. rhea2ec keys
# on the master, the SMILES file on every member, so the group is scanned.
GROUP = 4
WORKERS = int(os.environ.get('EC_SUBSTRATE_WORKERS', 12))


def download(name, path):
    """Fetch a Rhea bulk file once, into the cache."""
    if path.exists() and path.stat().st_size > 0:
        return True
    body, _, failed = get(FTP.format(name=name), timeout=180)
    if failed or not body:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return True


def _score(row):
    rhea, smiles = row
    try:
        return rhea, float(score_reaction(smiles)[0])
    except Exception:
        return rhea, 0.0


def score_all(frame, workers=WORKERS):
    """Every Rhea reaction through this project's own screen."""
    pairs = list(zip(frame.rhea, frame.smiles))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        scored = list(pool.map(_score, pairs, chunksize=200))
    return pd.DataFrame(scored, columns=['rhea', 'substrate_distance'])


def ceilings(scores, mapping):
    """EC -> (best substrate distance, the Rhea reaction achieving it).

    Only ECs that reach a scoring reaction appear. An EC whose every reaction
    scores 0.0 is absent rather than present-at-zero, so a caller cannot
    mistake "no amide chemistry in this class" for "not looked up".
    """
    by_id = dict(zip(scores.rhea.astype(int),
                     scores.substrate_distance.astype(float)))
    out = {}
    for row in mapping.itertuples():
        try:
            master, ec = int(row.MASTER_ID), str(row.ID)
        except (TypeError, ValueError):
            continue
        if not FULL_EC.match(ec):
            continue
        best, achieved = 0.0, None
        for offset in range(GROUP):
            value = by_id.get(master + offset, 0.0)
            if value > best:
                best, achieved = value, master + offset
        if best > 0 and best > out.get(ec, (0.0, None))[0]:
            out[ec] = (best, achieved)
    return out


def split_ec(value):
    return [e for e in re.split(r'[;,]\s*', str(value or '')) if FULL_EC.match(e)]


def attach(table, ceiling):
    """Per-candidate EC ceiling, and whether it exceeds its own reaction."""
    best, source, which = [], [], []
    for value in table.get('ec', pd.Series([''] * len(table))).fillna(''):
        found = [(ceiling[e][0], ceiling[e][1], e)
                 for e in split_ec(value) if e in ceiling]
        if found:
            score, rhea, ec = max(found)
            best.append(round(score, 4))
            source.append(ec)
            which.append(rhea)
        else:
            best.append(None)
            source.append(None)
            which.append(None)
    out = pd.DataFrame({
        'uniprot': table.uniprot,
        'ec_substrate_ceiling': best,
        'ec_substrate_source_ec': source,
        'ec_substrate_source_rhea': which,
    })
    own = pd.to_numeric(table.get('substrate_distance_measured', 0),
                        errors='coerce').fillna(0.0)
    inherited = pd.to_numeric(table.get('substrate_distance_inherited', 0),
                              errors='coerce').fillna(0.0)
    ceiling_value = pd.to_numeric(out.ec_substrate_ceiling,
                                  errors='coerce').fillna(0.0)
    out['ec_beats_own_reaction'] = ceiling_value > own.combine(
        inherited, max).values
    return out


def main():
    from .paths import ensure_directories
    ensure_directories()

    if not (download('rhea-reaction-smiles.tsv', SMILES_FILE)
            and download('rhea2ec.tsv', EC_FILE)):
        raise SystemExit('could not fetch the Rhea bulk files; '
                         'nothing written rather than a partial table')

    smiles = pd.read_csv(SMILES_FILE, sep='\t', header=None,
                         names=['rhea', 'smiles'])
    mapping = pd.read_csv(EC_FILE, sep='\t')
    print(f'Rhea: {len(smiles):,} reactions with SMILES, '
          f'{len(mapping):,} EC mappings', flush=True)

    if SCORED.exists():
        scores = pd.read_csv(SCORED, sep='\t')
        print(f'reusing cached scores for {len(scores):,} reactions', flush=True)
    else:
        scores = score_all(smiles)
        scores.to_csv(SCORED, sep='\t', index=False)
    forming = int((scores.substrate_distance > 0).sum())
    print(f'prove amide formation on a free amine: {forming:,}', flush=True)

    ceiling = ceilings(scores, mapping)
    source = RES / 'master_candidates_enriched.tsv'
    if not source.exists():
        source = RES / 'master_candidates.tsv'
    table = pd.read_csv(source, sep='\t', low_memory=False)
    attached = attach(table, ceiling)
    attached.to_csv(RES / 'ec_substrate_ceiling.tsv', sep='\t', index=False)

    scored_rows = attached.ec_substrate_ceiling.notna()
    with_ec = table.ec.fillna('').map(lambda v: bool(split_ec(v)))
    summary = {
        'rhea_reactions_scored': int(len(scores)),
        'rhea_amide_forming': forming,
        'ec_classes_with_amide_chemistry': len(ceiling),
        'candidates_with_a_full_ec': int(with_ec.sum()),
        'candidates_given_a_ceiling': int(scored_rows.sum()),
        'ceiling_at_or_above_0.6': int(
            (attached.ec_substrate_ceiling.fillna(0) >= 0.6).sum()),
        'ec_beats_own_reaction': int(attached.ec_beats_own_reaction.sum()),
        'note': 'A CEILING over an EC class, not the protein\'s own substrate. '
                'Reported and usable as a tie-break; never in master_priority.',
    }
    print(json.dumps(summary, indent=2), flush=True)
    (RES / 'ec_substrate_summary.json').write_text(
        json.dumps(summary, indent=2) + '\n')

    top = attached.dropna(subset=['ec_substrate_ceiling']).nlargest(
        10, 'ec_substrate_ceiling')
    print('\nHighest EC-class substrate ceilings:', flush=True)
    print(top.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
