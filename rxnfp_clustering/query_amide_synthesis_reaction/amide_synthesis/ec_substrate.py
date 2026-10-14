#!/usr/bin/env python
"""Substrate distance from each protein's OWN curated reaction, and its EC's ceiling.

WHERE THIS COMES FROM. Screening all of Rhea was tried as a RETRIEVAL route
and failed: 417 amide-forming reactions are unreachable from ModelSEED and
they yield 3 proteins, because almost none has an enzyme annotated to it (see
INFORMATION_SOURCES.md). The conclusion was that the route was dead. That was
the wrong conclusion about the right data - Rhea is poor at naming enzymes and
excellent at describing reactions, so it is useless for retrieval and valuable
for ANNOTATION.

WHAT IT ADDS, in order of strength. A candidate's `substrate_distance` is
measured on whichever ModelSEED reaction happened to match it, and a
homologue's is inherited from its family - a value describing a RELATIVE.
Scoring the whole Rhea corpus with this project's own screen supplies two
better things, measured 2026-10-13:

1. `substrate_distance_own_rhea` - THE PRIMARY OUTPUT. UniProt writes the Rhea
   id of the reaction it attributes to each protein into that protein's
   catalytic-activity line, and 38,127 of 54,199 candidates have one. 27,603
   of them score above zero, 12,736 at 0.6 or better, and 17,185 improve on
   the value they previously carried. This is a distance measured on the
   reaction the protein is CURATED TO CATALYSE - the same kind of evidence as
   `substrate_distance_measured`, differing only in source database, so
   `master_table` lets it supersede an inherited value.

2. `ec_substrate_ceiling` - the best amide chemistry anyone in the protein's
   EC class performs. 26,410 candidates, median 0.557, max 0.869. A ceiling
   over a class, so it is reported only; see below.

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
# UniProt writes the Rhea id of the reaction it attributes to THIS protein into
# the catalytic-activity line: "Reaction=...; Xref=Rhea:RHEA:11224, ...".
RHEA_XREF = re.compile(r'RHEA:(\d+)')
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


def own_reaction_distance(statements, scores):
    """Substrate distance on the reaction UniProt attributes to each protein.

    This is the strongest substrate evidence available to a protein that no
    ModelSEED reaction matched. 38,127 of 54,199 candidates cite a Rhea id in
    their OWN catalytic-activity statement, and the full Rhea corpus is
    already scored here - so instead of an inherited family value or a
    class-wide ceiling, the protein gets a distance measured on the reaction
    it is curated to catalyse.

    Same kind of evidence as `substrate_distance_measured`, which is scored on
    a ModelSEED reaction and already carries 3x weight in the ranking; only
    the source database differs.
    """
    by_id = dict(zip(scores.rhea.astype(int),
                     scores.substrate_distance.astype(float)))
    best, cited = [], []
    for statement in statements:
        ids = [int(x) for x in RHEA_XREF.findall(str(statement or ''))]
        top, source = 0.0, None
        for rhea in ids:
            # Scan the direction group in both directions: the statement may
            # cite any member and the score may sit on another.
            for offset in range(GROUP):
                for candidate in (rhea + offset, rhea - offset):
                    value = by_id.get(candidate, 0.0)
                    if value > top:
                        top, source = value, candidate
        best.append(round(top, 4) if top else None)
        cited.append(source)
    return best, cited


SOURCES = (
    ('enzymes_uniprot.tsv', 'uniprot'),
    ('family_expansion.tsv', 'Entry'),
    ('precomputed_link_candidates.tsv', 'uniprot'),
    ('enzyme_name_candidates.tsv', 'uniprot'),
    ('reference_actives.tsv', 'uniprot'),
    ('chemistry_neighbour_candidates.tsv', 'uniprot'),
    ('candidates_ranked.tsv', 'uniprot'),
    ('active_homolog_candidates.tsv', 'uniprot'),
)


def candidate_annotations():
    """(uniprot, ec, catalytic_activity) for every candidate, from the ROUTES.

    Deliberately NOT read from master_candidates.tsv. `master_table` consumes
    this stage's `substrate_distance_own_rhea` to rank, so reading master here
    would make the two mutually dependent - and a two-pass "master, then this,
    then master again" is precisely the order-dependent arrangement that has
    already silently discarded results twice in this project (`enrich` after
    `master`, `refs` reading `master`).

    Everything needed is a protein accession and its UniProt annotation, which
    every route file carries, so this runs BEFORE master and master simply
    joins the column.
    """
    frames = []
    # EVERY route file must be listed. The list is explicit rather than a
    # glob because the argument above is about a precise dependency, but an
    # explicit list has one failure mode: add a route, forget this. That
    # happened - `active_homolog_candidates.tsv` was missing and its 822
    # proteins silently had no EC ceiling and no own-reaction distance, while
    # the stage reported a healthy 54,199. `tests/test_ec_substrate.py` pins
    # the list against the routes the pipeline declares, and
    # `tools/audit_coverage.py` catches it from the other end.
    sources = SOURCES
    for name, key in sources:
        path = RES / name
        if not path.exists():
            continue
        frame = pd.read_csv(path, sep='\t', low_memory=False)
        if key not in frame.columns:
            continue
        columns = {key: 'uniprot'}
        for raw, ours in (('EC number', 'ec'), ('ec', 'ec'),
                          ('Catalytic activity', 'catalytic_activity'),
                          ('catalytic_activity', 'catalytic_activity')):
            if raw in frame.columns:
                columns[raw] = ours
        part = frame[list(columns)].rename(columns=columns)
        frames.append(part)
    if not frames:
        raise SystemExit('no route outputs found; run the retrieval stages')
    table = pd.concat(frames, ignore_index=True)
    for column in ('ec', 'catalytic_activity'):
        if column not in table.columns:
            table[column] = ''
    # Keep the richest record per accession: a route that fetched the full
    # UniProt entry beats one that carried only an accession.
    table['_filled'] = (table.catalytic_activity.fillna('').astype(str)
                        .str.len() + table.ec.fillna('').astype(str).str.len())
    table = (table.sort_values('_filled', ascending=False)
             .drop_duplicates('uniprot').drop(columns='_filled')
             .reset_index(drop=True))
    print(f'candidate annotations assembled from the route files: '
          f'{len(table):,} accessions', flush=True)
    return table


def split_ec(value):
    return [e for e in re.split(r'[;,]\s*', str(value or '')) if FULL_EC.match(e)]


def _column(frame, name, default=''):
    """Always a Series aligned to `frame`.

    `DataFrame.get(name, default)` returns the SCALAR default when the column
    is absent. This stage reads the route files, which legitimately do not
    carry the distance columns, and the scalar made `.fillna` raise. The same
    trap has now been fixed four times in this project - master_table, enrich,
    shortlist and here - so it is a helper rather than a habit.
    """
    if name in frame.columns:
        return frame[name]
    return pd.Series([default] * len(frame), index=frame.index)


def attach(table, ceiling):
    """Per-candidate EC ceiling.

    Whether the ceiling beats the protein's own distance is NOT decided here:
    this stage runs before `master`, so the measured and inherited distances
    do not exist yet. `master_table` computes that flag where all three values
    are in one place.
    """
    best, source, which = [], [], []
    for value in _column(table, 'ec').fillna(''):
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
    return pd.DataFrame({
        'uniprot': table.uniprot,
        'ec_substrate_ceiling': best,
        'ec_substrate_source_ec': source,
        'ec_substrate_source_rhea': which,
    })


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
    table = candidate_annotations()
    attached = attach(table, ceiling)

    # The protein's OWN curated reaction, which outranks a class ceiling.
    own, cited = own_reaction_distance(
        _column(table, 'catalytic_activity'), scores)
    attached['substrate_distance_own_rhea'] = own
    attached['own_rhea_reaction'] = cited
    attached.to_csv(RES / 'ec_substrate_ceiling.tsv', sep='\t', index=False)

    scored_rows = attached.ec_substrate_ceiling.notna()
    with_ec = _column(table, 'ec').fillna('').map(lambda v: bool(split_ec(v)))
    summary = {
        'rhea_reactions_scored': int(len(scores)),
        'rhea_amide_forming': forming,
        'ec_classes_with_amide_chemistry': len(ceiling),
        'candidates_with_a_full_ec': int(with_ec.sum()),
        'candidates_given_a_ceiling': int(scored_rows.sum()),
        'ceiling_at_or_above_0.6': int(
            (attached.ec_substrate_ceiling.fillna(0) >= 0.6).sum()),
        'candidates_citing_a_rhea_reaction': int(
            _column(table, 'catalytic_activity')
            .fillna('').astype(str).str.contains('RHEA:').sum()),
        'with_own_reaction_distance': int(
            attached.substrate_distance_own_rhea.notna().sum()),
        'own_reaction_distance_at_or_above_0.6': int(
            (attached.substrate_distance_own_rhea.fillna(0) >= 0.6).sum()),
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
