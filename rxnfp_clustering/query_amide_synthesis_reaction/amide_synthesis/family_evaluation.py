#!/usr/bin/env python
"""Evaluate each validated FAMILY as a unit: is its chemistry nylon-relevant?

WHY A FAMILY-LEVEL VIEW. The pipeline evaluates proteins, and a protein
admitted by family membership inherits its family's score. So the real
decision - the one that determines tens of thousands of candidates - is made
once per family, and until now it was made implicitly by an enrichment number
with no statement of the chemistry behind it.

Two wrong answers were tried first and are recorded here so neither is
repeated:

  Enumerate every member. Built and turned off (ENUMERATE_DOMAIN). The 61
  validated family groups hold 257,297 members, and because every member
  INHERITS its family's substrate distance, those rows would carry about 61
  distinct scores between them. It is a membership list, not an evaluation,
  and it buries the 816 reaction-anchored candidates - the only ones whose
  chemistry was measured - under three hundred times their number of
  homologues. This is a TARGETED search; completeness of a homology list is a
  different goal and trades against it.

  Trust the enrichment number alone. Enrichment says a family is unusually
  full of validated seeds. It does not say what bond those seeds make, and it
  demonstrably admits the wrong chemistry: PF06276 is shared by the NIS
  synthetases and by FhuF, a ferric siderophore REDUCTASE, and the InterPro
  axis re-admitted CHAP, a peptidoglycan hydrolase.

So each family is described by the MEASURED chemistry of its own seeds,
recomputed from their reaction SMILES with the same screen the rest of the
pipeline uses. The columns that decide nylon relevance are the ones the target
bond needs:

    donor_acids        adipate is a DIACID; a monoacid donor cannot extend a
                       chain at both ends
    acceptor_amines    HMDA is a DIAMINE; a monoamine acceptor terminates
    aa_bb_pair         a diacid AND a diamine - the nylon-6,6 arrangement
    self_condensing    one bifunctional monomer, the nylon-6 arrangement
    retains_acid/amine whether the PRODUCT can react again, which is what
                       separates polycondensation from a terminal cap
    donor_heavy_atoms  scaffold size against adipate's 10; a pterin-bearing
                       donor is a poor starting point however good the bond

The verdict column is a transparent consequence of those, never a model: a
reader can disagree with one line and recompute.

Run as a pipeline stage (`run_all.py --only famval`), or directly:
    python -m amide_synthesis.family_evaluation          # table + summary
    python -m amide_synthesis.family_evaluation --csv    # file only

The verdict is consumed by `shortlist`, which excludes members of a family
judged `reject` and records the reason. It runs AFTER `enrich` because the
member-contradiction check needs the annotated candidate table.
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

import pandas as pd

from .acylation import acylation_events
from .expand_families import SIGNATURES, seed_families
from .http import get_text
from .paths import CACHE, RES
from .substrate_distance import describe_event, score_event

# Adipate: 10 heavy atoms. A donor far larger than this carries a scaffold the
# pocket must also accommodate, which is what demoted FPGS despite its bond
# chemistry scoring 0.883.
ADIPATE_HEAVY_ATOMS = 10
SCAFFOLD_TOLERANCE = 3.0

# The signature's OWN name, resolved from InterPro. The first version of this
# tool inherited the label from a seed protein's `family` column, which is
# whichever domain happened to be listed first on that protein - so IPR016185
# was printed as "CHAP domain" when it is an acyl-CoA N-acyltransferase fold,
# and a family table whose labels are wrong is worse than no table.
# Two signatures are the same family when their validated-seed sets agree
# this well. See the union-find below for why this is Jaccard and not
# containment.
JACCARD_MERGE = float(os.environ.get('FAMILY_JACCARD_MERGE', 0.7))

NAME_CACHE = CACHE / 'signature_names.json'
API = 'https://www.ebi.ac.uk/interpro/api/entry/{db}/{accession}/'


def signature_names(accessions):
    """Accession -> its own curated name, cached on disk."""
    import json
    import time
    known = (json.loads(NAME_CACHE.read_text())
             if NAME_CACHE.exists() else {})
    changed = False
    for accession in dict.fromkeys(accessions):
        if not accession or accession in known:
            continue
        database = 'interpro' if accession.startswith('IPR') else 'pfam'
        text, _, failed = get_text(
            API.format(db=database, accession=accession), timeout=30)
        if failed:
            continue        # never cache a transient outage as a real answer
        try:
            known[accession] = json.loads(text)['metadata']['name']['name']
            changed = True
        except Exception:
            pass
        time.sleep(0.15)
    if changed:
        NAME_CACHE.parent.mkdir(parents=True, exist_ok=True)
        NAME_CACHE.write_text(json.dumps(known, indent=2))
    return known


def best_event(rxn_smiles):
    """The highest-scoring acylation in a reaction, with its features."""
    try:
        events = acylation_events(str(rxn_smiles))
    except Exception:
        return None
    best, features = -1.0, None
    for event in events:
        described = describe_event(event)
        score = score_event(described)
        if score > best:
            best, features = score, described
    return features if features else None


def family_chemistry(reactions):
    """Aggregate the measured chemistry of one family's seed reactions."""
    out = {
        'donor_acids': 0, 'acceptor_amines': 0, 'donor_heavy_atoms': None,
        'aa_bb_pair': False, 'self_condensing': False,
        'retains_acid': False, 'retains_amine': False,
        'best_substrate_distance': 0.0, 'example_reaction': '',
    }
    for name, smiles, distance in reactions:
        features = best_event(smiles)
        if not features:
            continue
        if float(distance or 0) >= out['best_substrate_distance']:
            out['best_substrate_distance'] = float(distance or 0)
            out['example_reaction'] = name
            out['donor_heavy_atoms'] = features.get('donor_parent_heavy_atoms')
        out['donor_acids'] = max(out['donor_acids'],
                                 int(features.get('acyl_n_acids') or 0))
        out['acceptor_amines'] = max(
            out['acceptor_amines'],
            int(features.get('acceptor_free_amines') or 0))
        out['aa_bb_pair'] |= bool(features.get('aa_bb_pair'))
        out['self_condensing'] |= bool(features.get('self_condensation'))
        out['retains_acid'] |= bool(features.get('product_retains_acid'))
        out['retains_amine'] |= bool(features.get('product_retains_amine'))
    return out


# A family whose own MEMBERS are overwhelmingly annotated as the wrong
# activity is not a lead however good its seeds look. Gamma-glutamyl-
# transpeptidase is the case that forced this: its seeds score 0.64, and
# 1,550 of its 1,594 members carry an annotation contradicting amide
# formation, because it is a hydrolase family that happens to run a
# transpeptidation side reaction. Judging it on seed chemistry alone called it
# a lead.
CONTRADICTED_MAJORITY = float(os.environ.get('FAMILY_CONTRA_MAX', 0.5))


def verdict(row):
    """Why this family is or is not a nylon-6,6 lead. Transparent, not fitted."""
    if row['best_substrate_distance'] <= 0.0:
        return 'reject', 'no scoring acylation on a free amine'
    if not (row['retains_acid'] and row['retains_amine']):
        return 'reject', 'product cannot react again (terminal amide)'
    members = int(row.get('members_in_table') or 0)
    contradicted = int(row.get('members_contradicted') or 0)
    if members and contradicted / members > CONTRADICTED_MAJORITY:
        return 'reject', (f'{contradicted} of {members} members are annotated '
                          'as a non-amide-forming activity')
    if row['aa_bb_pair']:
        return 'lead', 'diacid donor + diamine acceptor, the nylon-6,6 pairing'
    if row['self_condensing']:
        return 'lead', 'self-condensing bifunctional monomer (nylon-6 style)'
    bifunctional = row['donor_acids'] >= 2 or row['acceptor_amines'] >= 2
    heavy = row['donor_heavy_atoms']
    bulky = (heavy is not None
             and heavy > ADIPATE_HEAVY_ATOMS * SCAFFOLD_TOLERANCE)
    if bifunctional and not bulky:
        return 'lead', ('bifunctional on one side '
                        f"(donor acids {row['donor_acids']}, "
                        f"acceptor amines {row['acceptor_amines']})")
    if bifunctional and bulky:
        return 'watch', (f'bifunctional but the donor scaffold is {heavy} heavy '
                         f'atoms against adipate\'s {ADIPATE_HEAVY_ATOMS}')
    return 'watch', 'amide-forming but monofunctional on both sides'


def main(as_csv=False):
    from .paths import ensure_directories
    ensure_directories()
    candidates = pd.read_csv(RES / 'candidates_ranked.tsv', sep='\t')
    details = RES / 'enzymes_uniprot.tsv'
    if details.exists() and 'interpro' not in candidates.columns:
        full = (pd.read_csv(details, sep='\t', low_memory=False)
                .drop_duplicates('uniprot').set_index('uniprot'))
        if 'interpro' in full.columns:
            candidates['interpro'] = candidates.uniprot.map(full.interpro)

    screen = (pd.read_csv(RES / 'amide_forming_neighbours.tsv', sep='\t',
                          low_memory=False)
              .drop_duplicates('rxn_id').set_index('rxn_id'))
    master = RES / 'master_candidates_enriched.tsv'
    table = (pd.read_csv(master, sep='\t', low_memory=False)
             if master.exists() else pd.DataFrame())

    # Reactions per seed protein, so a family is described by its own seeds.
    by_protein = defaultdict(list)
    for row in candidates.itertuples():
        rxn = getattr(row, 'rxn_id', None)
        if rxn in screen.index:
            record = screen.loc[rxn]
            by_protein[row.uniprot].append((
                str(record.get('reaction_name', '') or rxn),
                record.get('rxn_smiles', ''),
                record.get('substrate_distance', 0.0)))

    rows = []
    for signature in SIGNATURES:
        for accession, meta in seed_families(candidates,
                                             signature=signature).items():
            reactions = []
            for seed in meta['seed_accessions']:
                reactions.extend(by_protein.get(seed, []))
            if not reactions:
                continue
            chemistry = family_chemistry(reactions)
            call, reason = verdict(chemistry)
            members = pd.DataFrame()
            if len(table):
                column = SIGNATURES[signature]['column']
                if column in table.columns:
                    members = table[table[column].fillna('').astype(str)
                                    .str.contains(accession, regex=False)]
            contradicted = 0
            if len(members) and 'annotation_contradicts_chemistry' in members:
                contradicted = int(members.annotation_contradicts_chemistry
                                   .fillna(False).astype(bool).sum())
            rows.append({
                'signature': accession, 'db': signature,
                'seed_set': frozenset(meta['seed_accessions']),
                'seed_label': str(meta.get('family', ''))[:44],
                'n_seeds': meta['n_seeds'],
                'verdict': call, 'reason': reason,
                **chemistry,
                'members_in_table': int(len(members)),
                'members_contradicted': contradicted,
                'members_with_pdb': int(members.pdb.notna().sum())
                if len(members) and 'pdb' in members else 0,
            })

    frame = pd.DataFrame(rows)
    if frame.empty:
        print('no families with seed reactions found')
        return

    # Each signature's own curated name, then ONE ROW PER FAMILY.
    #
    # Pfam and InterPro describe the same families from different angles, so
    # the per-signature table listed D-ala--D-ala ligase seven times and
    # glutamine synthetase ten: 298 rows for a few dozen real families, which
    # is not something a human can review. Collapsing on the resolved name is
    # the point of the tool - the signatures that voted are kept in a column.
    names = signature_names(frame.signature.tolist())
    frame['family'] = [names.get(s, l) for s, l
                       in zip(frame.signature, frame.seed_label)]
    frame.drop(columns=['seed_set']).to_csv(
        RES / 'family_evaluation_by_signature.tsv', sep='\t', index=False)

    # ONE ROW PER ENZYME FAMILY, grouped by SEED OVERLAP rather than by name.
    #
    # Resolving each signature's own name fixed the wrong labels but not the
    # redundancy: InterPro names every subdomain, so "D-ala D-ala ligase
    # C-terminus", "... N-terminus" and "D-alanine--D-alanine ligase" are
    # three rows for one enzyme. Signatures carried by the SAME seed proteins
    # are describing the same enzymes, so overlap is the honest grouping key
    # and it needs no name heuristics.
    parent = {s: s for s in frame.signature}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    seeds = dict(zip(frame.signature, frame.seed_set))
    signatures = list(frame.signature)
    for i, a in enumerate(signatures):
        for b in signatures[i + 1:]:
            sa, sb = seeds[a], seeds[b]
            if not sa or not sb:
                continue
            shared = len(sa & sb)
            if not shared:
                continue
            # MUTUAL similarity, not containment. Containment was tried first
            # and over-merged badly: a broad fold's seed set CONTAINS every
            # narrow family inside it, so union-find chained ATP-grasp,
            # D-ala--D-ala ligase, Mur ligase and the rest into one 137-
            # signature blob. InterPro encodes that superfamily -> family ->
            # domain hierarchy on purpose, and flattening it is as wrong as
            # the redundancy it was meant to fix. Jaccard merges only
            # signatures describing the SAME enzymes and leaves a superfamily
            # as its own, legitimately broader, row.
            if shared / len(sa | sb) >= JACCARD_MERGE:
                union(a, b)
    frame['family_group'] = [find(s) for s in frame.signature]

    def label(part):
        """The cluster's enzyme name: the least subdomain-ish, most-seeded."""
        noise = ('domain', 'terminal', 'terminus', 'superfamily', 'subdomain',
                 '-like', 'fold', 'conserved', 'site')
        ranked = sorted(
            part.itertuples(),
            key=lambda r: (sum(w in str(r.family).lower() for w in noise),
                           -int(r.n_seeds), len(str(r.family))))
        return str(ranked[0].family)

    names_by_group = {g: label(part)
                      for g, part in frame.groupby('family_group')}
    frame['family'] = frame.family_group.map(names_by_group)

    grouped = frame.groupby('family', as_index=False).agg(
        signatures=('signature', lambda s: ';'.join(sorted(s))),
        all_signature_names=('family', 'size'),
        n_signatures=('signature', 'size'),
        n_seeds=('n_seeds', 'max'),
        donor_acids=('donor_acids', 'max'),
        acceptor_amines=('acceptor_amines', 'max'),
        aa_bb_pair=('aa_bb_pair', 'any'),
        self_condensing=('self_condensing', 'any'),
        retains_acid=('retains_acid', 'any'),
        retains_amine=('retains_amine', 'any'),
        donor_heavy_atoms=('donor_heavy_atoms', 'min'),
        best_substrate_distance=('best_substrate_distance', 'max'),
        example_reaction=('example_reaction', 'first'),
        members_in_table=('members_in_table', 'max'),
        members_contradicted=('members_contradicted', 'max'),
        members_with_pdb=('members_with_pdb', 'max'),
    )
    # Re-derive the verdict from the COLLAPSED chemistry, so a family is judged
    # once on everything its signatures saw rather than per signature.
    calls = grouped.apply(lambda r: verdict(r), axis=1)
    grouped['verdict'] = [c for c, _ in calls]
    grouped['reason'] = [r for _, r in calls]

    order = {'lead': 0, 'watch': 1, 'reject': 2}
    frame = grouped.sort_values(
        ['verdict', 'best_substrate_distance'],
        key=lambda s: s.map(order) if s.name == 'verdict' else -s,
    ).reset_index(drop=True)
    frame.to_csv(RES / 'family_evaluation.tsv', sep='\t', index=False)

    if as_csv:
        print(f'written: {RES / "family_evaluation.tsv"}')
        return

    print(f'{len(frame)} distinct families (from '
          f'{int(frame.n_signatures.sum())} Pfam/InterPro signatures), '
          f'evaluated on their OWN seeds\' chemistry\n')
    print(frame.verdict.value_counts().to_string())
    show = ['family', 'n_sig', 'n_seeds', 'acids', 'amines', 'AA/BB', 'self',
            'donor_HA', 'sd', 'members', 'contra']
    for call in ('lead', 'watch', 'reject'):
        part = frame[frame.verdict == call]
        if part.empty:
            continue
        print(f'\n=== {call.upper()} ({len(part)}) ===')
        display = part.rename(columns={
            'n_signatures': 'n_sig', 'donor_acids': 'acids',
            'acceptor_amines': 'amines', 'aa_bb_pair': 'AA/BB',
            'self_condensing': 'self', 'donor_heavy_atoms': 'donor_HA',
            'best_substrate_distance': 'sd', 'members_in_table': 'members',
            'members_contradicted': 'contra'})
        display['family'] = display.family.astype(str).str.slice(0, 46)
        print(display[show].to_string(index=False))
        print('\nreasons:')
        for reason, n in part.reason.value_counts().items():
            print(f'  {n:>3}  {reason}')
    print(f'\nwritten: {RES / "family_evaluation.tsv"}')


if __name__ == '__main__':
    main(as_csv='--csv' in sys.argv)
