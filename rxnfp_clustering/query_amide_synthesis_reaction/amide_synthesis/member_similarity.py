#!/usr/bin/env python
"""Tell members of one family apart, instead of ordering them alphabetically.

THE PROBLEM THIS SOLVES, measured. A protein admitted by family membership
inherits its family's substrate distance, so every member of a family scores
the same. The consequence is not cosmetic:

    family_homolog   9,365 candidates,  94 distinct priorities
                     8,593 of them sit in a tie of >= 100 rows
    clade_homolog   36,889 candidates,  49 distinct priorities
                    36,274 of them sit in a tie of >= 100 rows
                    largest single tie: 10,250 rows at one priority

`master_table` sorts by `(master_priority, uniprot)`, so within those ties the
order is the ACCESSION STRING. Which 38 of 9,365 family homologs reach the
bench list has therefore been decided alphabetically. Any member-specific
signal is an improvement on that, and this computes three that need no new
data source:

  domain_jaccard   overlap between the member's Pfam/InterPro signature set
                   and its family's validated seeds. A member carrying the
                   seed's whole architecture is a better bet than one sharing
                   a single domain with it - which is exactly the
                   passenger-domain problem, asked per member rather than per
                   family.
  kmer_cosine      4-mer composition cosine against the seed sequences. An
                   alignment-free identity proxy; 46k x ~600 alignments is not
                   affordable, 4-mer cosine is.
  length_ratio     length agreement with the closest seed. A 200-residue
                   fragment sharing a domain with a 900-residue synthetase is
                   usually a fragment.

WHAT THIS IS NOT. It is NOT validated against activity, and the panel says
plainly that it could not be: identity to the known actives gives AUROC 0.596
with permutation p=0.30, and sequence length alone scores higher (see
tools/validate_against_activity.py). So it does not enter `master_priority`
and it does not gate anything. It is used as a TIE-BREAK inside an evidence
tier, replacing alphabetical order, which is the one place it cannot do harm:
every row it reorders had an identical score by construction.

The three components are reported separately so a reader can tie-break on one
of them instead, or ignore the combination entirely.
"""
from __future__ import annotations

import json
import os
import re

import numpy as np
import pandas as pd

from .paths import RES

SIGNATURE = re.compile(r'(?:PF|IPR)\d+')
KMER = int(os.environ.get('MEMBER_KMER', 4))
FEATURES = int(os.environ.get('MEMBER_HASH_FEATURES', 2 ** 18))
CHUNK = int(os.environ.get('MEMBER_CHUNK', 4000))

# Equal weight: there is no validation set here that could justify tuning
# them, and inventing weights for an unvalidated score is how this project's
# five removed score-gates started. Components are reported separately.
COMPONENTS = ('domain_jaccard', 'kmer_cosine', 'length_ratio')


def signature_set(value):
    return frozenset(SIGNATURE.findall(str(value or '')))


def domain_jaccard(member, seeds):
    """Best Jaccard overlap between one member's signatures and any seed's."""
    if not member or not seeds:
        return 0.0
    best = 0.0
    for seed in seeds:
        if not seed:
            continue
        union = len(member | seed)
        if union:
            best = max(best, len(member & seed) / union)
    return best


def length_ratio(length, seed_lengths):
    """Closest length agreement, as min/max so it is symmetric and <= 1."""
    try:
        length = float(length)
    except (TypeError, ValueError):
        return 0.0
    if not length or length <= 0 or not seed_lengths:
        return 0.0
    best = 0.0
    for other in seed_lengths:
        if other and other > 0:
            best = max(best, min(length, other) / max(length, other))
    return best


def kmer_matrix(sequences):
    """Sparse 4-mer composition vectors, L2-normalised."""
    from sklearn.feature_extraction.text import HashingVectorizer
    vectorizer = HashingVectorizer(
        analyzer='char', ngram_range=(KMER, KMER), n_features=FEATURES,
        norm='l2', alternate_sign=False)
    return vectorizer.transform([str(s or '').upper() for s in sequences])


def best_kmer_cosine(members, seeds, chunk=CHUNK):
    """Max cosine of each member against any seed. Chunked to bound memory."""
    if not len(members) or not len(seeds):
        return np.zeros(len(members))
    seed_matrix = kmer_matrix(seeds)
    out = np.zeros(len(members))
    for start in range(0, len(members), chunk):
        block = kmer_matrix(members[start:start + chunk])
        similarity = (block @ seed_matrix.T).toarray()
        out[start:start + block.shape[0]] = similarity.max(axis=1)
    return out


def score(table, seeds_by_family):
    """Per-member components plus their mean. No I/O."""
    frame = table.copy()
    families = frame.get('seed_pfam')
    if families is None:
        families = pd.Series([''] * len(frame), index=frame.index)

    member_signatures = [signature_set(f'{p};{i}') for p, i in
                         zip(frame.get('pfam', ''), frame.get('interpro', ''))]
    jaccard, ratio = [], []
    for signatures, length, family in zip(member_signatures,
                                          frame.get('length', 0), families):
        seed = seeds_by_family.get(str(family), {})
        jaccard.append(domain_jaccard(signatures, seed.get('signatures', [])))
        ratio.append(length_ratio(length, seed.get('lengths', [])))
    frame['domain_jaccard'] = np.round(jaccard, 4)
    frame['length_ratio'] = np.round(ratio, 4)

    # k-mer cosine, grouped by family so each member is compared only with
    # its OWN family's seeds - comparing against all seeds would reward
    # similarity to an unrelated validated protein.
    cosine = np.zeros(len(frame))
    sequences = frame.get('sequence', pd.Series([''] * len(frame))).tolist()
    for family, index in frame.groupby(families.values).groups.items():
        positions = [frame.index.get_loc(i) for i in index]
        seed = seeds_by_family.get(str(family), {})
        seed_sequences = seed.get('sequences', [])
        if not seed_sequences:
            continue
        cosine[positions] = best_kmer_cosine(
            [sequences[p] for p in positions], seed_sequences)
    frame['kmer_cosine'] = np.round(cosine, 4)

    frame['member_similarity'] = np.round(
        frame[list(COMPONENTS)].mean(axis=1), 4)
    return frame


def seed_profiles(candidates, sequences=None):
    """Signatures, lengths and sequences of each family's validated seeds.

    `sequences` must be supplied: NEITHER candidates_ranked.tsv NOR
    enzymes_uniprot.tsv carries a `sequence` column - the sequences live in
    enzymes.fasta. Reading them off the candidate frame silently gave every
    seed an empty sequence, so `kmer_cosine` returned 0.0 for all 46,254
    homologues while the stage reported success. That is exactly the
    reads-zero-for-everything failure this module's tests name, shipped
    anyway because the tests exercised the function and not the wiring.
    """
    from .expand_families import SIGNATURES, seed_families

    sequences = sequences or {}
    by_accession = candidates.drop_duplicates('uniprot').set_index('uniprot')
    out = {}
    for signature in SIGNATURES:
        for accession, meta in seed_families(candidates,
                                             signature=signature).items():
            slot = out.setdefault(accession, {'signatures': [], 'lengths': [],
                                              'sequences': []})
            for seed in meta['seed_accessions']:
                if seed not in by_accession.index:
                    continue
                row = by_accession.loc[seed]
                slot['signatures'].append(signature_set(
                    f"{row.get('pfam', '')};{row.get('interpro', '')}"))
                slot['lengths'].append(row.get('length'))
                sequence = sequences.get(seed) or row.get('sequence')
                if isinstance(sequence, str) and sequence:
                    slot['sequences'].append(sequence)
    return out


def main():
    from .paths import ensure_directories
    ensure_directories()

    candidates = pd.read_csv(RES / 'candidates_ranked.tsv', sep='\t')
    details = RES / 'enzymes_uniprot.tsv'
    if details.exists():
        full = (pd.read_csv(details, sep='\t', low_memory=False)
                .drop_duplicates('uniprot').set_index('uniprot'))
        for column in ('interpro', 'sequence', 'length'):
            if column in full.columns and column not in candidates.columns:
                candidates[column] = candidates.uniprot.map(full[column])
            elif column in full.columns:
                existing = candidates[column]
                candidates[column] = existing.where(
                    existing.notna(), candidates.uniprot.map(full[column]))

    source = RES / 'master_candidates_enriched.tsv'
    if not source.exists():
        source = RES / 'master_candidates.tsv'
    table = pd.read_csv(source, sep='\t', low_memory=False)
    homologs = table[table.evidence_strength.isin(
        ('family_homolog', 'clade_homolog'))].copy()
    print(f'{source.name}: {len(homologs):,} homologs to discriminate '
          f'({homologs.master_priority.nunique()} distinct priorities)',
          flush=True)

    from .master_table import read_fasta
    sequences = read_fasta(RES / 'enzymes.fasta')
    profiles = seed_profiles(candidates, sequences)
    with_sequences = sum(1 for p in profiles.values() if p['sequences'])
    print(f'seed profiles: {len(profiles)} families, {with_sequences} with '
          f'sequences (from {len(sequences)} in enzymes.fasta)', flush=True)
    if profiles and not with_sequences:
        raise SystemExit(
            'no seed has a sequence, so kmer_cosine would be 0 for every '
            'member. enzymes.fasta missing or empty - run the enzymes stage.')
    scored = score(homologs, profiles)

    # A component with one distinct value across tens of thousands of rows is
    # not a weak signal, it is a broken one, and it must not pass quietly.
    for component in COMPONENTS:
        if len(scored) > 100 and scored[component].nunique() <= 1:
            print(f'  ! WARNING: {component} has a single distinct value '
                  f'({scored[component].iloc[0]}) across {len(scored):,} '
                  'rows - it is contributing nothing', flush=True)
    columns = ['uniprot', 'seed_pfam', *COMPONENTS, 'member_similarity']
    scored[columns].to_csv(RES / 'member_similarity.tsv', sep='\t',
                           index=False)

    summary = {
        'homologs_scored': int(len(scored)),
        'with_a_seed_profile': int((scored.member_similarity > 0).sum()),
        'median': {c: round(float(scored[c].median()), 4)
                   for c in (*COMPONENTS, 'member_similarity')},
        'distinct_values': {
            c: int(scored[c].nunique()) for c in (*COMPONENTS,
                                                  'member_similarity')},
        'note': 'Reported and used only as a tie-break inside an evidence '
                'tier, never in master_priority. Not validated against '
                'activity; see tools/validate_against_activity.py.',
    }
    print(json.dumps(summary, indent=2), flush=True)
    (RES / 'member_similarity_summary.json').write_text(
        json.dumps(summary, indent=2) + '\n')

    print('\nHighest-similarity homologs (would now lead their tie):',
          flush=True)
    print(scored.nlargest(10, 'member_similarity')[
        ['uniprot', 'seed_pfam', *COMPONENTS, 'member_similarity']
    ].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
