#!/usr/bin/env python
"""One candidate table over every discovery route, with provenance kept intact.

Candidates now arrive by four routes that differ in how much they claim:

  reaction  a protein annotated to a ModelSEED-mappable reaction that the
            screen PROVED forms an amide on a free amine. `substrate_distance`
            is MEASURED on that reaction's own substrates.
  domain    a member of a Pfam family whose chemistry is characteristic
            (high enrichment of validated seeds). Distance is INHERITED.
  clade     a member of a validated protein's own UniRef50 cluster, used where
            the Pfam is too broad to mean anything (Mur ligase, AMP-binding).
            Distance is INHERITED.
  structural  not a discovery route: an independently computed geometric score
            joined in as corroboration where it exists.

MEASURED AND INHERITED ARE NOT INTERCHANGEABLE, and the ranking says so. A
reaction-anchored protein's distance describes a reaction it is annotated to
catalyse. A homolog's distance describes a reaction some RELATIVE catalyses, so
it is discounted - see INHERITED_DISCOUNT. Both are carried in separate columns
so the discount can be changed, or dropped, without recomputing anything.

WHAT THIS RANKING IS NOT. It has never been fitted to, or validated against,
measured activity. The one independent check available - a geometric score
computed elsewhere on 17 overlapping proteins - showed NO correlation with
substrate distance (Spearman +0.14, p=0.60). That is expected rather than
alarming, because the two ask different questions: substrate distance asks
whether the chemistry is nylon-like, the geometric score asks whether a
particular pocket can hold adipate and HMDA. Neither validates the other, and
a candidate supported by both is on firmer ground than one supported by either.
`struct_S` is therefore reported alongside, never folded into the priority.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from .paths import RES

# A relative's chemistry is weaker evidence than the protein's own annotation.
INHERITED_DISCOUNT = float(os.environ.get('INHERITED_DISCOUNT', 0.7))

# A homolog is admitted by DOMAIN membership, and a domain can be real without
# being exclusive to the chemistry. PF06276 is the clearest case: it is the
# C-terminal domain of the NIS synthetases AND of FhuF itself, a ferric
# siderophore REDUCTASE, which therefore arrives with the family's 0.83
# distance attached. Enrichment cannot catch this - 34 genuine NIS seeds carry
# the domain - so the protein's own annotation is checked instead.
#
# Nothing is deleted: the contradiction is recorded in
# `annotation_contradicts_chemistry` and subtracted from the priority, so a
# reader can restore these rows by ignoring one labelled column.
DISQUALIFYING = (
    'reductase', 'dehydrogenase', 'oxidase', 'transporter', 'permease',
    'hydrolase', 'amidase', 'esterase', 'peptidase', 'protease', 'nuclease',
    'phosphatase', 'isomerase', 'channel', 'receptor', 'binding protein')
# A bifunctional enzyme naming both is not a contradiction.
QUALIFYING = ('synthetase', 'synthase', 'ligase', 'transferase')
CONTRADICTION_PENALTY = 3.0

EVIDENCE_WEIGHT = {
    'measured_activity': 10.0,  # assayed on the nylon-6,6 diad itself
    'exact_reaction': 3.0,      # UniProt annotated to that Rhea reaction
    'precomputed_rhea_or_go': 2.5,  # colleague's global link table
    'activity_ec': 2.0,         # matching EC, broader
    'literature_curated': 2.0,  # carries an independent structural score
    'name_match': 1.0,          # distinctive product name
    'reaction_name_match': 0.5,  # ModelSEED's reaction label; weakest route
    'family_homolog': 1.0,      # enriched Pfam family
    'clade_homolog': 1.0,       # UniRef50 neighbourhood of a validated protein
}

# Measurement dominates prediction. A protein with an assayed titre on the
# target chemistry is not a hypothesis, and must not be displaced by one.
MEASURED_ACTIVITY_BONUS = 5.0

# An independently computed geometric proxy, read-only, from a separate project.
STRUCTURAL_SCORES = Path(
    '/homes/rzhu/Enzyme_agent/exp_data/amide_synthetase/structural_proxies/'
    'tables/selected_score_discovery.csv')

RENAME = {
    'Entry': 'uniprot', 'Entry Name': 'entry_name', 'Protein names': 'protein_name',
    'Gene Names': 'gene', 'Organism': 'organism', 'Taxonomic lineage': 'lineage',
    'EC number': 'ec', 'Length': 'length', 'Sequence': 'sequence',
    'Catalytic activity': 'catalytic_activity', 'PDB': 'pdb',
    'Protein existence': 'protein_existence', 'Reviewed': 'reviewed',
    'Cofactor': 'cofactor', 'Pathway': 'pathway', 'Subunit structure': 'subunit',
    'Function [CC]': 'function', 'Pfam': 'pfam', 'InterPro': 'interpro',
    'AlphaFoldDB': 'alphafold', 'Binding site': 'binding_site',
    'Active site': 'active_site', 'Keywords': 'keywords',
    'Gene Ontology (molecular function)': 'go_function',
    'Gene Ontology (biological process)': 'go_process',
}

COLUMNS = [
    'master_priority', 'uniprot', 'gene', 'protein_name', 'organism',
    'discovery_route', 'evidence_strength', 'discovery_tier',
    'substrate_distance_measured', 'substrate_distance_inherited',
    'self_condensation', 'struct_S', 'annotation_contradicts_chemistry',
    'activity_ugml', 'activity_std', 'is_strong_producer',
    'n_proteins_on_reaction', 'over_annotated_reaction',
    'family', 'pfam', 'interpro', 'seed_pfam', 'clade_seed',
    'rxn_id', 'reaction_name', 'query', 'cosine_similarity',
    'ec', 'catalytic_activity', 'cofactor', 'pathway', 'function',
    'reviewed', 'protein_existence', 'length', 'pdb', 'alphafold',
    'binding_site', 'active_site', 'keywords', 'go_function', 'go_process',
    'subunit', 'lineage', 'sequence',
]


# Stage 04 writes the full UniProt record under its own column names, and the
# sequences to FASTA. The candidate table keeps only a display subset, which
# left the BEST candidates - the reaction-anchored ones - with no sequence,
# InterPro, keywords or lineage at all, and therefore unusable for the
# structural stage. They are re-joined here rather than re-fetched.
REACTION_FIELD_ALIASES = {
    'binding_sites': 'binding_site',
    'active_sites': 'active_site',
    'go_molecular_function': 'go_function',
    'go_biological_process': 'go_process',
    'evidence': 'protein_existence',
}


def read_fasta(path):
    """{accession: sequence} from the stage-04 export."""
    sequences, accession, chunks = {}, None, []
    path = Path(path)
    if not path.exists():
        return sequences
    for line in path.read_text().splitlines():
        if line.startswith('>'):
            if accession:
                sequences[accession] = ''.join(chunks)
            accession, chunks = line[1:].split('|')[0], []
        elif accession:
            chunks.append(line.strip())
    if accession:
        sequences[accession] = ''.join(chunks)
    return sequences


def enrich_reaction_rows(reaction, details, sequences):
    """Fill the reaction-anchored rows from the full stage-04 record."""
    if details is not None and len(details):
        details = details.rename(columns=REACTION_FIELD_ALIASES)
        details = details.drop_duplicates('uniprot').set_index('uniprot')
        for column in details.columns:
            if column in ('rxn_id', 'reaction_name', 'cosine_similarity'):
                continue          # the candidate table's own values are current
            incoming = reaction.uniprot.map(details[column])
            if column in reaction.columns:
                existing = reaction[column]
                blank = existing.isna() | existing.astype(str).str.strip().eq('')
                reaction[column] = existing.where(~blank, incoming)
            else:
                reaction[column] = incoming
    if sequences:
        existing = reaction.get('sequence')
        incoming = reaction.uniprot.map(sequences)
        reaction['sequence'] = (incoming if existing is None else
                                existing.where(existing.notna()
                                               & existing.astype(str).str.strip().ne(''),
                                               incoming))
    return reaction


def _column(frame, name, default=''):
    """Always a Series aligned to `frame`.

    `DataFrame.get(name, default)` returns the SCALAR default when the column
    is absent, so every downstream `.fillna`/`.str` call raises AttributeError.
    Routes contribute different column sets by design, so absent columns are
    normal and must not be fatal.
    """
    if name in frame.columns:
        return frame[name]
    return pd.Series([default] * len(frame), index=frame.index)


def _truthy(series):
    return series.fillna('').astype(str).str.strip().ne('')


def load_structural(path=STRUCTURAL_SCORES):
    """Independently computed geometric scores, if the file is present."""
    if not Path(path).exists():
        return pd.DataFrame(columns=['uniprot', 'struct_S'])
    frame = pd.read_csv(path)
    return frame.rename(columns={'id': 'uniprot', 'score': 'struct_S'})[
        ['uniprot', 'struct_S']]


def build(reaction_candidates, expansion, structural=None,
          reaction_details=None, sequences=None, extra_sources=()):
    """Merge every route into one ranked table. No I/O.

    `extra_sources` are frames that already carry `uniprot`,
    `discovery_route` and `evidence_strength` - the precomputed link table and
    the reference/assayed proteins. They are appended last and de-duplicated
    against the stronger routes.
    """
    reaction = enrich_reaction_rows(reaction_candidates.copy(),
                                    reaction_details, sequences or {})
    reaction['discovery_route'] = 'reaction'
    reaction['substrate_distance_measured'] = _column(
        reaction, 'substrate_distance', 0.0)

    homologs = expansion.rename(columns=RENAME).copy() if len(expansion) else \
        pd.DataFrame(columns=['uniprot'])
    if len(homologs):
        homologs['discovery_route'] = homologs.evidence_strength.map(
            {'family_homolog': 'domain', 'clade_homolog': 'clade'}).fillna('domain')
        homologs['substrate_distance_inherited'] = (
            INHERITED_DISCOUNT * homologs.family_substrate_distance)
        homologs['self_condensation'] = _column(
            homologs, 'family_self_condensing', False)
        # A protein found by a reaction keeps the stronger record.
        homologs = homologs[~homologs.uniprot.isin(set(reaction.uniprot))]

    frames = [reaction, homologs]
    for extra in extra_sources:
        if extra is None or not len(extra):
            continue
        extra = extra.copy()
        seen = set(pd.concat(frames, ignore_index=True).uniprot.dropna())
        frames.append(extra[~extra.uniprot.isin(seen)])
    table = pd.concat(frames, ignore_index=True)
    for column in ('substrate_distance_measured', 'substrate_distance_inherited'):
        table[column] = pd.to_numeric(_column(table, column, 0.0),
                                      errors='coerce').fillna(0.0)

    structural = load_structural() if structural is None else structural
    table = table.merge(structural, on='uniprot', how='left')

    table['self_condensation'] = (pd.to_numeric(
        _column(table, 'self_condensation', False), errors='coerce')
        .fillna(0).astype(bool))
    reviewed = _column(table, 'reviewed').fillna('').astype(str).str.lower()
    table['is_reviewed'] = reviewed.str.startswith('reviewed')
    table['has_pdb'] = _truthy(_column(table, 'pdb'))
    observed = _column(table, 'protein_existence').fillna('').astype(str)
    table['protein_level_evidence'] = observed.str.contains(
        'protein level', case=False)

    names = (_column(table, 'protein_name').fillna('').astype(str) + ' '
             + _column(table, 'ec').fillna('').astype(str)).str.lower()
    disqualified = names.str.contains('|'.join(DISQUALIFYING), regex=True)
    qualified = names.str.contains('|'.join(QUALIFYING), regex=True)
    # Only homologs are challenged this way. A reaction-anchored protein was
    # proven by the screen on its own reaction, which outranks its name.
    table['annotation_contradicts_chemistry'] = (
        disqualified & ~qualified & table.discovery_route.ne('reaction'))

    distance = table[['substrate_distance_measured',
                      'substrate_distance_inherited']].max(axis=1)
    table['master_priority'] = (
        table.evidence_strength.map(EVIDENCE_WEIGHT).fillna(0.0)
        + 3.0 * distance
        + 2.0 * table.self_condensation.astype(int)
        + 1.0 * table.is_reviewed.astype(int)
        + 1.0 * table.has_pdb.astype(int)
        + 0.5 * table.protein_level_evidence.astype(int)
        - CONTRADICTION_PENALTY
        * table.annotation_contradicts_chemistry.astype(int)
        + MEASURED_ACTIVITY_BONUS
        * pd.to_numeric(_column(table, 'is_strong_producer', False),
                        errors='coerce').fillna(0).astype(float)
    ).round(4)

    for column in COLUMNS:
        if column not in table.columns:
            table[column] = pd.NA
    return (table[COLUMNS]
            .sort_values(['master_priority', 'uniprot'],
                         ascending=[False, True])
            .drop_duplicates('uniprot')
            .reset_index(drop=True))


def main():
    from .paths import ensure_directories
    ensure_directories()
    reaction = pd.read_csv(RES / 'candidates_ranked.tsv', sep='\t')
    expansion_path = RES / 'family_expansion.tsv'
    expansion = (pd.read_csv(expansion_path, sep='\t', low_memory=False)
                 if expansion_path.exists() else pd.DataFrame())
    details_path = RES / 'enzymes_uniprot.tsv'
    details = (pd.read_csv(details_path, sep='\t', low_memory=False)
               if details_path.exists() else None)
    extras = []
    for name in ('precomputed_link_candidates.tsv', 'reference_actives.tsv'):
        path = RES / name
        if path.exists():
            extras.append(pd.read_csv(path, sep='\t', low_memory=False))
    table = build(reaction, expansion,
                  reaction_details=details,
                  sequences=read_fasta(RES / 'enzymes.fasta'),
                  extra_sources=extras)
    table.to_csv(RES / 'master_candidates.tsv', sep='\t', index=False)

    summary = {
        'total_candidates': int(len(table)),
        'by_route': table.discovery_route.value_counts().to_dict(),
        'by_evidence': table.evidence_strength.value_counts().to_dict(),
        'reviewed': int(table.uniprot.isin(
            table[table.reviewed.astype(str).str.lower()
                  .str.startswith('reviewed')].uniprot).sum()),
        'with_pdb': int(_truthy(table.pdb).sum()),
        'with_alphafold': int(_truthy(table.alphafold).sum()),
        'with_catalytic_activity': int(_truthy(table.catalytic_activity).sum()),
        'with_sequence': int(_truthy(table.sequence).sum()),
        'with_structural_score': int(table.struct_S.notna().sum()),
        'self_condensing': int(table.self_condensation.sum()),
        'annotation_contradicts_chemistry': int(
            table.annotation_contradicts_chemistry.sum()),
    }
    print(json.dumps(summary, indent=2), flush=True)
    (RES / 'master_candidates_summary.json').write_text(
        json.dumps(summary, indent=2) + '\n')

    show = ['uniprot', 'gene', 'discovery_route', 'evidence_strength',
            'substrate_distance_measured', 'substrate_distance_inherited',
            'struct_S', 'master_priority']
    print('\nTop 25:', flush=True)
    print(table.head(25)[show].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
