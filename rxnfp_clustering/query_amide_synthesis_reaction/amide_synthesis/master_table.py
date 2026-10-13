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
    'exact_reaction': 3.0,   # UniProt annotated to that Rhea reaction
    'activity_ec': 2.0,      # matching EC, broader
    'name_match': 1.0,       # distinctive product name
    'family_homolog': 1.0,   # enriched Pfam family
    'clade_homolog': 1.0,    # UniRef50 neighbourhood of a validated protein
}

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
    'family', 'pfam', 'interpro', 'seed_pfam', 'clade_seed',
    'rxn_id', 'reaction_name', 'query', 'cosine_similarity',
    'ec', 'catalytic_activity', 'cofactor', 'pathway', 'function',
    'reviewed', 'protein_existence', 'length', 'pdb', 'alphafold',
    'binding_site', 'active_site', 'keywords', 'go_function', 'go_process',
    'subunit', 'lineage', 'sequence',
]


def _truthy(series):
    return series.fillna('').astype(str).str.strip().ne('')


def load_structural(path=STRUCTURAL_SCORES):
    """Independently computed geometric scores, if the file is present."""
    if not Path(path).exists():
        return pd.DataFrame(columns=['uniprot', 'struct_S'])
    frame = pd.read_csv(path)
    return frame.rename(columns={'id': 'uniprot', 'score': 'struct_S'})[
        ['uniprot', 'struct_S']]


def build(reaction_candidates, expansion, structural=None):
    """Merge the routes into one ranked table. No I/O."""
    reaction = reaction_candidates.copy()
    reaction['discovery_route'] = 'reaction'
    reaction['substrate_distance_measured'] = reaction.get(
        'substrate_distance', 0.0)

    homologs = expansion.rename(columns=RENAME).copy() if len(expansion) else \
        pd.DataFrame(columns=['uniprot'])
    if len(homologs):
        homologs['discovery_route'] = homologs.evidence_strength.map(
            {'family_homolog': 'domain', 'clade_homolog': 'clade'}).fillna('domain')
        homologs['substrate_distance_inherited'] = (
            INHERITED_DISCOUNT * homologs.family_substrate_distance)
        homologs['self_condensation'] = homologs.get('family_self_condensing', False)
        # A protein found by a reaction keeps the stronger record.
        homologs = homologs[~homologs.uniprot.isin(set(reaction.uniprot))]

    table = pd.concat([reaction, homologs], ignore_index=True)
    for column in ('substrate_distance_measured', 'substrate_distance_inherited'):
        table[column] = pd.to_numeric(table.get(column), errors='coerce').fillna(0.0)

    structural = load_structural() if structural is None else structural
    table = table.merge(structural, on='uniprot', how='left')

    table['self_condensation'] = table.get(
        'self_condensation', False).fillna(False).astype(bool)
    reviewed = table.get('reviewed', '').fillna('').astype(str).str.lower()
    table['is_reviewed'] = reviewed.str.startswith('reviewed')
    table['has_pdb'] = _truthy(table.get('pdb', pd.Series('', index=table.index)))
    observed = table.get('protein_existence', '').fillna('').astype(str)
    table['protein_level_evidence'] = observed.str.contains(
        'protein level', case=False)

    names = (table.protein_name.fillna('').astype(str) + ' '
             + table.get('ec', pd.Series('', index=table.index))
             .fillna('').astype(str)).str.lower()
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
    table = build(reaction, expansion)
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
        'self_condensing': int(table.self_condensation.fillna(False).sum()),
        'annotation_contradicts_chemistry': int(
            table.annotation_contradicts_chemistry.fillna(False).sum()),
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
