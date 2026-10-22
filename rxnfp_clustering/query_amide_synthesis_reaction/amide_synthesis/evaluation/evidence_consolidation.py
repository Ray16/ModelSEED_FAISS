"""Consolidate multiple independent evidence sources per reaction-to-enzyme mapping.

A single annotation (UniProt enzyme -> reaction) is the base of every panel mapping.
Robustness improves when that base is corroborated by *independent* lines of evidence:
primary experimental literature, independent curated reaction databases, and
EC-class agreement. This module gathers every source that is actually available for
each of the ten panel reactions, counts the independent axes, and flags mappings that
rest on a single source -- including the glycylglycine case, where the enzyme (BacD)
is well supported for its OWN reaction but only inferred for the target reaction.

Offline and deterministic: reads the saved panel mapping table, the saved UniProt
snapshot for the inferred candidate, and the pinned ModelSEED reaction aliases/ECs.
No network calls, no new retrieval. Agreement among sources does not prove the
mapping is biochemically correct for adipate/HMDA; it measures corroboration breadth.
"""
import csv
import json
import re
from pathlib import Path

from amide_synthesis.infrastructure.paths import DB, RES

# Reaction sources are not equally reliable. Rank them by how directly each curates
# reaction chemistry, so corroboration is weighted rather than counted flat:
#   1 expert-curated, reaction-level, ChEBI-structured (Rhea)
#   2 curated biochemical / pathway databases (MetaCyc, KEGG, EcoCyc)
#   3 genome-scale metabolic reconstructions (BiGG and named iXXX models)
#   4 automated cross-database reconciliation -- least reliable (MetaNetX)
SOURCE_RELIABILITY = {
    'Rhea': 1,
    'MetaCyc': 2, 'KEGG': 2, 'EcoCyc': 2,
    'BiGG': 3, 'iAF1260': 3, 'iSO783': 3, 'iYO844': 3, 'iJO1366': 3, 'iML1515': 3,
    'metanetx.reaction': 4,
}
# "Independent curated reaction DBs" for corroboration counts the directly-curated
# tiers (1-2) only. Reconstructions and the MetaNetX aggregator are recorded for
# transparency but deliberately excluded from the corroboration count.
CURATED_REACTION_DBS = {s for s, t in SOURCE_RELIABILITY.items() if t <= 2}
MODEL_OR_AGGREGATOR = {s for s, t in SOURCE_RELIABILITY.items() if t >= 3}


def reliability_tier(source):
    """Lower is more reliable; unknown sources are treated as the weakest tier."""
    return SOURCE_RELIABILITY.get(source, 4)

PANEL_LINKS = ('results/top10_mapping/20261020_chemical_panel/'
               'supported_reaction_links.tsv')
INFERRED_SNAPSHOT = 'results/top10_mapping/glycylglycine_evidence/raw/P39641.json'


def read_aliases():
    """Map each reaction id to the set of external sources that name it."""
    sources, ecs = {}, {}
    with (DB / 'Aliases/Unique_ModelSEED_Reaction_Aliases.txt').open() as handle:
        reader = csv.reader(handle, delimiter='\t')
        next(reader, None)
        for row in reader:
            if len(row) >= 3:
                sources.setdefault(row[0], set()).add(row[2])
    ec_path = DB / 'Aliases/Unique_ModelSEED_Reaction_ECs.txt'
    if ec_path.exists():
        with ec_path.open() as handle:
            reader = csv.reader(handle, delimiter='\t')
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    ecs.setdefault(row[0], set()).add(row[1])
    return sources, ecs


def ligase_class(ec):
    """The first three EC levels where defined, so provisional 4th digits still match."""
    out = set()
    for value in ec:
        parts = (value or '').split('.')
        if len(parts) >= 3 and parts[0] == '6':
            out.add('.'.join(parts[:3]))
    return out


def parse_reaction_evidence(raw):
    """Distinct Rhea ids, PubMed ids and EC strings from the saved reaction_evidence."""
    rhea, pubmed, tiers = set(), set(), set()
    for item in json.loads(raw or '[]'):
        if item.get('rhea_id'):
            rhea.add(str(item['rhea_id']))
        if item.get('evidence_tier'):
            tiers.add(item['evidence_tier'])
        for evidence in item.get('evidence', []):
            if evidence.get('source') == 'PubMed' and evidence.get('id'):
                pubmed.add(str(evidence['id']))
    return rhea, pubmed, tiers


def consolidate_row(link, aliases, ecs):
    """Evidence axes for one exact reaction-annotated mapping."""
    rxn = link['rxn_id']
    # Follow the approved MetaCyc/Rhea bridge so rxn47618 inherits rxn54616's registries.
    registries = set(aliases.get(rxn, set()))
    bridged = link.get('source_reaction')
    if bridged and bridged != rxn:
        registries |= aliases.get(bridged, set())
    curated = sorted(registries & CURATED_REACTION_DBS, key=reliability_tier)
    best_tier = min((reliability_tier(s) for s in registries), default=None)
    rhea, pubmed, tiers = parse_reaction_evidence(link.get('reaction_evidence', ''))
    uniprot_ec = {e for e in re.split(r'[;, ]+', link.get('ec', '')) if e}
    modelseed_ec = ecs.get(rxn, set()) | ({bridged and None} and ecs.get(bridged, set()))
    ec_consistent = bool(ligase_class(uniprot_ec) & ligase_class(modelseed_ec)) if (
        uniprot_ec and modelseed_ec) else None

    axes = {
        'uniprot_reaction_annotation': True,  # every exact row carries this by construction
        'primary_literature': len(pubmed) >= 1,
        'curated_reaction_db': len(curated) >= 1,
        'ec_class_consistent': bool(ec_consistent),
    }
    satisfied = sum(bool(v) for v in axes.values())
    return {
        'rxn_id': rxn,
        'accession': link['accession'],
        'enzyme': link.get('protein_name', ''),
        'gene': link.get('gene_names', ''),
        'organism': link.get('organism', ''),
        'assignment_type': 'exact_native_annotation',
        'mapping_route': link.get('mapping_route', ''),
        'chemical_status': link.get('chemical_status', ''),
        'evidence_tier': ';'.join(sorted(tiers)) or 'curated_direct',
        'n_primary_pubmed_refs': len(pubmed),
        'pubmed_ids': ';'.join(sorted(pubmed)),
        'rhea_ids': ';'.join(sorted(rhea)),
        'n_curated_reaction_dbs': len(curated),
        'curated_reaction_dbs': ';'.join(curated),
        'best_source_reliability_tier': best_tier,
        'model_or_aggregator_dbs': ';'.join(sorted(registries & MODEL_OR_AGGREGATOR,
                                                   key=reliability_tier)),
        'uniprot_ec': ';'.join(sorted(uniprot_ec)),
        'modelseed_ec': ';'.join(sorted(modelseed_ec)),
        'ec_class_consistent': ec_consistent,
        **{'axis_' + k: v for k, v in axes.items()},
        'n_independent_evidence_axes': satisfied,
        'corroboration': ('multi_source' if satisfied >= 3 else
                          'supported' if satisfied == 2 else 'single_source'),
    }


def inferred_row(snapshot_path, target_rxn, aliases, ecs):
    """The glycylglycine candidate: strong evidence for its OWN reaction, inference for the target."""
    data = json.loads(Path(snapshot_path).read_text())
    description = data.get('proteinDescription', {})
    own_ec, pubmed, own_rhea = set(), set(), set()
    recommended = description.get('recommendedName', {})
    for node in [recommended, *description.get('alternativeNames', [])]:
        for ec in node.get('ecNumbers', []):
            own_ec.add(ec.get('value'))
    for comment in data.get('comments', []):
        reaction = comment.get('reaction', {})
        for ref in reaction.get('reactionCrossReferences', []):
            if str(ref.get('id', '')).startswith('RHEA:'):
                own_rhea.add(ref['id'].split(':')[1])
        for evidence in reaction.get('evidences', []):
            if evidence.get('source') == 'PubMed' and evidence.get('id'):
                pubmed.add(str(evidence['id']))
    target_registries = aliases.get(target_rxn, set())
    target_curated = sorted(target_registries & CURATED_REACTION_DBS, key=reliability_tier)
    return {
        'rxn_id': target_rxn,
        'accession': data.get('primaryAccession', 'P39641'),
        'enzyme': recommended.get('fullName', {}).get('value', ''),
        'gene': ';'.join(g.get('geneName', {}).get('value', '') for g in data.get('genes', [])),
        'organism': data.get('organism', {}).get('scientificName', ''),
        'assignment_type': 'inferred_related_reaction',
        'mapping_route': 'ec_inferred',
        'chemical_status': 'no_exact_native_assignment',
        'evidence_tier': 'inference_only',
        # These counts describe the enzyme's OWN (alanine-anticapsin) reaction, not the target.
        'n_primary_pubmed_refs': len(pubmed),
        'pubmed_ids': ';'.join(sorted(pubmed)),
        'rhea_ids': ';'.join(sorted(own_rhea)),
        'n_curated_reaction_dbs': len(target_curated),
        'curated_reaction_dbs': ';'.join(target_curated),
        'best_source_reliability_tier': min(
            (reliability_tier(s) for s in target_registries), default=None),
        'model_or_aggregator_dbs': ';'.join(sorted(target_registries & MODEL_OR_AGGREGATOR,
                                                   key=reliability_tier)),
        'uniprot_ec': ';'.join(sorted(e for e in own_ec if e)),
        'modelseed_ec': ';'.join(sorted(ecs.get(target_rxn, set()))),
        'ec_class_consistent': None,
        'axis_uniprot_reaction_annotation': False,
        'axis_primary_literature': False,
        'axis_curated_reaction_db': len(target_curated) >= 1,
        'axis_ec_class_consistent': False,
        'n_independent_evidence_axes': 0,
        'corroboration': 'inference_only',
        'note': ('BacD/YwfE is an experimentally characterised dipeptide ATP-grasp ligase '
                 '(alanine--anticapsin, EC 6.3.2.49); its support does not establish the '
                 'glycylglycine target reaction. Kept as an explicitly inferred candidate.'),
    }


def build():
    aliases, ecs = read_aliases()
    links = list(csv.DictReader((RES.parent / PANEL_LINKS).open(), delimiter='\t'))
    rows = [consolidate_row(link, aliases, ecs) for link in links]
    rows.append(inferred_row(RES.parent / INFERRED_SNAPSHOT, 'rxn09897', aliases, ecs))
    rows.sort(key=lambda r: (-r['n_independent_evidence_axes'], -r['n_curated_reaction_dbs'],
                             -r['n_primary_pubmed_refs'], r['rxn_id']))
    return rows


def summarise(rows):
    exact = [r for r in rows if r['assignment_type'] == 'exact_native_annotation']
    return {
        'n_mappings': len(rows),
        'n_exact_native_annotation': len(exact),
        'n_inferred_related_reaction': sum(
            r['assignment_type'] == 'inferred_related_reaction' for r in rows),
        'corroboration_breakdown': {
            label: sum(r['corroboration'] == label for r in rows)
            for label in ('multi_source', 'supported', 'single_source', 'inference_only')},
        'exact_with_primary_literature': sum(r['axis_primary_literature'] for r in exact),
        'exact_with_curated_reaction_db': sum(r['axis_curated_reaction_db'] for r in exact),
        'exact_ec_class_consistent': sum(r['axis_ec_class_consistent'] for r in exact),
        'single_source_exact_mappings': sorted(
            r['accession'] for r in exact if r['corroboration'] == 'single_source'),
        'interpretation': (
            'Counts corroboration breadth, not catalytic accuracy. Every exact mapping has a '
            'UniProt reaction annotation; independent axes (primary literature, curated reaction '
            'databases, EC-class agreement) vary. "inference_only" flags the glycylglycine '
            'candidate, which has no exact native assignment.'),
    }


def figure(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size': 18, 'savefig.dpi': 300, 'figure.dpi': 300})

    axes_keys = ['axis_uniprot_reaction_annotation', 'axis_primary_literature',
                 'axis_curated_reaction_db', 'axis_ec_class_consistent']
    axes_labels = ['UniProt\nannotation', 'Primary\nliterature',
                   'Curated\nreaction DB', 'EC-class\nconsistent']
    matrix = np.array([[1.0 if row[k] else 0.0 for k in axes_keys] for row in rows])
    labels = [f"{r['rxn_id']}  {(r['gene'].split(';')[0] or r['accession'])}" for r in rows]

    fig, ax = plt.subplots(figsize=(8.5, 0.62 * len(rows) + 2.2))
    ax.imshow(matrix, cmap='RdYlBu_r', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(axes_labels)))
    ax.set_xticklabels(axes_labels)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    for i, row in enumerate(rows):
        for j, key in enumerate(axes_keys):
            mark = '✓' if row[key] else '–'
            ax.text(j, i, mark, ha='center', va='center',
                    color='white' if row[key] else '#555555', fontsize=18)
        tag = ('inferred' if row['assignment_type'] == 'inferred_related_reaction'
               else f"{row['n_curated_reaction_dbs']} DB / {row['n_primary_pubmed_refs']} ref")
        ax.text(len(axes_keys) - 0.35, i, tag, ha='left', va='center', fontsize=18)
    ax.set_xlim(-0.5, len(axes_keys) + 1.6)
    ax.set_xlabel('Independent evidence axis')
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)


def main():
    rows = build()
    out_dir = RES / 'top10_mapping'
    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) + ['note'] if 'note' not in rows[0] else list(rows[0])
    fieldnames = list(dict.fromkeys(sum((list(r) for r in rows), [])))
    with (out_dir / 'panel_evidence_consolidation.tsv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in fieldnames})
    summary = summarise(rows)
    (out_dir / 'panel_evidence_consolidation.json').write_text(json.dumps(
        {'summary': summary, 'mappings': rows}, indent=2))
    figure(rows, out_dir / 'panel_evidence_consolidation.png')
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
