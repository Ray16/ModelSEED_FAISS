"""Audit current top-five mappings, retrieve complete pools and remove redundancy.

Run with ``python -m amide_synthesis.evaluation.top5_mapping_audit``. Raw live
responses, chemical checks, rejected matches, cluster membership and hashes are
preserved together. The production mapping is not rewritten by this audit.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from Bio import Align
from Bio.Align import substitution_matrices
import pandas as pd

from amide_synthesis.infrastructure.paths import PROJECT
from amide_synthesis.enzyme_mapping.uniprot_snapshot import (
    SnapshotClient, protein_metadata, reaction_annotations,
)
from amide_synthesis.enzyme_mapping.reaction_chemistry import rhea_structure, audit_equation
from amide_synthesis.enzyme_mapping.reaction_equivalence import equation_key, equation_hash
from amide_synthesis.candidate_selection.sequence_redundancy import cluster_proteins


SPECS = {
    'rxn05123': {'rhea': ['13633', '13634', '13635', '13636'], 'ec': '6.3.1.11',
                 'genes': ['puuA'], 'name_terms': ['glutamylputrescine', 'glutamate--putrescine']},
    'rxn46045': {'rhea': ['44340', '44341', '44342', '44343'], 'ec': '6.3.2.46',
                 'genes': ['ddaG'], 'name_terms': ['fumaroyl-dap', 'fumarate--(s)-2,3-diaminopropanoate']},
    'rxn21038': {'rhea': ['63808', '63809', '63810', '63811'], 'ec': '',
                 'genes': ['asbA'], 'name_terms': ['spermidine-citrate ligase', 'petrobactin biosynthesis protein asba']},
    'rxn42510': {'rhea': ['59136', '59137', '59138', '59139'], 'ec': '6.3.2.58',
                 'genes': ['sfnaD', 'sfaD'], 'name_terms': ['d-ornithine--citrate ligase']},
    'rxn47618': {'rhea': ['59124', '59125', '59126', '59127'], 'ec': '6.3.2.54',
                 'genes': ['sbnE'], 'name_terms': ['diaminopropanoate--citrate ligase', 'staphyloferrin b biosynthesis protein sbne']},
}


def write_table(path, rows):
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w') as handle:
        writer = csv.DictWriter(handle, columns, delimiter='\t', extrasaction='raise')
        writer.writeheader()
        writer.writerows(rows)


def mapping_subset(path, wanted):
    result = {rxn: {} for rxn in wanted}
    with Path(path).open() as handle:
        for line in handle:
            accession, reactions, sequence = line.rstrip('\n').split('\t')
            for reaction in set(reactions.split(',')).intersection(wanted):
                if accession in result[reaction] and result[reaction][accession] != sequence:
                    raise ValueError('Conflicting sequences in mapping: ' + accession)
                result[reaction][accession] = sequence
    return result


def reference_alignment(sequence, references):
    aligner = Align.PairwiseAligner()
    aligner.mode = 'local'
    aligner.substitution_matrix = substitution_matrices.load('BLOSUM62')
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    best = None
    for ref in references:
        alignment = aligner.align(sequence, ref['sequence'])[0]
        matches, aligned = 0, 0
        for (a, b), (c, d) in zip(*alignment.aligned):
            matches += sum(x == y for x, y in zip(sequence[a:b], ref['sequence'][c:d]))
            aligned += b - a
        row = {'reference_accession': ref['accession'],
               'identity': matches / max(1, alignment.length),
               'query_coverage': aligned / len(sequence),
               'reference_coverage': aligned / len(ref['sequence'])}
        if best is None or row['identity'] * min(row['query_coverage'], row['reference_coverage']) > best['identity'] * min(best['query_coverage'], best['reference_coverage']):
            best = row
    return best or {}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--cd-hit', type=Path, default=Path('/homes/rzhu/miniforge3/envs/palm/bin/cd-hit'))
    args = parser.parse_args(argv)
    output = args.output or PROJECT / 'results/top5_mapping_audit' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    if (output / 'manifest.json').exists():
        raise FileExistsError('Completed audit snapshots are immutable; choose a new --output directory')
    output.mkdir(parents=True, exist_ok=True)
    client = SnapshotClient(output / 'raw')
    top5_path = PROJECT / 'results/modelseed_nylon_top100/top5_reactions.tsv'
    top5 = list(csv.DictReader(top5_path.open(), delimiter='\t'))
    if len(top5) != 5 or set(r['rxn_id'] for r in top5) != set(SPECS):
        raise ValueError('The highlighted five changed; audit specifications need scientific review')
    policy = json.loads((PROJECT / 'mapping_policy.json').read_text())
    old = mapping_subset(PROJECT / 'uniprot-msd-links.tsv', set(SPECS))
    direct = mapping_subset(PROJECT / policy['default_links'], set(SPECS))
    corpus = pd.read_parquet(PROJECT / 'data/modelseed_master.parquet', columns=['rxn_id', 'rxn_smiles'])
    equations = corpus.set_index('rxn_id').rxn_smiles.to_dict()
    chemistry, query_meta, summary, all_rows, rejected, cluster_rows = [], {}, [], [], [], []
    per_reaction = {}
    for hit in top5:
        reaction = hit['rxn_id']
        spec = SPECS[reaction]
        print('Auditing ' + reaction, flush=True)
        rhea_sides, participants = rhea_structure(client, spec['rhea'][0])
        chemical = audit_equation(hit['rxn_smiles'], rhea_sides)
        chemical.update(rxn_id=reaction, rhea_id=spec['rhea'][0], participants=participants,
                        local_definition=hit['definition'], local_smiles=hit['rxn_smiles'])
        chemistry.append(chemical)
        if chemical['status'] == 'chemical_conflict':
            (output / 'chemical_audit.json').write_text(json.dumps(chemistry, indent=2))
            raise ValueError('Full reaction chemistry conflicts with Rhea: ' + reaction)
        terms = ['rhea:' + r for r in spec['rhea']]
        if spec['ec']:
            terms.append('ec:' + spec['ec'])
        terms.extend('gene_exact:' + g for g in spec['genes'])
        terms.extend('protein_name:"' + n + '"' for n in spec['name_terms'])
        records, query_meta[reaction] = client.proteins(reaction, '(' + ' OR '.join(terms) + ')')
        parsed = [(record, protein_metadata(record), reaction_annotations(record, spec['rhea'])) for record in records]
        references = [metadata for _, metadata, statements in parsed if statements]
        if not references:
            raise ValueError('No reaction-annotated reference available: ' + reaction)
        kept, trusted = [], []
        for record, metadata, statements in parsed:
            accession = metadata['accession']
            ec_match = bool(spec['ec']) and spec['ec'] in metadata['ec'].split(';')
            name_match = any(n in metadata['all_names'].lower() for n in spec['name_terms'])
            alignment = reference_alignment(metadata['sequence'], references) if not statements else {}
            # Names and EC annotations remain hypotheses. Gene-only matches are rejected.
            passes_alignment = bool(alignment) and alignment['identity'] >= 0.3 and min(alignment['query_coverage'], alignment['reference_coverage']) >= 0.8
            route = ('reaction_annotation' if statements else
                     'ec_inferred' if ec_match and passes_alignment else
                     'name_and_homology_inferred' if name_match and passes_alignment else 'rejected')
            row = dict(rxn_id=reaction, **metadata, mapping_route=route,
                       chemical_status=chemical['status'],
                       source_reaction='rxn54616' if reaction == 'rxn47618' else reaction,
                       reaction_evidence=json.dumps(statements, separators=(',', ':')),
                       **alignment)
            if route == 'rejected':
                row['rejection_reason'] = ('no_specific_reaction_EC_or_protein_name' if not (ec_match or name_match)
                                           else 'insufficient_reference_identity_or_coverage')
                rejected.append(row)
                continue
            kept.append(row)
            if statements:
                trusted.append(row)
        # Require exact snapshot identifier and sequence agreement for all saved direct links.
        live = {r['accession']: r['sequence'] for r in trusted}
        if direct[reaction] != {a: live.get(a) for a in direct[reaction]}:
            raise ValueError('Saved mapping sequence absent/changed in live exact-reaction records')
        per_reaction[reaction] = {'trusted': trusted, 'expanded': kept}
        row = {'rxn_id': reaction, 'reaction': hit['display_title'],
               'legacy_accessions': len(old[reaction]), 'saved_direct_accessions': len(direct[reaction]),
               'live_reaction_supported_accessions': len(trusted), 'expanded_candidates': len(kept),
               'rejected_search_hits': sum(r['rxn_id'] == reaction for r in rejected),
               'chemical_status': chemical['status'],
               'reviewed_expanded': sum(r['reviewed'] for r in kept),
               'fragment_expanded': sum(r['fragment'] for r in kept)}
        for label, pool in [('trusted', trusted), ('expanded', kept)]:
            for identity in [0.99, 0.95]:
                clustered = cluster_proteins(pool, output / 'clusters' / reaction / label,
                                            args.cd_hit, identity=identity)
                row[label + '_exact_sequences'] = clustered['exact_sequences']
                row[label + '_clusters' + str(round(identity * 100))] = clustered['representatives']
                cluster_rows.extend(dict(rxn_id=reaction, pool=label, identity=identity, **r) for r in clustered['members'])
        summary.append(row)
        all_rows.extend(kept)
        print(json.dumps(row), flush=True)
    # Verify the missing alias through an independent live MetaCyc cross-reference.
    body, _ = client.get('metacyc_RXN-19820.tsv', 'https://www.rhea-db.org/rhea',
                         {'query': 'RXN-19820', 'columns': 'rhea-id,equation,reaction-xref(MetaCyc),ec', 'format': 'tsv'})
    bridge_rows = list(csv.DictReader(body.splitlines(), delimiter='\t'))
    if not any(r['Reaction identifier'] == 'RHEA:59124' and 'MetaCyc:RXN-19820' in r['Cross-reference (MetaCyc)'] for r in bridge_rows):
        raise ValueError('Missing independent Rhea/MetaCyc bridge support')
    if equation_key(equations['rxn47618']) != equation_key(equations['rxn54616']):
        raise ValueError('Proposed equivalent ModelSEED reactions differ')
    write_table(output / 'counts.tsv', summary)
    write_table(output / 'reaction_candidates.tsv', all_rows)
    write_table(output / 'rejected_search_hits.tsv', rejected)
    write_table(output / 'cluster_membership.tsv', cluster_rows)
    write_table(output / 'supported_reaction_links.tsv', [r for r in all_rows if r['mapping_route'] == 'reaction_annotation'])
    (output / 'chemical_audit.json').write_text(json.dumps(chemistry, indent=2))
    (output / 'query_snapshots.json').write_text(json.dumps(query_meta, indent=2))
    union = {}
    trusted_union = {}
    for pools in per_reaction.values():
        for r in pools['expanded']:
            if r['accession'] in union and union[r['accession']]['sequence'] != r['sequence']:
                raise ValueError('Sequence changed between reaction snapshots')
            union[r['accession']] = r
        trusted_union.update({r['accession']: r for r in pools['trusted']})
    totals = {'unique_trusted_accessions': len(trusted_union), 'unique_expanded_accessions': len(union),
              'expanded_reaction_protein_pairs': len(all_rows),
              'trusted_reaction_protein_pairs': sum(r['live_reaction_supported_accessions'] for r in summary)}
    for label, pool in [('trusted', list(trusted_union.values())), ('expanded', list(union.values()))]:
        for identity in [0.99, 0.95]:
            clustered = cluster_proteins(pool, output / 'clusters/union' / label, args.cd_hit, identity=identity)
            totals[label + '_unique_sequences'] = clustered['exact_sequences']
            totals[label + '_clusters' + str(round(identity * 100))] = clustered['representatives']
    releases = {x['release'] for x in query_meta.values()}
    if len(releases) != 1:
        raise ValueError('Different UniProt releases in one audit')
    result = {'generated_utc': datetime.now(timezone.utc).isoformat(), 'uniprot_release': next(iter(releases)),
              'counts': summary, 'totals': totals, 'status': 'complete',
              'bridge_equation_sha256': equation_hash(equations['rxn47618']),
              'clustering': {'method': 'CD-HIT 4.8.1, accurate greedy representative clustering',
                             'identity': 'identical residues / full shorter-sequence length',
                             'coverage_each_sequence': 0.9, 'thresholds': [0.99, 0.95]},
              'limits': ['Reaction annotations concern native activity, not adipate/HMDA activity.',
                         'EC/name/homology-supported candidates are not verified reaction assignments.',
                         'Compatible missing stereochemistry is reported explicitly; no global chemical-certification claim.',
                         'Clusters can join strains/species; same taxonomy or proteome alone never causes a merge.',
                         'Sequence variation within a proteome can represent a true paralog, not just a duplicate record.']}
    (output / 'summary.json').write_text(json.dumps(result, indent=2))
    from amide_synthesis.evaluation.mapping_audit_report import render
    render(result, output)
    inputs = [top5_path, PROJECT / policy['default_links'], PROJECT / 'uniprot-msd-links.tsv',
              PROJECT / 'data/modelseed_master.parquet', Path(__file__)]
    inputs += list((PROJECT / 'amide_synthesis/enzyme_mapping').glob('reaction_*.py'))
    inputs += [PROJECT / 'amide_synthesis/enzyme_mapping/uniprot_snapshot.py',
               PROJECT / 'amide_synthesis/candidate_selection/sequence_redundancy.py',
               PROJECT / 'amide_synthesis/evaluation/mapping_audit_report.py']
    manifest = {'status': 'complete', 'generated_utc': result['generated_utc'],
                'input_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                'output_sha256': {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'manifest.json'}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('Report: ' + str(output / 'report.html'), flush=True)
    print(json.dumps(totals, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
