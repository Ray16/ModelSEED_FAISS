"""Discover selected-reaction Rhea links from exact local aliases and MetaCyc.

Every proposed match is compared to the full Rhea equation. No numerical
participant IDs, names or EC numbers can create a reaction equivalence.
"""
import argparse
import csv
import json
from pathlib import Path
import re

from amide_synthesis.enzyme_mapping.uniprot_snapshot import SnapshotClient, protein_metadata, reaction_annotations
from amide_synthesis.enzyme_mapping.reaction_chemistry import rhea_structure, audit_equation
from amide_synthesis.evaluation.top5_mapping_audit import write_table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reactions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    client = SnapshotClient(args.output / 'raw')
    rows = list(csv.DictReader(args.reactions.open(), delimiter='\t'))
    profiles, checks, supported = {}, [], []
    for row in rows:
        rid = row['rxn_id']
        print('Discovering aliases', rid, flush=True)
        rhea = set()
        for section in row['aliases'].split('|'):
            if section.startswith('Rhea:'):
                rhea.update(re.findall(r'\d+', section.partition(':')[2]))
        discovered = []
        for section in row['aliases'].split('|'):
            if not section.startswith('MetaCyc:'):
                continue
            for term in section.partition(':')[2].split(';'):
                term = term.strip()
                body, _ = client.get(term + '.tsv', 'https://www.rhea-db.org/rhea',
                    {'query': term, 'columns': 'rhea-id,equation,reaction-xref(MetaCyc),ec', 'format': 'tsv'})
                for found in csv.DictReader(body.splitlines(), delimiter='\t'):
                    if 'MetaCyc:' + term not in found.get('Cross-reference (MetaCyc)', '').split(';'):
                        continue
                    rhea.add(found['Reaction identifier'].removeprefix('RHEA:'))
                    discovered.append(found)
        if not rhea:
            checks.append(dict(rxn_id=rid, status='no_exact_Rhea_alias_found'))
            continue
        # Prefer the master returned by MetaCyc lookup; local alias order is not a claim of direction.
        reference = discovered[0]['Reaction identifier'].removeprefix('RHEA:') if discovered else min(rhea, key=int)
        sides, participants = rhea_structure(client, reference)
        try:
            chemistry = audit_equation(row['rxn_smiles'], sides)
        except ValueError as error:
            if 'symbolic' not in str(error):
                raise
            chemistry = {'status': 'reference_structure_symbolic', 'qualification': str(error)}
        checks.append(dict(rxn_id=rid, rhea=reference, participants=participants, **chemistry))
        # Query the explicit directional family from the Rhea graph.
        query = f'''PREFIX rh: <http://rdf.rhea-db.org/>
SELECT ?reaction WHERE {{ rh:{reference} (rh:directionalReaction|rh:bidirectionalReaction) ?reaction . }}'''
        body, _ = client.get('family_' + reference + '.json', 'https://sparql.rhea-db.org/sparql',
                             {'query': query, 'format': 'json'})
        rhea.update(x['reaction']['value'].rsplit('/', 1)[-1]
                    for x in json.loads(body)['results']['bindings'])
        ec = sorted({x for found in discovered for x in re.findall(r'\d+\.\d+\.\d+\.\d+', found.get('EC number',''))})
        profiles[rid] = dict(rhea=sorted(rhea, key=int), ec=ec[0] if len(ec)==1 else '',
                             genes=[], name_terms=[], alias_source='ModelSEED/Rhea exact MetaCyc cross-reference',
                             chemical_status=chemistry['status'])
        records, _ = client.proteins(rid + '_direct', '(' + ' OR '.join('rhea:' + r for r in profiles[rid]['rhea']) + ')')
        for record in records:
            statements = reaction_annotations(record, profiles[rid]['rhea'])
            if statements:
                supported.append(dict(rxn_id=rid, **protein_metadata(record),
                    chemical_status=chemistry['status'], reaction_evidence=json.dumps(statements)))
        (args.output / 'chemical_checks.json').write_text(json.dumps(checks, indent=2) + '\n')
        (args.output / 'mapping_profiles.json').write_text(json.dumps({
            'schema_version':1, 'alignment': {'identity_min':.3, 'coverage_each_min':.8, 'mode':'local',
                'substitution_matrix':'BLOSUM62','open_gap_score':-10,'extend_gap_score':-.5},
            'reactions': profiles}, indent=2) + '\n')
        write_table(args.output / 'supported_preview.tsv', supported)
        print(rid, reference, chemistry['status'], 'supported', sum(r['rxn_id']==rid for r in supported), flush=True)
    (args.output / 'chemical_checks.json').write_text(json.dumps(checks, indent=2) + '\n')
    if len(profiles) != len(rows):
        raise ValueError('Some selected reactions lack an exact Rhea alias; see chemical_checks.json')


if __name__ == '__main__':
    main()
