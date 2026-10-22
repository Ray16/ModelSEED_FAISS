"""Join current supported proteins to an existing campaign and inspect every pose.

Read-only with respect to the campaign: no submission, restart or model selection.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import yaml

from amide_synthesis.infrastructure.paths import PROJECT
from amide_synthesis.structure_modeling.poses import inspect


def audit(supported, campaign):
    manifest_path = campaign / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    references, pending, hashes = [], [], {}
    items = manifest['inputs']
    if isinstance(items, dict):
        items = list(items.values())
    for record in supported:
        matches = [item for item in items if item['accession'] == record['accession']]
        identity = {'reaction': record['rxn_id'], 'accession': record['accession'],
                    'enzyme_name': record['protein_name'], 'gene_names': record['gene_names'],
                    'organism': record['organism'], 'sequence_sha256': record['sequence_sha256']}
        if not matches:
            pending.append(dict(identity, status='not_in_campaign'))
            continue
        for item in matches:
            sequence = yaml.safe_load(Path(item['input']).read_text())['sequences'][0]['protein']['sequence']
            if hashlib.sha256(sequence.encode()).hexdigest() != record['sequence_sha256']:
                raise ValueError('Campaign/current protein sequence mismatch: ' + record['accession'])
            directory = campaign / 'outputs' / item['name']
            if not list(directory.glob('by_seed/*/s42/*_model_*.pdb')):
                pending.append(dict(identity, status='no_saved_models'))
                continue
            checked = inspect(item, directory, include_near_attack=False)
            references.append(dict(identity, ensemble=item['name'], actual_models_checked=checked['n_models'],
                                   angle_deg=checked['A_deg'], harmonic_distance_A=checked['Dh_A'],
                                   assembly_qc=checked['assembly_qc']))
            for path in [Path(item['input']), *directory.glob('by_seed/*/s42/*.pdb'),
                         *directory.glob('by_seed/*/s42/confidence_*.json')]:
                hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {'checked_utc': datetime.now(timezone.utc).isoformat(),
            'scope': 'Existing protein + ATP + Mg + adipate ensembles; no HMDA or activated intermediate',
            'campaign_manifest': str(manifest_path.resolve()),
            'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            'references': references, 'pending': pending,
            'completed_proteins': len({r['accession'] for r in references}),
            'actual_models_checked': sum(r['actual_models_checked'] for r in references),
            'current_candidate_sequence_hashes_match': True, 'model_input_output_sha256': hashes,
            'interpretation': 'Composition and sequence checks do not certify biological assembly or catalytic activity.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--supported', type=Path, default=PROJECT / 'results/top5_mapping_audit/20261020_mapping_review/supported_reaction_links.tsv')
    parser.add_argument('--campaign', type=Path, default=PROJECT / 'results/boltz2_activation_20261016')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError('Choose a new --output; preserve previous checks')
    records = list(csv.DictReader(args.supported.open(), delimiter='\t'))
    result = audit(records, args.campaign)
    result['supported_input_sha256'] = hashlib.sha256(args.supported.read_bytes()).hexdigest()
    result['code_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(f"{result['completed_proteins']} proteins / {result['actual_models_checked']} models; "
          f"{len(result['pending'])} pending. {args.output}")


if __name__ == '__main__':
    main()
