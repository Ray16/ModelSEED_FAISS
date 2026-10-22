#!/usr/bin/env python
"""Independently audit mapping artifacts against their frozen source snapshots.

This deliberately does not import the production reaction/evidence parser.
It checks reconstruction consistency, not experimental catalytic accuracy.
"""
import argparse
from collections import Counter, defaultdict
import csv
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import re

PROJECT = Path(__file__).resolve().parents[1]
REVIEW = PROJECT / 'results/colleague_review_20261017'
ORDER = {'experimental_direct': 0, 'curated_direct': 1,
         'homology_supported_direct': 2, 'computational_direct': 3}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_mapping(path):
    result = {}
    for line in path.open():
        accession, reactions, sequence = line.rstrip('\n').split('\t')
        if accession in result:
            raise ValueError('Duplicate mapping accession: ' + accession)
        result[accession] = (set(reactions.split(',')),
                             hashlib.sha256(sequence.encode()).hexdigest())
    return result


def scoped_annotations(text):
    """Separate fields by their semantic headers before inspecting any IDs."""
    result = defaultdict(set)
    for part in text.split('CATALYTIC ACTIVITY:')[1:]:
        for scope in re.split(r'(?=PhysiologicalDirection=)', part):
            crossrefs = re.search(r'Xref=([^;]+);', scope)
            if crossrefs is None:
                continue
            evidence = re.search(r'Evidence=(\{[^}]*\});', scope)
            value = evidence.group(1) if evidence else ''
            codes = set(re.findall(r'ECO:\d+', value))
            if 'ECO:0000269' in codes:
                tier = 'experimental_direct'
            elif 'ECO:0000250' in codes:
                tier = 'homology_supported_direct'
            elif codes & {'ECO:0000255', 'ECO:0000256'}:
                tier = 'computational_direct'
            else:
                tier = 'curated_direct'
            for ref in crossrefs.group(1).split(','):
                match = re.fullmatch(r'\s*Rhea:(RHEA:\d+)\s*', ref)
                if match:
                    result[match.group(1)].add((tier, value))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mapping-dir', type=Path, default=REVIEW / 'mapping')
    parser.add_argument('--output-prefix', type=Path, required=True)
    args = parser.parse_args()
    csv.field_size_limit(10 ** 8)
    source = REVIEW / 'evidence/uniprot_reviewed_rhea_snapshot.tsv.gz'
    direct_path = args.mapping_dir / 'uniprot-msd-direct-reviewed.tsv'
    exp_path = args.mapping_dir / 'uniprot-msd-experimental-reviewed.tsv'
    evidence_path = args.mapping_dir / 'direct_mapping_evidence.tsv.gz'
    mapping = read_mapping(direct_path)
    experimental = read_mapping(exp_path)
    metadata = {}
    aliases = defaultdict(set)
    dbfiles = sorted((PROJECT.parents[1] / 'ModelSEEDDatabase/Biochemistry').glob('reaction_*.tsv'))
    for path in dbfiles:
        for row in csv.DictReader(path.open(), delimiter='\t'):
            metadata[row['id']] = row
            if row['is_obsolete'] != '0':
                continue
            for group in row.get('aliases', '').split('|'):
                if group.startswith('Rhea: '):
                    for value in group[6:].split(';'):
                        if value.strip().isdigit():
                            aliases['RHEA:' + value.strip()].add(row['id'])
    evidence = defaultdict(list)
    evidence_rows = 0
    pairs = set()
    tiers = {}
    exp_pairs = set()
    problems = Counter(evidence_scope_mismatch_rows=0, evidence_tier_mismatch_rows=0)
    with gzip.open(evidence_path, 'rt') as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            evidence_rows += 1
            accession, rxn, rid = row['accession'], row['modelseed_reaction'], row['rhea_reaction']
            evidence[accession].append((rxn, rid, row['evidence_tier'], row['evidence'],
                                        row['sequence_sha256'], int(row['alias_fanout'])))
            pair = (accession, rxn)
            pairs.add(pair)
            if ORDER[row['evidence_tier']] < ORDER.get(tiers.get(pair), 99):
                tiers[pair] = row['evidence_tier']
            if row['evidence_tier'] == 'experimental_direct':
                exp_pairs.add(pair)
            problems['invalid_reaction_identifier'] += not bool(re.fullmatch(r'RHEA:\d+', rid))
            problems['missing_or_obsolete_reaction'] += rxn not in metadata or metadata[rxn]['is_obsolete'] != '0'
            problems['alias_mismatch'] += rxn not in aliases[rid]
            problems['fanout_mismatch'] += int(row['alias_fanout']) != len(aliases[rid])
            problems['mapping_sequence_mismatch'] += accession not in mapping or mapping[accession][1] != row['sequence_sha256']
    mapping_pairs = {(a, rxn) for a, (rxns, _) in mapping.items() for rxn in rxns}
    expected_experimental = {(a, rxn) for a, (rxns, _) in experimental.items() for rxn in rxns}
    problems['mapping_pairs_without_evidence'] = len(mapping_pairs - pairs)
    problems['evidence_pairs_without_mapping'] = len(pairs - mapping_pairs)
    problems['experimental_subset_missing_pairs'] = len(exp_pairs - expected_experimental)
    problems['experimental_subset_extra_pairs'] = len(expected_experimental - exp_pairs)
    problems['experimental_sequence_mismatch'] = sum(a not in mapping or seq != mapping[a][1]
        for a, (_, seq) in experimental.items())
    differences = []
    expected_pairs = set()
    expected_exp_pairs = set()
    expected_tiers = {}
    source_n = 0
    source_accessions = set()
    with gzip.open(source, 'rt') as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            source_n += 1
            accession = row['Entry']
            source_accessions.add(accession)
            scopes = scoped_annotations(row['Catalytic activity'])
            seqhash = hashlib.sha256(row['Sequence'].encode()).hexdigest()
            for rid, supports in scopes.items():
                for rxn in aliases[rid]:
                    pair = (accession, rxn)
                    expected_pairs.add(pair)
                    best = min((tier for tier, _ in supports), key=ORDER.get)
                    if ORDER[best] < ORDER.get(expected_tiers.get(pair), 99):
                        expected_tiers[pair] = best
                    if any(tier == 'experimental_direct' for tier, _ in supports):
                        expected_exp_pairs.add(pair)
            for rxn, rid, tier, value, observed_seq, fanout in evidence.get(accession, []):
                supports = scopes.get(rid, set())
                problems['source_sequence_mismatch'] += seqhash != observed_seq
                problems['unsupported_source_reaction'] += not supports
                if (tier, value) not in supports:
                    problems['evidence_scope_mismatch_rows'] += 1
                    tier_mismatch = tier not in {t for t, _ in supports}
                    problems['evidence_tier_mismatch_rows'] += tier_mismatch
                    differences.append(dict(accession=accession, reaction=rxn, rhea=rid,
                        observed_tier=tier, expected_support=json.dumps(sorted(supports)),
                        observed_evidence=value, tier_mismatch=tier_mismatch))
    problems['evidence_accession_absent_from_source'] = len(set(evidence) - source_accessions)
    problems['missing_reconstructable_pairs'] = len(expected_pairs - mapping_pairs)
    problems['extra_unreconstructable_pairs'] = len(mapping_pairs - expected_pairs)
    problems['experimental_subset_vs_scoped_source_missing'] = len(expected_exp_pairs - expected_experimental)
    problems['experimental_subset_vs_scoped_source_extra'] = len(expected_experimental - expected_exp_pairs)
    problems['best_pair_tier_mismatches'] = sum(tier != expected_tiers.get(pair)
                                                for pair, tier in tiers.items())
    # An independent structured-JSON cross-check on the existing focused/sample
    # audit. This is deliberately labeled nonrandom for the rebuilt mapping.
    structured = Counter()
    for path in sorted((REVIEW / 'evidence/uniprot_mapping_snapshot').glob('*.json')):
        accession = path.stem
        if accession not in mapping:
            continue
        row = json.loads(path.read_text())
        if row.get('unavailable'):
            structured['unavailable_records'] += 1
            continue
        structured['proteins_checked'] += 1
        structured['sequence_mismatches'] += hashlib.sha256(row['sequence']['value'].encode()).hexdigest() != mapping[accession][1]
        refs = set()
        for comment in row.get('comments', []):
            if comment.get('commentType') == 'CATALYTIC ACTIVITY':
                for ref in comment.get('reaction', {}).get('reactionCrossReferences', []):
                    if re.fullmatch(r'RHEA:\d+', ref['id']):
                        refs.add(ref['id'])
                for direction in comment.get('physiologicalReactions', []):
                    ref = direction.get('reactionCrossReference', {})
                    if re.fullmatch(r'RHEA:\d+', ref.get('id', '')):
                        refs.add(ref['id'])
        for rxn in mapping[accession][0]:
            structured['pairs_checked'] += 1
            structured['pairs_without_exact_json_rhea'] += not any(rxn in aliases[rid] for rid in refs)
    fanout = []
    for rid, rxns in sorted(aliases.items()):
        if len(rxns) > 1:
            for rxn in sorted(rxns):
                fanout.append(dict(rhea=rid, reaction=rxn, name=metadata[rxn]['name'],
                                   definition=metadata[rxn]['definition']))
    fanout_rxns = {rxn for rxns in aliases.values() if len(rxns) > 1 for rxn in rxns}
    mapped_rxns = {rxn for _, rxn in mapping_pairs}
    statuses = Counter(metadata[rxn]['status'] for rxn in mapped_rxns)
    selected = list(csv.DictReader((REVIEW / 'sequences/selected50.tsv').open(), delimiter='\t'))
    frozen_panel = dict(proteins=len(selected),
        pairs_missing_from_mapping=sum((r['accession'], r['reaction']) not in mapping_pairs for r in selected),
        sequence_hash_mismatches=sum(mapping.get(r['accession'], (set(), ''))[1] != r['sequence_sha256'] for r in selected))
    result = dict(generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='Reconstruction/annotation consistency against frozen inputs; no global biochemical-accuracy estimate.',
        source_proteins=source_n, mapped_proteins=len(mapping), mapped_pairs=len(mapping_pairs),
        mapped_reactions=len({rxn for _, rxn in mapping_pairs}), evidence_rows=evidence_rows,
        best_pair_evidence_tiers=dict(Counter(tiers.values())),
        expected_best_pair_tiers=dict(Counter(expected_tiers.values())),
        experimental_pairs=len(expected_experimental), expected_scoped_experimental_pairs=len(expected_exp_pairs),
        problems=dict(sorted(problems.items())), structured_json_crosscheck=dict(structured),
        structured_json_scope='All available previously audited proteins overlapping rebuilt mapping; nonrandom for this mapping.',
        alias_fanout_rhea_ids=len({r['rhea'] for r in fanout}),
        fanout_pairs=sum(rxn in fanout_rxns for _, rxn in mapping_pairs),
        modelseed_reaction_status_counts=dict(statuses),
        modelseed_non_OK_reactions=sum(n for status, n in statuses.items() if status != 'OK'),
        modelseed_non_OK_pairs=sum(metadata[rxn]['status'] != 'OK' for _, rxn in mapping_pairs),
        status_interpretation='Raw ModelSEED flags only; non-OK includes unknown/symbolic formula and charge issues, not a measured false-assignment rate.',
        frozen_50_panel=frozen_panel,
        sha256={str(p.resolve().relative_to(PROJECT)):digest(p) for p in [direct_path, exp_path, evidence_path, source]},
        source_database_sha256={str(p.relative_to(PROJECT.parents[1])):digest(p) for p in dbfiles},
        audit_script_sha256=digest(Path(__file__).resolve()))
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n')
    for suffix, rows, fields in [('_evidence_scope_differences.tsv', differences,
            ['accession','reaction','rhea','observed_tier','expected_support','observed_evidence','tier_mismatch']),
            ('_alias_fanout.tsv', fanout, ['rhea','reaction','name','definition'])]:
        with Path(str(args.output_prefix)+suffix).open('w') as handle:
            writer = csv.DictWriter(handle, fields, delimiter='\t')
            writer.writeheader(); writer.writerows(rows)
    print(json.dumps({k:v for k,v in result.items() if k not in ['sha256','source_database_sha256']}, indent=2))
    structured_failures = any(structured[key] for key in
        ['unavailable_records', 'sequence_mismatches', 'pairs_without_exact_json_rhea'])
    panel_failures = frozen_panel['pairs_missing_from_mapping'] or frozen_panel['sequence_hash_mismatches']
    return int(bool(any(problems.values()) or structured_failures or panel_failures))


if __name__ == '__main__':
    raise SystemExit(main())
