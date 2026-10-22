"""Replay the current target → five reactions → filtered proteins without an agent.

Inputs are immutable database snapshots and explicit curated configuration.
Recompute the full mechanistic screen, fingerprint ranking, chemical selection,
reaction-scoped mapping, alignments and clustering in a new output directory.
No network fallback, manual shortlist editing or structure submissions occur.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import pandas as pd

from .infrastructure.paths import PROJECT

DEFAULT_LOCK = PROJECT / 'config/top10_reproduction.lock.json'
PACKAGES = ['numpy', 'pandas', 'rdkit', 'biopython', 'drfp', 'requests', 'matplotlib', 'pyarrow']


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def environment():
    return {'python': platform.python_version(),
            'packages': {name: metadata.version(name) for name in PACKAGES}}


def verify_hashes(root, entries):
    for name, expected in entries.items():
        path = Path(root) / name
        if not path.is_file() or digest(path) != expected:
            raise ValueError('Pinned input missing or changed: ' + str(path))


def compare_table(actual, expected, columns=None, keys=None):
    """Compare scientific values, allowing floating-point roundoff, not selection drift."""
    left, right = [pd.read_csv(p, sep='\t', keep_default_na=False) for p in [actual, expected]]
    if columns is not None:
        left, right = left[columns], right[columns]
    if keys:
        left, right = [frame.sort_values(keys).reset_index(drop=True) for frame in [left, right]]
    pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=False,
                                  rtol=1e-6, atol=1e-7)
    return len(left)


def export_representatives(mapping):
    candidates = pd.read_csv(mapping / 'reaction_candidates.tsv', sep='\t', keep_default_na=False)
    members = pd.read_csv(mapping / 'cluster_membership.tsv', sep='\t')
    counts = pd.read_csv(mapping / 'counts.tsv', sep='\t')
    for threshold in [99, 95]:
        selected = members[(members.pool == 'expanded') & (members.identity == threshold / 100)]
        keys = set(zip(selected.rxn_id, selected.representative))
        rows = candidates[[pair in keys for pair in zip(candidates.rxn_id, candidates.accession)]]
        rows = rows.sort_values(['rxn_id', 'accession'])
        if len(rows) != int(counts['expanded_clusters' + str(threshold)].sum()):
            raise ValueError('Representative export differs from cluster counts')
        rows.to_csv(mapping / f'representatives{threshold}.tsv', sep='\t', index=False)
        (mapping / f'representatives{threshold}.fasta').write_text(''.join(
            f'>{r.accession}|{r.rxn_id}|{r.mapping_route}\n{r.sequence}\n'
            for r in rows.itertuples()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock', type=Path, default=DEFAULT_LOCK)
    parser.add_argument('--output', type=Path, help='New, empty directory; defaults to a timestamped run')
    parser.add_argument('--check-only', action='store_true', help='Verify snapshots, tools and versions without running')
    args = parser.parse_args(argv)
    lock = json.loads(args.lock.read_text())
    verify_hashes(PROJECT, lock['input_sha256'])
    if environment() != lock['environment']:
        raise ValueError('Runtime differs from the pinned environment; use the documented environment')
    snapshot = PROJECT / lock['mapping_snapshot']
    count = lock.get('reaction_count', 5)
    selection_policy = lock.get('selection_policy', 'legacy_monomer')
    original = json.loads((snapshot / 'manifest.json').read_text())
    verify_hashes(snapshot, original['output_sha256'])
    if args.check_only:
        print('Pinned inputs, complete response snapshot, tool binaries and package versions verified.')
        return 0

    output = (args.output or PROJECT / 'results/reproducibility' /
              datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Reproduction output must be new or empty; completed runs are immutable')
    output.mkdir(parents=True, exist_ok=True)
    logs = output / 'logs'
    logs.mkdir()
    # Bound CPU use and eliminate inherited options that could change scientific rules.
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false', PYTHONHASHSEED='0',
               N_WORKERS='2', SUBSTRATE_DISTANCE_FLOOR='0.0')
    for variable in ['AMIDE_SYNTHESIS_WORKDIR', 'MODELSEED_BIOCHEMISTRY', 'UNIPROT_MSD_LINKS']:
        env.pop(variable, None)
    commands = []

    def run(name, command, extra_env=None):
        print(name + ' …', flush=True)
        commands.append({'stage': name, 'argv': command, 'environment': extra_env or {}})
        with (logs / (name + '.log')).open('w') as handle:
            subprocess.run(command, cwd=PROJECT, env=dict(env, **(extra_env or {})),
                           stdout=handle, stderr=subprocess.STDOUT, check=True)

    try:
        # The normalized corpus and embeddings are pinned intermediate inputs. Their
        # builder is reaction_search.rebuild_corpus; they contain no agent selections.
        work = output / 'screen'
        for folder in ['data', 'embeddings', 'results']:
            (work / folder).mkdir(parents=True)
        for name in ['data/modelseed_master.parquet', 'embeddings/ecfp4_drfp_master.npy',
                     'results/corpus_manifest.json']:
            (work / name).symlink_to(PROJECT / name)
        run('mechanistic_screen', [sys.executable, '-m', 'amide_synthesis.reaction_search.amide_enzymes'],
            {'AMIDE_SYNTHESIS_WORKDIR': str(work)})
        actual_screen = work / 'results/amide_forming_neighbours.tsv'
        a = set(pd.read_csv(actual_screen, sep='\t').rxn_id)
        b = set(pd.read_csv(PROJECT / 'results/amide_forming_neighbours.tsv', sep='\t').rxn_id)
        if a != b:
            raise ValueError('Recomputed eligible reaction set differs from the pinned baseline')
        ranking = output / 'reactions'
        run('reaction_ranking', [sys.executable, str(PROJECT / 'rank_modelseed_nylon.py'),
                                '--output', str(ranking), '--count', str(count), '--selection-policy', selection_policy, '--screen', str(actual_screen),
                                '--rxnfp-python', lock['rxnfp_python']])
        mapping = output / 'mapping'
        run('enzyme_mapping', [sys.executable, '-m', 'amide_synthesis.evaluation.top5_mapping_audit',
                               '--output', str(mapping), '--top5', str(ranking / f'top{count}_reactions.tsv'),
                               '--profiles', str(PROJECT / lock['mapping_profiles']),
                               '--snapshot', str(snapshot / 'raw'), '--cd-hit', lock['cd_hit']])
        export_representatives(mapping)
        checks = {'eligible_reaction_set': len(a)}
        reference = PROJECT / lock['ranking_reference']
        columns = ['rxn_id', 'rxn_smiles', 'target_query', 'amide_rank',
                   'drfp_tanimoto', 'hybrid_cosine', 'organic_transformation_group']
        for name in ['all_amide_scores.tsv', 'top100_reactions.tsv', f'top{count}_reactions.tsv']:
            checks[name] = compare_table(ranking / name, reference / name, columns)
        for name, keys in [('counts.tsv', ['rxn_id']),
                           ('reaction_candidates.tsv', ['rxn_id', 'accession']),
                           ('supported_reaction_links.tsv', ['rxn_id', 'accession']),
                           ('rejected_search_hits.tsv', ['rxn_id', 'accession']),
                           ('cluster_membership.tsv', ['rxn_id', 'pool', 'identity', 'accession'])]:
            check_columns = list(pd.read_csv(snapshot / name, sep='\t', nrows=0).columns)
            if name == 'counts.tsv':
                check_columns.remove('reaction')  # Display names can improve without changing scientific counts.
            checks[name] = compare_table(mapping / name, snapshot / name, columns=check_columns, keys=keys)
        checks['chemical_audit'] = (json.loads((mapping / 'chemical_audit.json').read_text()) ==
                                    json.loads((snapshot / 'chemical_audit.json').read_text()))
        if not checks['chemical_audit']:
            raise ValueError('Recomputed full-equation chemistry differs from frozen evidence')
        # Detect inputs edited during a long run, rather than blessing mixed inputs.
        verify_hashes(PROJECT, lock['input_sha256'])
        from .infrastructure.freshness import state, status
        freshness = {name: status(name) for name in state()['stages']}
        if any(value != 'current' for value in freshness.values()):
            raise ValueError('A tracked production artifact is stale: ' + json.dumps(freshness))
        summary = json.loads((mapping / 'summary.json').read_text())
        summary.update(reproduction_checks=checks, tracked_production_stages=freshness,
                       model_panel_status='This replay ends with filtered pools; structural campaigns are tracked separately',
                       ten_per_reaction_capacity_before_model_QC={r['rxn_id']: min(10, r['expanded_clusters95'])
                                                                  for r in summary['counts']})
        (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        from html import escape
        rows = ''.join('<tr>' + ''.join('<td>' + escape(str(r[k])) + '</td>' for k in [
            'rxn_id', 'reaction', 'live_reaction_supported_accessions', 'expanded_candidates',
            'expanded_exact_sequences', 'expanded_clusters95']) + '</tr>' for r in summary['counts'])
        (output / 'report.html').write_text(
            '<!doctype html><meta charset="utf-8"><title>Reproducible nylon enzyme discovery</title>'
            '<style>body{font:16px/1.5 system-ui;max-width:1100px;margin:40px auto}td,th{padding:10px;text-align:left}img{width:100%}</style>'
            '<h1>Reproducible target-to-enzyme discovery</h1>'
            '<p>Recomputed from pinned local inputs and saved UniProt/Rhea/ChEBI responses. '
            'No network fallback or agent selection. All scientific regression checks passed.</p>'
            '<p><a href="reactions/report.html">Reaction selection and structures</a> · '
            '<a href="mapping/report.html">Mapping evidence and redundancy</a> · '
            '<a href="mapping/representatives95.fasta">95% representatives FASTA</a> · '
            '<a href="manifest.json">Inputs, environment and commands</a></p>'
            '<table><tr><th>ModelSEED</th><th>Reaction</th><th>Supported IDs</th><th>Expanded IDs</th>'
            '<th>Exact sequences</th><th>95% representatives</th></tr>' + rows + '</table>'
            '<img src="mapping/mapping_counts.svg">'
            '<p>Native-reaction annotation and inferred candidates remain distinct. '
            'Modeling is tracked separately; this replay submits no predictions.</p>')
        manifest = {
            'status': 'complete', 'generated_utc': datetime.now(timezone.utc).isoformat(),
            'scope': f'Pinned corpus → recomputed screen → top100 → {count} chemically selected reactions → mapped/filtered proteins',
            'lock_sha256': digest(args.lock), 'environment': environment(), 'commands': commands,
            'fixed_environment': {k: env[k] for k in ['N_WORKERS', 'SUBSTRATE_DISTANCE_FLOOR',
                 'PYTHONHASHSEED', 'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']},
            'input_sha256': lock['input_sha256'],
            'code_sha256': {str(p.relative_to(PROJECT)): digest(p) for p in sorted(
                list((PROJECT / 'amide_synthesis').rglob('*.py')) + list(PROJECT.glob('*.py')))},
            'regression_checks': checks,
            'reproducibility': 'Scientific tables and cluster memberships; plot metadata, timestamps and paths may differ',
            'output_sha256': {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob('*'))
                              if p.is_file() and not p.is_symlink()},
        }
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print('Verified reproduction: ' + str(output / 'report.html'), flush=True)
        return 0
    except Exception as error:
        (output / 'failure.json').write_text(json.dumps({'status': 'failed', 'error': str(error),
                                                        'commands': commands}, indent=2) + '\n')
        raise
