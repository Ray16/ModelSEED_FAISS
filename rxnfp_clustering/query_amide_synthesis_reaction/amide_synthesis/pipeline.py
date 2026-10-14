"""Pipeline orchestration; each stage runs in a separate Python process."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass

from .paths import PROJECT, LOGS, ensure_directories


@dataclass(frozen=True)
class Stage:
    name: str
    module: str
    description: str
    needs_uniprot: bool = False


# Order is dependency order, and `--only` is reordered to match it.
#
# Two ordering constraints are load-bearing and were each a bug once:
#   * `enrich` must run AFTER `master` and writes its OWN file, because
#     rewriting master_candidates.tsv in place let a later master run silently
#     discard every enriched value.
#   * `refs` must not read master_candidates.tsv to decide what is missing -
#     that made its output depend on whether a later stage had already run.
STAGES = (
    Stage('corpus', 'rebuild_corpus', 'rebuild reactions + ECFP4/DRFP'),
    Stage('search', 'encode_and_search', 'encode queries, cosine nearest neighbours'),
    Stage('screen', 'amide_enzymes', 'mechanism screens + substrate distance'),
    Stage('enzymes', 'fetch_enzymes', 'resolve UniProt enzymes (network)', True),
    Stage('table', 'candidate_table', 'prioritised candidate table (+ family)'),
    Stage('families', 'expand_families',
          'expand validated families by domain / clade (network)', True),
    Stage('links', 'precomputed_links',
          'ingest the external UniProt->ModelSEED link table'),
    Stage('refs', 'reference_actives',
          'assayed panel + literature-curated proteins (network)', True),
    Stage('neighbours', 'chemistry_neighbours',
          'borrow enzymes across chemically equivalent reactions'),
    Stage('names', 'enzyme_names',
          'candidates named for the target function in UniProt (network)', True),
    # BEFORE master: master consumes `substrate_distance_own_rhea` to rank, so
    # this reads the route files rather than the master table. See
    # ec_substrate.candidate_annotations.
    Stage('actives', 'active_homologs',
          'candidates reached from the MEASURED actives (network)', True),
    Stage('kegg', 'kegg_orthologs',
          'resolve unresolved reactions through KEGG orthology (network)',
          True),
    Stage('ecsub', 'ec_substrate',
          'substrate distance from each protein\'s own Rhea reaction (network)',
          True),
    Stage('master', 'master_table', 'merge every route, keeping provenance'),
    Stage('enrich', 'enrich',
          'fill annotation for bare candidates; UniParc for the panel (network)',
          True),
    Stage('pdb', 'pdb_evidence', 'experimental structures per candidate (network)',
          True),
    Stage('geometry', 'adenylation_geometry',
          'adenylation geometry in cofactor-bound structures (network)', True),
    Stage('ligands', 'ligand_evidence',
          'classify crystallographically bound ligands (network)', True),
    Stage('members', 'member_similarity',
          'discriminate members within a family (tie-break only)'),
    Stage('famval', 'family_evaluation',
          'judge each family on its own seeds\' chemistry (network)', True),
    Stage('properties', 'curated_properties',
          'curated kinetics / location / domains from UniProt (network)', True),
    Stage('shortlist', 'shortlist', 'bench-ready diverse shortlist + FASTA'),
    Stage('models', 'model_quality',
          'AlphaFold confidence + curated properties on the shortlist (network)',
          True),
    Stage('converged', 'converged',
          'candidates backed by several INDEPENDENT lines of evidence'),
    Stage('dossier', 'dossier',
          'full dossier per converged candidate (network)', True),
    Stage('native', 'native_substrates', 'native reactions + family grouping'),
    Stage('deckfigs', 'deck_figures', 'family and substrate-gap figures'),
    Stage('figure', 'project_tsne', 't-SNE map with candidates overlaid'),
)


def select_stages(*, only=None, skip_corpus=False, offline=False):
    """Select stages in dependency order, regardless of the order of --only."""
    return tuple(stage for stage in STAGES
                 if (only is None or stage.name in only)
                 and not (skip_corpus and stage.name == 'corpus')
                 and not (offline and stage.needs_uniprot))


def run_stage(stage, python, env, log_dir=LOGS):
    """Stream stdout/stderr to the terminal and the stage log; return exit code."""
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / f'{stage.name}.log').open('w') as handle:
        handle.write(f'# {stage.name}: {stage.description}\n'
                     f'# {time.strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        with subprocess.Popen(
            [python, '-u', '-m', f'amide_synthesis.{stage.module}'],
            cwd=PROJECT, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        ) as process:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                handle.write(line)
                handle.flush()
            return process.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-corpus', action='store_true',
                        help='reuse the existing corpus and embeddings')
    parser.add_argument('--offline', action='store_true',
                        help='reuse UniProt outputs and cached Pfam names; no HTTP')
    parser.add_argument('--only', nargs='+', choices=[s.name for s in STAGES],
                        help='run only these stages, in pipeline order')
    parser.add_argument('--python', default=os.environ.get('PIPELINE_PYTHON', sys.executable),
                        help='stage interpreter (default: current Python or PIPELINE_PYTHON)')
    parser.add_argument('--dry-run', action='store_true', help='list stages without running them')
    args = parser.parse_args(argv)
    stages = select_stages(only=args.only, skip_corpus=args.skip_corpus, offline=args.offline)

    env = dict(os.environ)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        env.setdefault(key, '4')
    if args.offline:
        env['AMIDE_SYNTHESIS_OFFLINE'] = '1'
    print(f'python   {args.python}')
    print(f'workers  N_WORKERS={env.get("N_WORKERS", "auto (70% of idle cores, cap 48)")}')
    if args.dry_run:
        for stage in stages:
            print(f'{stage.name:9} {stage.description}')
        return 0

    ensure_directories()
    timings = []
    overall = time.monotonic()
    for stage in stages:
        print(f'== {stage.name}: {stage.description}', flush=True)
        start = time.monotonic()
        try:
            code = run_stage(stage, args.python, env)
        except OSError as exc:
            print(f'Could not start {stage.name}: {exc}', file=sys.stderr)
            code = 1
        elapsed = time.monotonic() - start
        timings.append((stage.name, elapsed, code))
        print(f'   {"done" if code == 0 else "FAILED"} in {elapsed:.1f}s\n', flush=True)
        if code:
            break

    for name, elapsed, code in timings:
        print(f'  {name:9} {elapsed:7.1f}s  {"ok" if code == 0 else "FAILED"}')
    print(f'  {"TOTAL":9} {time.monotonic() - overall:7.1f}s')
    if timings and timings[-1][2]:
        return 1
    if any(name == 'table' for name, _, _ in timings):
        from .paths import RES
        print(f'Candidates: {RES / "candidates_ranked.tsv"}')
    return 0
