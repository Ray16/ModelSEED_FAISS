"""Pipeline orchestration; each stage runs in a separate Python process."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from amide_synthesis.infrastructure.paths import PROJECT, LOGS, ensure_directories
from amide_synthesis.infrastructure.freshness import RULES, status
from .stages import STAGES, Stage, PHASES




# Order is dependency order, and `--only` is reordered to match it.
#
# Two ordering constraints are load-bearing and were each a bug once:
#   * `enrich` must run AFTER `master` and writes its OWN file, because
#     rewriting master_candidates.tsv in place let a later master run silently
#     discard every enriched value.
#   * `refs` must not read master_candidates.tsv to decide what is missing -
#     that made its output depend on whether a later stage had already run.


def select_stages(*, only=None, skip_corpus=False, offline=False, phases=None):
    """Select stages in dependency order, regardless of the order of --only."""
    return tuple(stage for stage in STAGES
                 if (only is None or stage.name in only)
                 and (phases is None or stage.phase in phases)
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

def expand_stale_dependencies(stages, *, offline=False):
    """Add stale producers, in pipeline order, rather than consume old tables."""
    by_name={s.name:s for s in STAGES};wanted={s.name for s in stages}
    memo={}
    def add(name):
        if name not in RULES:return False
        if name in memo:return memo[name]
        changed=False
        consumed=set(RULES[name][0])
        for parent,(_,outputs) in RULES.items():
            if parent==name or not consumed.intersection(outputs):continue
            ancestor_changed=add(parent)
            if status(parent)=='current' and not ancestor_changed:continue
            if offline and by_name[parent].needs_uniprot:
                raise RuntimeError(f'{name} needs stale {parent}; refresh that stage with network access first')
            wanted.add(parent);changed=True
        memo[name]=changed
        return changed
    for stage in list(stages):add(stage.name)
    return tuple(s for s in STAGES if s.name in wanted)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-corpus', action='store_true',
                        help='reuse the existing corpus and embeddings')
    parser.add_argument('--offline', action='store_true',
                        help='reuse UniProt outputs and cached Pfam names; no HTTP')
    parser.add_argument('--only', nargs='+', choices=[s.name for s in STAGES],
                        help='run only these stages, in pipeline order')
    parser.add_argument('--phase', nargs='+', choices=list(PHASES),
                        help='select scientific phases; stale dependencies are added on execution')
    parser.add_argument('--python', default=os.environ.get('PIPELINE_PYTHON', sys.executable),
                        help='stage interpreter (default: current Python or PIPELINE_PYTHON)')
    parser.add_argument('--dry-run', action='store_true', help='list stages without running them')
    args = parser.parse_args(argv)
    stages = select_stages(only=args.only, skip_corpus=args.skip_corpus, offline=args.offline, phases=args.phase)
    # Dry-run remains a pure stage listing. Real execution repairs stale
    # prerequisites automatically; stage entry points independently enforce it.
    if not args.dry_run:
        try:stages=expand_stale_dependencies(stages,offline=args.offline)
        except RuntimeError as exc:
            print(str(exc),file=sys.stderr);return 1

    env = dict(os.environ)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        env.setdefault(key, '4')
    if args.offline:
        env['AMIDE_SYNTHESIS_OFFLINE'] = '1'
    print(f'python   {args.python}')
    print(f'workers  N_WORKERS={env.get("N_WORKERS", "auto (70% of idle cores, cap 48)")}')
    if args.dry_run:
        for stage in stages:
            print(f'{stage.phase:20} {stage.name:9} {stage.description}')
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
        from amide_synthesis.infrastructure.paths import RES
        print(f'Candidates: {RES / "candidates_ranked.tsv"}')
    return 0
