"""Reservation-gated Boltz runner for the step-2 (aminolysis) panel.

Mirrors the step-1 fleet's lambda13 preflight, gpu_reserve gating, disjoint-CPU
workers, claim-based resumability and bounded retries, but verifies the step-2
complex (acyl intermediate + HMDA + Mg, plus ADP on the acyl-phosphate route)
instead of the ATP+adipate activation complex. Near-attack geometry is computed
separately once models exist; this runner only produces and composition-checks them.
"""
import argparse
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

import gemmi

from amide_synthesis.structure_modeling.fleet import (
    GATE, KIT, PYTHON, NATIVE, write, record, starttime, available_cpus, msa_ready)
from amide_synthesis.infrastructure.paths import PROJECT

RUN = None


def composition(model):
    counts = {}
    for chain in model:
        heavy = [a for r in chain for a in r if a.element.name != 'H']
        counts[chain.name] = {
            'heavy': len(heavy),
            'C': sum(a.element.name == 'C' for a in heavy),
            'N': sum(a.element.name == 'N' for a in heavy),
            'O': sum(a.element.name == 'O' for a in heavy),
            'P': sum(a.element.name == 'P' for a in heavy),
            'Mg': sum(a.element.name == 'Mg' for a in heavy),
        }
    return counts


def inspect_step2(item, out):
    """Confirm all 50 models carry the intended step-2 composition."""
    name = item['name']
    models = sorted(Path(out).glob(f'by_seed/{name}/s42/{name}_model_*.pdb'))
    expected = {f'{name}_model_{i}.pdb' for i in range(50)}
    if {p.name for p in models} != expected:
        raise ValueError(f'Expected 50 unique models; found {len(models)}')
    mg_total = item['mg_count']
    inter = tuple(item['intermediate_heavy'])  # (C, N, O, P) of the acyl intermediate
    adp = item.get('adp', False)
    for path in models:
        if not path.with_name('confidence_' + path.stem + '.json').exists():
            raise ValueError('Missing confidence for ' + path.name)
        model = gemmi.read_structure(str(path))[0]
        counts = composition(model)
        protein = ''.join(gemmi.find_tabulated_residue(r.name).one_letter_code
                          for r in model['A'])
        if protein != item['sequence']:
            raise ValueError('Protein sequence mismatch ' + path.name)
        ligands = [c for k, c in counts.items() if k != 'A']
        if sum(c['Mg'] for c in ligands) != mg_total:
            raise ValueError('Wrong Mg count ' + path.name)
        if not any(c['heavy'] == 8 and c['C'] == 6 and c['N'] == 2 for c in ligands):
            raise ValueError('HMDA (C6N2) not found ' + path.name)
        if not any((c['C'], c['N'], c['O'], c['P']) == inter for c in ligands):
            raise ValueError('Acyl intermediate composition not found ' + path.name)
        if adp and not any(c['heavy'] == 27 and c['P'] == 2 and c['N'] == 5 for c in ligands):
            raise ValueError('ADP not found ' + path.name)
    return {'name': name, 'accession': item['accession'], 'route': item['route'],
            'mg_count': mg_total, 'n_models': 50,
            'composition_qc': 'protein sequence, HMDA, acyl intermediate, Mg'
                              + (' and ADP' if adp else '') + ' verified'}


def predict(item, out, log):
    argv = ['bash', KIT, 'pred', '--config', 'h100', '--mode', 'exact', '--input', item['input'],
            '--out_dir', str(out), '--seed', '42', '--diffusion_samples', '50',
            '--max_parallel_samples', '41', '--recycling_steps', '3', '--sampling_steps', '200',
            '--num_workers', '0', '--preprocessing-threads', '1', '--output_format', 'pdb']
    with Path(log).open('a') as handle:
        child = subprocess.Popen(argv, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)

        def stop(signum, frame):
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait()
            raise SystemExit(128 + signum)
        signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
        code = child.wait()
    if code:
        raise RuntimeError('Predictor exit ' + str(code))


def worker(manifest, gpu):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(gpu):
        raise RuntimeError('GPU visibility does not match the gate assignment')
    if len(os.sched_getaffinity(0)) != 8:
        raise RuntimeError('Worker must have eight assigned CPUs')
    info = record(gpu=gpu, visible_gpu=os.environ['CUDA_VISIBLE_DEVICES'])
    write(RUN / 'state' / f'worker_gpu{gpu}.json', info)
    consecutive = 0
    while True:
        pending = claimed_any = False
        for item in manifest['inputs']:
            name = item['name']; done = RUN / 'state' / f'{name}.done.json'
            if done.exists():
                continue
            fail = RUN / 'state' / f'{name}.failure.json'
            failure = json.loads(fail.read_text()) if fail.exists() else {'attempts': 0, 'retry_after': 0}
            if failure['attempts'] >= 2:
                continue
            pending = True
            if failure['retry_after'] > time.time():
                continue
            claim = RUN / 'state' / f'{name}.claim'
            try:
                claim.mkdir()
            except FileExistsError:
                continue
            claimed_any = True
            try:
                write(claim / 'owner.json', info)
                out = RUN / 'outputs' / name
                try:
                    validated = inspect_step2(item, out)
                except (ValueError, KeyError, RuntimeError):
                    if out.exists():
                        out.rename(RUN / 'outputs' / f'{name}.partial.{int(time.time())}')
                    attempt = failure['attempts'] + 1
                    write(RUN / 'state' / f'{name}.running.json', record(gpu=gpu, attempt=attempt))
                    print('PREDICTION_START', name, 'gpu', gpu, 'attempt', attempt, flush=True)
                    predict(item, out, RUN / 'logs' / f'{name}.attempt{attempt}.log')
                    validated = inspect_step2(item, out)
                write(done, validated); consecutive = 0
                print('COMPLETE', name, flush=True)
            except Exception as exc:  # noqa: BLE001 - bounded retry, recorded
                consecutive += 1
                write(fail, {'attempts': failure['attempts'] + 1, 'error': str(exc),
                             'retry_after': time.time() + 30, **record(gpu=gpu)})
                print('FAILED', name, type(exc).__name__, str(exc), flush=True)
            finally:
                (claim / 'owner.json').unlink(missing_ok=True); claim.rmdir()
            if consecutive >= 3:
                print('CIRCUIT_BREAKER', flush=True); return 2
        if not pending:
            return 0
        if not claimed_any:
            if list((RUN / 'state').glob('*.claim')):
                time.sleep(10)
            else:
                time.sleep(5)


def coordinator(manifest):
    if socket.gethostname() != 'lambda13':
        raise RuntimeError('This exact kit is preflighted on lambda13')
    write(RUN / 'state/coordinator.json', record(status='waiting_for_msa'))
    deadline = time.time() + 10800
    while not msa_ready(manifest):
        if time.time() > deadline:
            raise RuntimeError('MSA readiness timeout')
        time.sleep(10)
    import torch, boltz.main, boltz2_opt  # noqa: F401 - preflight on lambda13
    for cache in ['boltz2_conf.ckpt', 'ccd.pkl', 'mols']:
        if not (Path('/nfs/lambda_stor_01/homes/rzhu/.boltz') / cache).exists():
            raise RuntimeError('Missing checkpoint/cache ' + cache)
    write(RUN / 'evidence/launch_preflight.json', record(status='dependencies_ready',
          torch=torch.__version__, online_cpus=os.cpu_count(), load=os.getloadavg()))
    while True:
        remaining = sum(not (RUN / 'state' / (i['name'] + '.done.json')).exists()
                        for i in manifest['inputs'])
        if not remaining:
            write(RUN / 'state/coordinator.json', record(status='complete'))
            return 0
        result = subprocess.run([GATE, 'list', '--hosts', 'lambda13'], text=True,
                                capture_output=True, timeout=50)
        gpus = [int(l.split()[1]) for l in result.stdout.splitlines() if l.startswith('lambda13 ')]
        cpus = available_cpus()
        count = min(4, len(gpus), remaining, max(0, (len(cpus) - 2) // 8))
        if max(os.getloadavg()) + count * 8 + 2 > os.cpu_count():
            count = 0
        if not count:
            if time.time() > deadline:
                raise RuntimeError('No reservable GPU/CPU capacity before deadline')
            write(RUN / 'state/coordinator.json', record(status='waiting_for_free_gpu_or_cpu',
                  remaining=remaining)); time.sleep(20); continue
        os.sched_setaffinity(0, set(cpus[-2:])); cpus = cpus[:-2]
        workers = []
        for i, gpu in enumerate(gpus[:count]):
            assigned = cpus[i * 8:(i + 1) * 8]
            env = dict(os.environ, **{k: '8' for k in NATIVE}, PYTHONDONTWRITEBYTECODE='1',
                       TOKENIZERS_PARALLELISM='false')
            env['PYTHONPATH'] = str(PROJECT) + os.pathsep + env.get('PYTHONPATH', '')
            argv = ['taskset', '-c', ','.join(map(str, assigned)), GATE, 'run', str(gpu),
                    '--label', 'amide_step2_' + RUN.name, '--', PYTHON, '-m',
                    'amide_synthesis.structure_modeling.step2_fleet', 'worker',
                    '--run-dir', str(RUN), '--gpu', str(gpu)]
            log = (RUN / 'logs' / f'worker_gpu{gpu}.log').open('a')
            proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env,
                                    start_new_session=True)
            workers.append(proc)
            write(RUN / 'state' / f'launch_gpu{gpu}.json', {'pid': proc.pid, 'gpu': gpu, 'cpus': assigned})
            time.sleep(3)
        write(RUN / 'state/coordinator.json', record(status='workers_launched', count=len(workers)))
        for proc in workers:
            proc.wait()
        os.sched_setaffinity(0, set(available_cpus()) | set(os.sched_getaffinity(0)))


def main(argv=None, *, run_dir=None):
    global RUN
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['coordinator', 'worker'])
    ap.add_argument('--run-dir', type=Path, default=run_dir, required=run_dir is None)
    ap.add_argument('--gpu', type=int)
    args = ap.parse_args(argv)
    RUN = Path(args.run_dir).resolve()
    manifest = json.loads((RUN / 'manifest.json').read_text())
    if args.mode == 'worker':
        if args.gpu is None:
            ap.error('worker requires --gpu')
        return worker(manifest, args.gpu)
    return coordinator(manifest)


if __name__ == '__main__':
    raise SystemExit(main())
