"""Run an explicitly configured, reservation-gated Boltz campaign.

The coordinator waits for exact-sequence batched MSAs and CPU/GPU capacity.
Workers have exclusive GPU reservations, disjoint CPUs, bounded retries and
assembly checks. This module launches work only through an explicit command.
"""
import argparse
import csv
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from amide_synthesis.structure_modeling.poses import inspect, aggregate as aggregate_run
from amide_synthesis.infrastructure.paths import PROJECT

RUN = None  # Configured explicitly by main; importing never launches work.

def aggregate():
    return aggregate_run(RUN, write=True)
GATE = '/nfs/lambda_stor_01/homes/rzhu/bin/gpu_reserve'
KIT = '/nfs/lambda_stor_01/homes/rzhu/Tool/uplifting-biomolecular-modeling/boltz2/_kit_run.sh'
PYTHON = '/nfs/lambda_stor_01/homes/rzhu/boltz2-venv/bin/python'
NATIVE = ['OMP_NUM_THREADS','OMP_THREAD_LIMIT','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']

def write(path, doc):
    temp = path.with_suffix(f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(doc, indent=2)); temp.replace(path)

def starttime(pid):
    return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[19]

def record(**kwargs):
    return {'host':socket.gethostname(), 'pid':os.getpid(), 'process_start':starttime(os.getpid()),
            'utc_epoch':time.time(), 'cpu_affinity':sorted(os.sched_getaffinity(0)), **kwargs}

def msa_ready(manifest):
    for item in manifest['inputs']:
        p = Path(item['msa'])
        if not p.exists(): return False
        try:
            with p.open() as f:
                rows = csv.DictReader(f); first = next(rows)
                sequence = first['sequence']; aligned = ''.join(c for c in sequence if not c.islower()).replace('-','')
                if hashlib.sha256(aligned.encode()).hexdigest() != item['sequence_sha256']:
                    raise ValueError('MSA query sequence mismatch: ' + item['accession'])
                if next(rows, None) is None:
                    raise ValueError('MSA contains only its query: ' + item['accession'])
        except (KeyError,StopIteration): return False
    return True

def available_cpus():
    allowed = set(os.sched_getaffinity(0)); occupied = set()
    for p in Path('/proc').glob('[0-9]*'):
        if int(p.name) == os.getpid(): continue
        try:
            command = (p/'cmdline').read_bytes().split(b'\0')[0]
            if b'python' not in command: continue
            affinity = set(os.sched_getaffinity(int(p.name)))
            if len(affinity) < len(allowed): occupied |= affinity
        except OSError: pass
    return sorted(allowed - occupied)

def coordinator(manifest):
    if socket.gethostname() != 'lambda13': raise RuntimeError('This exact kit is preflighted on lambda13')
    write(RUN/'state/coordinator.json', record(status='waiting_for_batched_msa'))
    deadline = time.time() + 10800
    while not msa_ready(manifest):
        if time.time() > deadline: raise RuntimeError('MSA readiness timeout; no GPUs reserved')
        time.sleep(10)
    # CPU-only imports and checkpoint checks happen before GPU reservation.
    import torch, rdkit, boltz.main, boltz2_opt, gemmi, yaml
    import tempfile
    with tempfile.TemporaryDirectory(prefix='amide_preflight_') as t:
        Path(t,'writecheck').write_text('ok')
    preflight = record(status='dependencies_ready', torch=torch.__version__, rdkit=rdkit.__version__,
                       boltz2_opt=boltz2_opt.__version__, online_cpus=os.cpu_count(), load=os.getloadavg())
    for p in ['boltz2_conf.ckpt','ccd.pkl','mols']:
        if not (Path('/nfs/lambda_stor_01/homes/rzhu/.boltz')/p).exists(): raise RuntimeError('Missing checkpoint/cache '+p)
    write(RUN/'evidence/launch_preflight.json', preflight)
    while True:
        result = subprocess.run([GATE,'list','--hosts','lambda13'], text=True, capture_output=True, timeout=50)
        gpus = [int(l.split()[1]) for l in result.stdout.splitlines() if l.startswith('lambda13 ')]
        cpus = available_cpus()
        remaining=sum(not (RUN/'state'/(item['name']+'.done.json')).exists() for item in manifest['inputs'])
        if not remaining:
            return 0
        count = min(4, len(gpus), remaining, max(0,(len(cpus)-2)//8))
        if max(os.getloadavg()) + count*8 + 2 > os.cpu_count(): count = 0
        if count: break
        if time.time()>deadline: raise RuntimeError('No reservable GPU/CPU capacity before deadline')
        write(RUN/'state/coordinator.json',record(status='waiting_for_free_gpu_or_cpu'));time.sleep(15)
    workers = []
    os.sched_setaffinity(0, set(cpus[-2:]))
    cpus = cpus[:-2]
    for i, gpu in enumerate(gpus[:count]):
        assigned = cpus[i*8:(i+1)*8]
        env = dict(os.environ, **{k:'8' for k in NATIVE}, PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false')
        env['PYTHONPATH'] = str(PROJECT) + os.pathsep + env.get('PYTHONPATH', '')
        cpu_string = ','.join(map(str, assigned))
        argv = ['taskset','-c',cpu_string,GATE,'run',str(gpu),'--label','amide_'+RUN.name,'--',
                PYTHON,'-m','amide_synthesis.structure_modeling.fleet','worker','--run-dir',str(RUN),'--gpu',str(gpu)]
        log = (RUN/'logs'/f'worker_gpu{gpu}.log').open('a')
        p = subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        workers.append({'pid':p.pid,'gpu':gpu,'cpus':assigned,'argv':argv,'process':p})
        write(RUN/'state'/f'launch_gpu{gpu}.json',{k:v for k,v in workers[-1].items() if k!='process'})
        time.sleep(3)
    write(RUN/'state/coordinator.json',record(status='workers_launched', workers=[{k:v for k,v in w.items() if k!='process'} for w in workers]))
    codes = [w['process'].wait() for w in workers]
    rows = aggregate()
    write(RUN/'state/coordinator.json',record(status='complete' if len(rows)==len(manifest['inputs']) else 'incomplete',completed=len(rows),worker_exit_codes=codes))
    return 0 if len(rows)==len(manifest['inputs']) else 1

def worker(manifest, gpu):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(gpu): raise RuntimeError('GPU visibility does not match the gate assignment')
    if len(os.sched_getaffinity(0)) != 8: raise RuntimeError('Worker must have eight assigned CPUs')
    if max(os.getloadavg()) + 8 > os.cpu_count(): raise RuntimeError('CPU capacity no longer adequate')
    info = record(gpu=gpu, visible_gpu=os.environ['CUDA_VISIBLE_DEVICES'],threads={k:os.environ[k] for k in NATIVE})
    write(RUN/'state'/f'worker_gpu{gpu}.json',info)
    consecutive = 0
    while True:
        pending = False; claimed_any = False
        for item in manifest['inputs']:
            name = item['name']; done = RUN/'state'/f'{name}.done.json'
            if done.exists():
                inspect(item,RUN/'outputs'/name)
                continue
            fail = RUN/'state'/f'{name}.failure.json'
            failure = json.loads(fail.read_text()) if fail.exists() else {'attempts':0,'retry_after':0}
            if failure['attempts'] >= 2: continue
            pending = True
            if failure['retry_after'] > time.time(): continue
            claim = RUN/'state'/f'{name}.claim'
            try: claim.mkdir()
            except FileExistsError: continue
            claimed_any = True; child = None
            try:
                write(claim/'owner.json',info)
                if done.exists(): continue
                out = RUN/'outputs'/name
                try:
                    validated = inspect(item,out)
                except (ValueError,KeyError,RuntimeError):
                    validated = None
                if validated is None:
                    if out.exists():
                        archive = RUN/'outputs'/f'{name}.partial.{int(time.time())}'
                        out.rename(archive)
                    attempt = failure['attempts']+1
                    argv = ['bash',KIT,'pred','--config','h100','--mode','exact','--input',item['input'],'--out_dir',str(out),
                            '--seed','42','--diffusion_samples','50','--max_parallel_samples','41','--recycling_steps','3',
                            '--sampling_steps','200','--num_workers','0','--preprocessing-threads','1','--output_format','pdb']
                    write(RUN/'state'/f'{name}.running.json',record(gpu=gpu,attempt=attempt,argv=argv))
                    print('PREDICTION_START',name,'gpu',gpu,'attempt',attempt,flush=True)
                    with (RUN/'logs'/f'{name}.attempt{attempt}.log').open('a') as log:
                        child = subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                        def stop(signum, frame):
                            if child.poll() is None:
                                os.killpg(child.pid,signal.SIGTERM)
                                try: child.wait(timeout=30)
                                except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL);child.wait()
                            raise SystemExit(128+signum)
                        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
                        code=child.wait()
                    if code: raise RuntimeError('Predictor exit '+str(code))
                    validated=inspect(item,out)
                write(done,validated);consecutive=0
                print('COMPLETE',name,validated['A_deg'],flush=True)
            except Exception as exc:
                consecutive += 1
                write(fail,{'attempts':failure['attempts']+1,'error':str(exc),'retry_after':time.time()+30,**record(gpu=gpu)})
                print('FAILED',name,type(exc).__name__,str(exc),flush=True)
            finally:
                (claim/'owner.json').unlink(missing_ok=True);claim.rmdir()
            if consecutive>=3: print('CIRCUIT_BREAKER',flush=True);return 2
        if not pending: return 0
        if not claimed_any:
            live = False
            for c in (RUN/'state').glob('*.claim/owner.json'):
                owner=json.loads(c.read_text())
                try: live |= owner['host']==socket.gethostname() and starttime(owner['pid'])==owner['process_start']
                except OSError: pass
            if not live:
                # Remaining work is in retry cooldown or has unresolved ownership;
                # do not steal claims based on age.
                unresolved=list((RUN/'state').glob('*.claim'))
                if unresolved: print('UNRESOLVED_CLAIMS',unresolved,flush=True);return 3
            time.sleep(10)

def main(argv=None, *, run_dir=None):
    global RUN
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['coordinator', 'worker'])
    ap.add_argument('--run-dir', type=Path, default=run_dir, required=run_dir is None)
    ap.add_argument('--gpu', type=int)
    args = ap.parse_args(argv)
    RUN = Path(args.run_dir).resolve()
    if args.mode == 'worker' and args.gpu is None:
        ap.error('worker requires --gpu')
    manifest = json.loads((RUN/'manifest.json').read_text())
    return coordinator(manifest) if args.mode == 'coordinator' else worker(manifest, args.gpu)

if __name__ == '__main__':
    raise SystemExit(main())
