"""Read-only audit of this run's GPU assignments and inherited CPU/thread limits."""
import json
import os
import subprocess
from pathlib import Path


def verify(run_dir, *, write=False):
    RUN = Path(run_dir)
    workers=[json.loads(p.read_text()) for p in (RUN/'state').glob('worker_gpu*.json')]
    processes={}
    for p in Path('/proc').glob('[0-9]*'):
        try:
            stat=(p/'stat').read_text().split(') ',1)[1].split()
            processes[int(p.name)]={'parent':int(stat[1]),'argv':(p/'cmdline').read_bytes().decode().replace('\0',' ')}
        except (OSError,UnicodeError):pass
    records=[];errors=[]
    for w in workers:
        descendants={w['pid']}
        while True:
            more={pid for pid,v in processes.items() if v['parent'] in descendants}
            if more<=descendants:break
            descendants|=more
        for pid in descendants:
            if pid not in processes or 'python' not in processes[pid]['argv'].split(' ')[0]:continue
            try:
                affinity=sorted(os.sched_getaffinity(pid))
                env=dict(s.split('=',1) for s in Path(f'/proc/{pid}/environ').read_bytes().decode().split('\0') if '=' in s)
                r={'pid':pid,'gpu':w['gpu'],'affinity':affinity,'CUDA_VISIBLE_DEVICES':env.get('CUDA_VISIBLE_DEVICES'),
                   'threads':{k:env.get(k) for k in w['threads']},'argv':processes[pid]['argv']}
                if affinity!=w['cpu_affinity']:errors.append('CPU affinity mismatch '+str(pid))
                if r['CUDA_VISIBLE_DEVICES']!=str(w['gpu']):errors.append('GPU visibility mismatch '+str(pid))
                if any(v!='8' for v in r['threads'].values()):errors.append('Native pool cap mismatch '+str(pid))
                records.append(r)
            except OSError:pass
    gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    mapping={line.split(',')[1].strip():int(line.split(',')[0]) for line in gpus.splitlines()}
    resident=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'],text=True)
    resident_rows=[]
    for line in resident.splitlines():
        fields=[f.strip() for f in line.split(',')]
        if len(fields)>=4:resident_rows.append({'gpu':mapping[fields[0]],'pid':int(fields[1]),'process':fields[2],'memory':fields[3]})
    for w in workers:
        matches=[r for r in resident_rows if r['gpu']==w['gpu']]
        if len(matches)>1:errors.append('Multiple resident compute processes on GPU '+str(w['gpu']))
    sets=[set(w['cpu_affinity']) for w in workers]
    if any(a & b for i,a in enumerate(sets) for b in sets[i+1:]):errors.append('Overlapping worker CPUs')
    report={'workers':workers,'processes':records,'resident_gpu_processes':resident_rows,'online_cpus':os.cpu_count(),
            'load':os.getloadavg(),'errors':errors,'ok':not errors}
    if write:
        (RUN/'evidence/startup_verification.json').write_text(json.dumps(report, indent=2))
    return report

def main(argv=None, *, run_dir=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=run_dir, required=run_dir is None)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args(argv)
    report = verify(args.run_dir, write=args.write)
    print(json.dumps(report, indent=2))
    return int(not report['ok'])
