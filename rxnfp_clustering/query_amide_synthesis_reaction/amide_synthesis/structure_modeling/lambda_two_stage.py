"""Reusable reservation-gated two-stage Boltz runner for a declared enzyme cohort."""
import argparse
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback

PROJECT=Path(__file__).resolve().parents[2]
GATE='/nfs/lambda_stor_01/homes/rzhu/bin/gpu_reserve'
PYTHON='/homes/rzhu/miniforge3/envs/boltz-2/bin/python'
LIBRARY='/homes/rzhu/miniforge3/envs/boltz-2/lib:/nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/mpi-shim/lib'
CAPS=['OMP_NUM_THREADS','OMP_THREAD_LIMIT','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']
CHILD=None


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def identity(**extra):
    return dict(host=socket.gethostname(),pid=os.getpid(),process_start_ticks=
        Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19],
        cpu_affinity=sorted(os.sched_getaffinity(0)),created_epoch=time.time(),**extra)


def prepare(campaign):
    import yaml
    root=campaign/'lambda_fleet';root.mkdir(exist_ok=True)
    if (root/'manifest.json').exists():raise FileExistsError('Preserve the existing Lambda campaign')
    source=json.loads((campaign/'aurora/manifest.json').read_text())
    for sub in ['configs','runner','logs','claims','results','failures','state','evidence']:(root/sub).mkdir(exist_ok=True)
    support=PROJECT/'amide_synthesis/structure_modeling/aurora_c5_support'
    for name in ['build_step2.py','geometry_qc.py']:shutil.copyfile(support/name,root/'runner'/name)
    shutil.copyfile(Path(__file__).with_name('aurora_ligand_qc.py'),root/'runner/ligand_qc.py')
    shutil.copyfile(Path(__file__).with_name('boltz_predict_cuda_fp32.py'),root/'runner/predict.py')
    atomic(root/'runtime_settings.json',dict(max_parallel_samples=2,max_residues_for_parallel=600,
        precision='FP32',diffusion_samples=50,recycling_steps=3,sampling_steps=200))
    records=[]
    for item in source['inputs']:
        doc=yaml.safe_load((campaign/'aurora/configs'/item['config_file']).read_text())
        msa=campaign/'msa_cache'/item['msa_file']
        if digest(msa)!=item['msa_sha256']:raise ValueError('Fresh MSA changed')
        protein=next(e['protein'] for e in doc['sequences'] if 'protein' in e)
        protein['msa']=str(msa)
        path=root/'configs'/item['config_file'];path.write_text(yaml.safe_dump(doc,sort_keys=False))
        records.append(dict(item,msa=str(msa),input=str(path),config_sha256=digest(path)))
    # Put the two largest mechanism classes first; these are real campaign work,
    # rather than disposable checks or a hard-coded accession list.
    first=[]
    for mg in sorted({r['mg_count'] for r in records}):
        first.append(max([r for r in records if r['mg_count']==mg],key=lambda r:r['sequence_length'])['accession'])
    records.sort(key=lambda r:(r['accession'] not in first,first.index(r['accession']) if r['accession'] in first else 0,r['accession']))
    atomic(root/'manifest.json',dict(campaign_id=source['campaign_id'],provider='Lambda CUDA',
        status='prepared',sequence_count=len(records),expected_structures=len(records)*100,
        substrates='glutarate and cadaverine',implementation='stock_boltz2_cuda_fp32_serial',
        source_aurora_manifest_sha256=digest(campaign/'aurora/manifest.json'),
        excluded_hosts=['lambda13'],samples_per_stage=50,seed=42,recycling_steps=3,sampling_steps=200,
        inputs=records,producer_sha256={str(p.relative_to(root)):digest(p) for p in (root/'runner').glob('*.py')}))
    print(json.dumps(dict(prepared=len(records),root=str(root)),indent=2))


def stop(signum,frame):
    global CHILD
    if CHILD is not None and CHILD.poll() is None:
        os.killpg(CHILD.pid,signal.SIGTERM)
        try:CHILD.wait(timeout=30)
        except subprocess.TimeoutExpired:os.killpg(CHILD.pid,signal.SIGKILL);CHILD.wait()
    raise SystemExit(128+signum)


def verified(path,record):
    if not path.exists():return False
    state=json.loads(path.read_text())
    if state.get('source_input_config_sha256')!=record['config_sha256'] or state.get('source_msa_sha256')!=record['msa_sha256']:
        raise ValueError('Output belongs to different substrate/configuration or MSA')
    if state.get('status') not in ('verified','verified_with_warnings') or len(state.get('poses',[]))!=50:return False
    for pose in state['poses']:
        pdb=Path(pose['path']);confidence=pdb.with_name('confidence_'+pdb.stem+'.json')
        if digest(pdb)!=pose['pdb_sha256'] or digest(confidence)!=pose['confidence_sha256']:raise ValueError('Verified output changed')
    return True


def run_stage(root,record,stage,config,folder,gpu):
    global CHILD
    import yaml
    from geometry_qc import inspect_outputs
    name=record['name']+('_step2' if stage==2 else '')
    path=folder/(name+'.yaml');path.write_text(yaml.safe_dump(config,sort_keys=False))
    output=folder/'output'
    argv=[PYTHON,str(root/'runner/predict.py'),'predict',str(path),'--out_dir',str(output),
        '--cache','/nfs/lambda_stor_01/homes/rzhu/.boltz','--accelerator','gpu','--devices','1',
        '--diffusion_samples','50','--max_parallel_samples','1','--seed','42',
        '--recycling_steps','3','--sampling_steps','200','--output_format','pdb',
        '--num_workers','0','--preprocessing-threads','1','--no_kernels']
    atomic(folder/'execution.json',identity(gpu=gpu,stage=stage,argv=argv,status='starting'))
    print(json.dumps(dict(event='PREDICTION_START',accession=record['accession'],stage=stage,gpu=gpu)),flush=True)
    with (folder/'predict.log').open('x') as log:
        CHILD=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        code=CHILD.wait();CHILD=None
    if code:raise RuntimeError('Predictor exited '+str(code))
    item=dict(name=name,accession=record['accession'],stage=stage,protein_sequence_sha256=record['sequence_sha256'])
    poses=inspect_outputs(item,config,output)
    pred=output/('boltz_results_'+name)/'predictions'/name
    for p in poses:p['path']=str(pred/p['model'])
    return dict(status='verified_with_warnings' if any(p.get('pocket_satisfied') is False for p in poses) else 'verified',
        accession=record['accession'],stage=stage,protein_sequence_sha256=record['sequence_sha256'],
        source_input_config_sha256=record['config_sha256'],source_msa_sha256=record['msa_sha256'],
        config_sha256=digest(path),poses=poses,argv=argv,producer='stock_cuda_fp32_serial')


def process(root,record,gpu):
    import yaml
    from build_step2 import build
    work=root/'results'/record['accession'];work.mkdir(exist_ok=True)
    for stage in (1,2):
        state=work/('stage'+str(stage)+'.json')
        if verified(state,record):continue
        if digest(record['input'])!=record['config_sha256'] or digest(record['msa'])!=record['msa_sha256']:
            raise ValueError('Input or MSA changed')
        config=yaml.safe_load(Path(record['input']).read_text())
        meta=None
        if stage==2:
            first=json.loads((work/'stage1.json').read_text())
            config,meta=build(config,[p['path'] for p in first['poses']])
        folder=work/'attempts'/('stage'+str(stage)+'_'+str(time.time_ns()));folder.mkdir(parents=True)
        result=run_stage(root,record,stage,config,folder,gpu)
        if meta:result['step1_pocket']=meta
        atomic(state,result)
        print(json.dumps(dict(event='STAGE_VERIFIED',accession=record['accession'],stage=stage,models=50)),flush=True)


def worker(root,gpu):
    if socket.gethostname()=='lambda13':raise RuntimeError('Lambda13 is excluded by the user')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(gpu):raise RuntimeError('GPU gate assignment mismatch')
    if len(os.sched_getaffinity(0))!=8 or any(os.environ.get(k)!='8' for k in CAPS):raise RuntimeError('CPU/thread assignment mismatch')
    if max(os.getloadavg())+8>os.cpu_count():raise RuntimeError('CPU capacity no longer available')
    sys.path.insert(0,str(root/'runner'))
    import gemmi,yaml,rdkit
    manifest=json.loads((root/'manifest.json').read_text())
    for p,expected in manifest['producer_sha256'].items():
        if digest(root/p)!=expected:raise ValueError('Producer code changed')
    info=identity(gpu=gpu,visible_gpu=os.environ['CUDA_VISIBLE_DEVICES'],threads={k:os.environ[k] for k in CAPS})
    atomic(root/'state'/('worker_'+socket.gethostname()+'_'+str(gpu)+'.json'),info)
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,stop)
    deadline=time.time()+12*3600;consecutive=0
    while time.time()<deadline and consecutive<3:
        pending=False;claimed=False
        for record in manifest['inputs']:
            if time.time()>deadline or consecutive>=3:break
            accession=record['accession'];work=root/'results'/accession
            if (work/'stage1.json').exists() and (work/'stage2.json').exists():continue
            failure=root/'failures'/(accession+'.json')
            f=json.loads(failure.read_text()) if failure.exists() else {'attempts':0,'retry_after':0}
            if f['attempts']>=2:continue
            pending=True
            if f['retry_after']>time.time():continue
            claim=root/'claims'/accession
            try:claim.mkdir()
            except FileExistsError:continue
            claimed=True;atomic(claim/'owner.json',info)
            try:
                process(root,record,gpu);consecutive=0
                if failure.exists():
                    (root/'state/resolved_failures').mkdir(exist_ok=True)
                    failure.rename(root/'state/resolved_failures'/(accession+'_'+str(time.time_ns())+'.json'))
            except Exception as error:
                consecutive+=1
                atomic(failure,dict(attempts=f['attempts']+1,retry_after=time.time()+60,error=repr(error),
                    traceback=traceback.format_exc(),owner=info))
                print(json.dumps(dict(event='PREDICTION_FAILED',accession=accession,error=repr(error))),flush=True)
            finally:
                if CHILD is not None and CHILD.poll() is None:raise RuntimeError('Predictor still alive; retain claim')
                (claim/'owner.json').unlink(missing_ok=True);claim.rmdir()
        if not pending:return 0
        if not claimed:
            # Live peers keep their claims. Unresolved ownership is never stolen.
            claims=list((root/'claims').glob('*/owner.json'))
            if not claims and all(json.loads(p.read_text()).get('attempts',0)>=2 for p in (root/'failures').glob('*.json')):
                if all((root/'results'/r['accession']/'stage2.json').exists() or (root/'failures'/(r['accession']+'.json')).exists() for r in manifest['inputs']):return 2
            time.sleep(10)
    return 2 if consecutive>=3 else 0


def remote(operation,payload):
    import json,os,subprocess,sys,time
    from pathlib import Path
    if operation=='resources':
        allowed=set(os.sched_getaffinity(0));occupied=set()
        for p in Path('/proc').glob('[0-9]*'):
            try:
                if int(p.name)==os.getpid() or not (p/'cmdline').read_bytes():continue
                affinity=set(os.sched_getaffinity(int(p.name)))
                if len(affinity)<os.cpu_count()/2:occupied.update(affinity)
            except OSError:pass
        return dict(host=os.uname().nodename,cpus=sorted(allowed-occupied),load=max(os.getloadavg()),online=os.cpu_count())
    if operation=='alive':
        result=[]
        for item in payload:
            try:
                stat=Path('/proc',str(item['pid']),'stat').read_text().split(') ',1)[1].split()
                live=stat[0]!='Z' and stat[19]==item['process_start_ticks']
            except OSError:live=False
            result.append(dict(item,alive=live))
        return result
    if operation=='launch':
        plan=payload;root=Path(plan['root'])
        # Recheck CPU bindings immediately before this node's batch launches.
        available=remote('resources',{})
        assigned={c for w in plan['workers'] for c in w['cpus']}
        if not assigned<=set(available['cpus']) or available['load']+len(assigned)>available['online']:
            raise RuntimeError('CPU allocation lost capacity before launch')
        results=[]
        for worker in plan['workers']:
            argv=worker['argv'];log=root/'logs'/('worker_'+os.uname().nodename+'_'+str(worker['gpu'])+'_'+str(time.time_ns())+'.log')
            with log.open('a') as f:
                p=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            info=dict(worker,pid=p.pid,host=os.uname().nodename,log=str(log),
                process_start_ticks=Path('/proc',str(p.pid),'stat').read_text().split(') ',1)[1].split()[19],created_epoch=time.time())
            results.append(info);time.sleep(3)
        return results
    raise ValueError('Unknown remote operation')


def rpc(host,operation,payload):
    text=Path(__file__).read_text()
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='remote')
    source='\n'.join(text.splitlines()[node.lineno-1:node.end_lineno])
    code=source+'\nimport json\nprint(json.dumps(remote('+repr(operation)+','+repr(payload)+')))\n'
    control_env=dict(os.environ)
    for key in ['LD_LIBRARY_PATH','LD_PRELOAD']:control_env.pop(key,None)
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',host,'python3','-'],
        input=code,capture_output=True,text=True,timeout=50,env=control_env)
    if p.returncode:raise RuntimeError('Remote '+host+': '+p.stderr[-1500:])
    return json.loads(p.stdout)


def coordinator(root):
    import fcntl
    with (root/'state/coordinator.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return coordinate(root)


def coordinate(root):
    manifest=json.loads((root/'manifest.json').read_text());hosts=['lambda1','lambda2']
    path=root/'state/coordinator.json'
    state=json.loads(path.read_text()) if path.exists() else dict(workers=[],phase='launching')
    while True:
        live=[]
        try:
            for host in hosts:
                items=[w for w in state['workers'] if w['host']==host and w.get('alive',True)]
                if items:live+=rpc(host,'alive',items)
        except (subprocess.TimeoutExpired,RuntimeError,json.JSONDecodeError) as error:
            state.update(phase='waiting_for_worker_observation',observation_error=str(error),updated_epoch=time.time())
            atomic(path,state);time.sleep(30);continue
        prior={(w['host'],w['pid']):w for w in live}
        state['workers']=[prior.get((w['host'],w['pid']),w) for w in state['workers']]
        active=[w for w in state['workers'] if w.get('alive',True)]
        one=len(list((root/'results').glob('*/stage1.json')));two=len(list((root/'results').glob('*/stage2.json')))
        terminal=[p for p in (root/'failures').glob('*.json') if json.loads(p.read_text())['attempts']>=2]
        state.update(phase='running',stage1_complete=one,stage2_complete=two,total=manifest['sequence_count'],
            active_workers=len(active),terminal_failures=len(terminal),updated_epoch=time.time())
        atomic(path,state)
        if two==manifest['sequence_count'] and not active:
            from amide_synthesis.structure_modeling.aurora_target_analyse import analyse
            destination=root/'analysis/final'
            if not (destination/'complete.json').exists():analyse(root,destination)
            state.update(phase='analysis_complete');atomic(path,state);return 0
        exited_errors=[]
        for w in state['workers']:
            if w.get('alive',True):continue
            p=root/'state'/('exit_'+w['host']+'_'+str(w['gpu'])+'.json')
            if p.exists() and json.loads(p.read_text()).get('returncode'):
                exited_errors.append(str(p))
        if exited_errors:
            state.update(phase='worker_error_requires_review',errors=exited_errors);atomic(path,state);return 2
        if two+len(terminal)>=manifest['sequence_count'] and not active:
            state.update(phase='terminal_candidate_failures');atomic(path,state);return 2
        control_env=dict(os.environ)
        for key in ['LD_LIBRARY_PATH','LD_PRELOAD']:control_env.pop(key,None)
        try:
            gate=subprocess.run([GATE,'list','--hosts',','.join(hosts)],capture_output=True,text=True,timeout=55,env=control_env)
        except subprocess.TimeoutExpired:
            state.update(phase='waiting_for_gpu_discovery',observation_error='GPU gate discovery timed out; existing work preserved',updated_epoch=time.time())
            atomic(path,state);time.sleep(30);continue
        if gate.returncode:
            state.update(phase='waiting_for_gpu_discovery',observation_error=gate.stderr[-1000:],updated_epoch=time.time())
            atomic(path,state);time.sleep(30);continue
        free={h:[] for h in hosts}
        for line in gate.stdout.splitlines():
            h,g=line.split();free[h].append(int(g))
        for host in hosts:
            if not free[host]:continue
            try:resources=rpc(host,'resources',{})
            except (subprocess.TimeoutExpired,RuntimeError,json.JSONDecodeError) as error:
                state.update(observation_error=str(error),updated_epoch=time.time());atomic(path,state);continue
            available=resources['cpus']
            host_active=sum(w['host']==host for w in active)
            count=min(max(0,3-host_active),len(free[host]),len(available)//8)
            while count and resources['load']+count*8>resources['online']:count-=1
            plans=[]
            for i,gpu in enumerate(free[host][:count]):
                cpus=available[i*8:(i+1)*8]
                env=['env','LD_LIBRARY_PATH='+LIBRARY,'PYTHONPATH='+str(PROJECT),'PYTHONUNBUFFERED=1',
                    'TOKENIZERS_PARALLELISM=false',*[key+'=8' for key in CAPS]]
                argv=['taskset','-c',','.join(map(str,cpus)),GATE,'run',str(gpu),'--label',
                    'C5_'+manifest['campaign_id'],'--',*env,PYTHON,'-m',
                    'amide_synthesis.structure_modeling.lambda_two_stage','worker','--run-dir',str(root),'--gpu',str(gpu)]
                plans.append(dict(gpu=gpu,cpus=cpus,argv=argv))
            if plans:
                state['workers']+=rpc(host,'launch',dict(root=str(root),workers=plans));atomic(path,state)
        time.sleep(30)


def start(campaign):
    """Start/resume the saved launcher and rolling analysis, without duplicating them."""
    root=campaign/'lambda_fleet'
    if not (root/'manifest.json').exists():prepare(campaign)
    control_host='lambda1'
    code='''from pathlib import Path
import os,sys,json,subprocess,time
sys.path.insert(0,PROJECT_VALUE)
from amide_synthesis.structure_modeling.lambda_two_stage import remote,PYTHON,LIBRARY,CAPS
root=Path(ROOT_VALUE)
def alive(owner):
 try:
  state=Path('/proc',str(owner['pid']),'stat').read_text().split(') ',1)[1].split()
  return state[0]!='Z' and state[19]==owner['process_start_ticks']
 except OSError:return False
started=[]
for role,owner_name,mode in [('coordinator','coordinator_owner.json','coordinator'),('rolling_analysis','rolling_analysis_owner.json','rolling')]:
 owner_path=root/'state'/owner_name
 if owner_path.exists():
  prior=json.loads(owner_path.read_text())
  if prior['host']!=os.uname().nodename:raise RuntimeError('Owner host must be audited before resuming')
  if alive(prior):started.append({'role':role,'state':'already_running','pid':prior['pid']});continue
  backup=root/'state'/'prior_owners';backup.mkdir(exist_ok=True)
  owner_path.rename(backup/(owner_name+'.'+str(time.time_ns())))
 available=remote('resources',{})
 cpus=available['cpus'][-(2 if mode=='coordinator' else 1):]
 if not cpus or available['load']+len(cpus)>available['online']:raise RuntimeError('No control CPU capacity')
 env=dict(os.environ,PYTHONPATH=PROJECT_VALUE,PYTHONUNBUFFERED='1')
 for key in ['LD_LIBRARY_PATH','LD_PRELOAD']:env.pop(key,None)
 for key in CAPS:env[key]='1'
 if mode=='rolling':
  env['LD_LIBRARY_PATH']=LIBRARY
  args=['-m','amide_synthesis.evaluation.rolling_cofold','--run-dir',str(root),'--watch','--interval','60']
 else:args=['-m','amide_synthesis.structure_modeling.lambda_two_stage','coordinator','--run-dir',str(root)]
 argv=['taskset','-c',','.join(map(str,cpus)),PYTHON,*args]
 with (root/'logs'/(role+'.log')).open('a') as log:p=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=log,env=env,start_new_session=True)
 record={'host':os.uname().nodename,'pid':p.pid,'process_start_ticks':Path('/proc',str(p.pid),'stat').read_text().split(') ',1)[1].split()[19],'cpus':cpus,'argv':argv,'created_epoch':time.time()}
 owner_path.write_text(json.dumps(record,indent=2)+'\\n');started.append({'role':role,'state':'started','pid':p.pid})
print(json.dumps(started))
'''.replace('PROJECT_VALUE',repr(str(PROJECT))).replace('ROOT_VALUE',repr(str(root)))
    env=dict(os.environ)
    for key in ['LD_LIBRARY_PATH','LD_PRELOAD']:env.pop(key,None)
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',control_host,'python3','-'],
        input=code,text=True,capture_output=True,env=env,timeout=30)
    if result.returncode:raise RuntimeError(result.stderr)
    print(result.stdout)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','start','worker','coordinator']);p.add_argument('--campaign',type=Path);p.add_argument('--run-dir',type=Path);p.add_argument('--gpu',type=int)
    a=p.parse_args()
    if a.mode=='prepare':prepare(a.campaign.resolve());return 0
    if a.mode=='start':start(a.campaign.resolve());return 0
    root=a.run_dir.resolve()
    if a.mode=='coordinator':return coordinator(root)
    try:code=worker(root,a.gpu)
    except Exception:
        atomic(root/'state'/('exit_'+socket.gethostname()+'_'+str(a.gpu)+'.json'),identity(returncode=2,error=traceback.format_exc()))
        raise
    atomic(root/'state'/('exit_'+socket.gethostname()+'_'+str(a.gpu)+'.json'),identity(returncode=code))
    return code


if __name__=='__main__':raise SystemExit(main())
