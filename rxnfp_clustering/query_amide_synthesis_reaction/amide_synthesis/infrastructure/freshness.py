"""Content-based freshness for mapping-dependent scientific artifacts.

Fresh means consistent with the declared input snapshots and implementation,
not that an external database has been queried continuously.
"""
from pathlib import Path
from functools import wraps
import hashlib,json,datetime,os,re,fcntl
from amide_synthesis.infrastructure.paths import RES, PROJECT, FIG

from amide_synthesis.stages import RULES, MODULES, BY_NAME

class StaleArtifactError(RuntimeError):pass
_HASH_CACHE={}

def fingerprint(path):
    p=Path(path)
    if not p.exists():return None
    st=p.stat();key=(str(p.resolve()),st.st_size,st.st_mtime_ns)
    if key in _HASH_CACHE:return _HASH_CACHE[key]
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    _HASH_CACHE[key]=h.hexdigest()
    return _HASH_CACHE[key]

def inputs(stage,res=RES,project=PROJECT):
    files=[Path(res)/f for f in RULES[stage][0]]
    files += [Path(project)/'mapping_policy.json',
              BY_NAME[stage].source,
              PROJECT/'amide_synthesis/enzyme_mapping/mapping_quality.py']
    files += [PROJECT/'amide_synthesis'/name for name in BY_NAME[stage].source_dependencies]
    policy=Path(project)/'mapping_policy.json'
    if policy.exists():
        d=json.loads(policy.read_text());files += [Path(project)/d[k] for k in ('default_links','evidence')]
        if d.get('reaction_bridges'):
            files.append(Path(project)/d['reaction_bridges'])
            files.append(Path(project)/'data/modelseed_master.parquet')
            registry=Path(project)/d['reaction_bridges']
            if registry.exists():
                for bridge in json.loads(registry.read_text()).get('bridges',[]):
                    files.append(Path(project)/bridge['audit_manifest'])
    override=os.environ.get('UNIPROT_MSD_LINKS')
    if override:files.append(Path(override))
    result={str(p.resolve()):fingerprint(p) for p in files}
    result['__runtime_options__']=runtime_options(stage)
    return result

def runtime_options(stage):
    source=BY_NAME[stage].source.read_text()
    names=set(re.findall(r'os\.environ\.get\([\'\"]([^\'\"]+)',source))|{'UNIPROT_MSD_LINKS'}
    return {name:os.environ.get(name) for name in sorted(names)}

def state(res=RES):
    p=Path(res)/'artifact_freshness.json'
    return json.loads(p.read_text()) if p.exists() else {'stages':{}}

def status(stage,res=RES,project=PROJECT):
    record=state(res)['stages'].get(stage)
    if record is None:return 'untracked'
    if record['inputs']!=inputs(stage,res,project):return 'stale_inputs'
    if any(fingerprint(p)!=h or h is None for p,h in record['outputs'].items()):return 'stale_outputs'
    return 'current'

def require_inputs(stage,res=RES,project=PROJECT,_visited=None):
    # Only an actual producer of an input is a dependency; unrelated lookup
    # sidecars do not force an entire network pipeline to run.
    visited=set() if _visited is None else _visited
    if stage in visited:return
    visited.add(stage);consumed=set(RULES[stage][0]);issues=[]
    for parent,(_,outputs) in RULES.items():
        if parent!=stage and consumed.intersection(outputs):
            result=status(parent,res,project)
            if result!='current':issues.append(parent+':'+result)
            else:require_inputs(parent,res,project,visited)
    if issues:raise StaleArtifactError('Refresh prerequisite stages before '+stage+': '+', '.join(issues))

def record(stage,res=RES,project=PROJECT):
    outputs={str((Path(res)/f).resolve()):fingerprint(Path(res)/f) for f in RULES[stage][1]}
    missing=[p for p,h in outputs.items() if h is None]
    if missing:raise StaleArtifactError('Stage '+stage+' did not produce '+', '.join(missing))
    entry={'inputs':inputs(stage,res,project),'outputs':outputs,
       'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'freshness_engine_sha256':fingerprint(__file__),
       'meaning':'consistent with recorded input snapshots and code'}
    with (Path(res)/'.artifact_freshness.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        doc=state(res);doc['stages'][stage]=entry
        p=Path(res)/'artifact_freshness.json';tmp=p.with_suffix(f'.{os.getpid()}.tmp')
        tmp.write_text(json.dumps(doc,indent=2));tmp.replace(p)

def tracked_stage(stage):
    def decorate(fn):
        @wraps(fn)
        def wrapped(*args,**kwargs):
            # Unit tests may redirect a stage's RES independently. Never let an
            # isolated fixture stamp the production artifact registry.
            actual=Path(fn.__globals__.get('RES',RES)).resolve()
            if actual!=Path(RES).resolve():return fn(*args,**kwargs)
            require_inputs(stage)
            before=inputs(stage)
            result=fn(*args,**kwargs)
            if result in (None,0):
                if inputs(stage)!=before:
                    raise StaleArtifactError('Inputs or implementation changed during '+stage+'; rerun before publishing')
                record(stage)
            return result
        return wrapped
    return decorate
