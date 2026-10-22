"""Content-based freshness for mapping-dependent scientific artifacts.

Fresh means consistent with the declared input snapshots and implementation,
not that an external database has been queried continuously.
"""
from pathlib import Path
from functools import wraps
import hashlib,json,datetime,os,re,fcntl
from .paths import RES,PROJECT,FIG

RULES={
 'families':(['candidates_ranked.tsv','enzymes_uniprot.tsv'],['family_expansion.tsv','family_expansion_report.tsv','family_expansion_summary.json']),
 'links':(['amide_forming_neighbours.tsv'],['precomputed_link_candidates.tsv','precomputed_link_summary.json']),
 'table':(['enzymes_uniprot.tsv','enzyme_annotations.tsv','amide_forming_neighbours.tsv','enzymes.fasta'],['candidates_ranked.tsv','candidates_top.fasta']),
 'neighbours':(['candidates_ranked.tsv','precomputed_link_candidates.tsv','amide_forming_neighbours.tsv'],['chemistry_neighbour_candidates.tsv','chemistry_neighbour_summary.json']),
 'master':(['candidates_ranked.tsv','family_expansion.tsv','enzymes_uniprot.tsv','enzymes.fasta','precomputed_link_candidates.tsv','reference_actives.tsv','chemistry_neighbour_candidates.tsv','enzyme_name_candidates.tsv','active_homolog_candidates.tsv','kegg_ortholog_candidates.tsv','ec_substrate_ceiling.tsv'],['master_candidates.tsv','master_candidates_summary.json']),
 'enrich':(['master_candidates.tsv'],['master_candidates_enriched.tsv','enrichment_summary.json']),
 'pdb':(['master_candidates_enriched.tsv'],['pdb_entry_facts.tsv','pdb_evidence.tsv','pdb_evidence_summary.json']),
 'geometry':(['pdb_evidence.tsv'],['adenylation_geometry.tsv','adenylation_geometry_summary.json']),
 'ligands':(['pdb_entry_facts.tsv','pdb_evidence.tsv','master_candidates_enriched.tsv'],['ligand_evidence.tsv','ligand_evidence_summary.json','pocket_capacity.tsv']),
 'members':(['candidates_ranked.tsv','enzymes_uniprot.tsv','enzymes.fasta','master_candidates_enriched.tsv'],['member_similarity.tsv','member_similarity_summary.json']),
 'famval':(['candidates_ranked.tsv','enzymes_uniprot.tsv','amide_forming_neighbours.tsv','master_candidates_enriched.tsv'],['family_evaluation.tsv']),
 'properties':(['master_candidates_enriched.tsv'],['curated_properties.tsv','curated_properties_summary.json']),
 'shortlist':(['master_candidates_enriched.tsv','pdb_evidence.tsv','member_similarity.tsv','family_evaluation.tsv','curated_properties.tsv'],['shortlist.tsv','shortlist.fasta','shortlist_summary.json']),
 'models':(['shortlist.tsv','curated_properties.tsv','ec_substrate_ceiling.tsv','ligand_evidence.tsv'],['alphafold_quality.tsv','shortlist_annotated.tsv','alphafold_quality_summary.json']),
 'converged':(['master_candidates_enriched.tsv','family_evaluation.tsv','pdb_evidence.tsv','ligand_evidence.tsv','curated_properties.tsv','alphafold_quality.tsv'],['converged_candidates.tsv','converged_summary.json','evidence_lines.tsv']),
 'dossier':(['converged_candidates.tsv','evidence_lines.tsv','master_candidates_enriched.tsv','shortlist_annotated.tsv','pocket_capacity.tsv','alphafold_quality.tsv','curated_properties.tsv','ligand_evidence.tsv','ec_substrate_ceiling.tsv','pdb_evidence.tsv','adenylation_geometry.tsv','kinetics_by_ec.tsv','anl_similarity_to_actives.tsv'],['CANDIDATES.md']),
 'native':(['candidates_ranked.tsv','enzyme_annotations.tsv'],['native_substrates.tsv','enzyme_families.tsv']),
 'deckfigs':(['enzyme_families.tsv','native_substrates.tsv'],['../figures/fig_families.png','../figures/fig_substrate_gap.png']),
 'figure':(['candidates_ranked.tsv','queries.tsv','amide_forming_neighbours.tsv','corpus_manifest.json'],['../figures/fig_tsne_nylon_query.png','query_tsne_coords.tsv']),
}
MODULES={'families':'expand_families','links':'precomputed_links','table':'candidate_table','neighbours':'chemistry_neighbours',
         'master':'master_table','enrich':'enrich','pdb':'pdb_evidence','geometry':'adenylation_geometry',
         'ligands':'ligand_evidence','members':'member_similarity','famval':'family_evaluation',
         'properties':'curated_properties','shortlist':'shortlist','models':'model_quality',
         'converged':'converged','dossier':'dossier','native':'native_substrates',
         'deckfigs':'deck_figures','figure':'project_tsne'}
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
              Path(__file__).with_name(MODULES[stage]+'.py'),
              Path(__file__).with_name('mapping_quality.py')]
    policy=Path(project)/'mapping_policy.json'
    if policy.exists():
        d=json.loads(policy.read_text());files += [Path(project)/d[k] for k in ('default_links','evidence')]
    override=os.environ.get('UNIPROT_MSD_LINKS')
    if override:files.append(Path(override))
    result={str(p.resolve()):fingerprint(p) for p in files}
    result['__runtime_options__']=runtime_options(stage)
    return result

def runtime_options(stage):
    source=Path(__file__).with_name(MODULES[stage]+'.py').read_text()
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
