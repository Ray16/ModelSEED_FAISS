"""Verify and publish a completed refresh; never stamp stale outputs current."""

from amide_synthesis.infrastructure.paths import PROJECT
from amide_synthesis.infrastructure.review import REVIEW
from pathlib import Path
import sys,csv,json,hashlib,datetime
RUN=REVIEW
from amide_synthesis.infrastructure.freshness import RULES,status,require_inputs

def main():
    stages={s:status(s) for s in RULES}
    bad={s:v for s,v in stages.items() if v!='current'}
    if bad:raise RuntimeError('Refresh incomplete: '+json.dumps(bad))
    for s in RULES:require_inputs(s)
    def read(path):return list(csv.DictReader(path.open(),delimiter='\t'))
    links=read(PROJECT/'results/precomputed_link_candidates.tsv')
    assert all(r['mapping_evidence_tier'] in ['experimental_direct','curated_direct','homology_supported_direct','computational_direct'] for r in links)
    assert all('RHEA-COMP' not in r['mapping_rhea_ids'] for r in links)
    assert all(r['mapping_sequence_sha256']==hashlib.sha256(r['sequence'].encode()).hexdigest() for r in links)
    audit=read(RUN/'mapping/mapping_link_audit.tsv')
    rejected={(r['accession'],r['reaction']) for r in audit if r['component_collision']=='True' or r['noncatalytic_subunit']=='True'}
    leaked=rejected & {(r['uniprot'],r['rxn_id']) for r in links}
    if leaked:raise RuntimeError('Rejected mapping reappeared: '+str(leaked))
    manifest=json.loads((RUN/'cofold/manifest.json').read_text());selected=read(RUN/'sequences/selected50.tsv')
    assert {(r['accession'],r['reaction'],r['sequence_sha256']) for r in manifest['inputs']}=={(r['accession'],r['reaction'],r['sequence_sha256']) for r in selected}
    assert len(list((RUN/'cofold/state').glob('*.done.json')))==50
    audit_path=PROJECT/'results/mapping_robustness_20261019/after_audit.json'
    mapping_audit=json.loads(audit_path.read_text())
    if any(mapping_audit['problems'].values()):
        raise RuntimeError('Full mapping consistency audit failed: '+str(mapping_audit['problems']))
    if any(mapping_audit['frozen_50_panel'][key] for key in
           ['pairs_missing_from_mapping','sequence_hash_mismatches']):
        raise RuntimeError('Frozen cohort no longer supported by corrected mapping')
    for path,digest in mapping_audit['sha256'].items():
        if hashlib.sha256((PROJECT/path).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Mapping changed since independent audit: '+path)
    summaries={}
    for name in ['precomputed_link','master_candidates','enrichment','shortlist','converged','family_expansion']:
        p=PROJECT/'results'/(name+'_summary.json')
        if p.exists():summaries[name]=json.loads(p.read_text())
    doc={'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'status':'current_for_declared_input_snapshots','stage_status':stages,
         'refreshed_stages':len(stages),'summaries':summaries,
         'checks':{'all_precomputed_links_carry_evidence':True,'no_component_id_as_reaction':True,
                   'all_link_sequences_match_evidence_hash':True,'audited_rejected_pairs_do_not_reappear':True,
                   'frozen_50_protein_cohort_unchanged':True,'cofold_ensembles_complete':50,
                   'full_mapping_consistency_audit_passed':True,
                   'frozen_50_pairs_and_sequences_supported_by_corrected_mapping':True},
         'archive':'../../archive/refactor_20261019/snapshot',
         'mapping_correction_archive':'../mapping_robustness_20261019/before_correction',
         'refactor_manifest':'../../archive/refactor_20261019/manifest.json',
         'stage_registry':'../../amide_synthesis/stages.py',
         'earlier_archive':'archive/pipeline_before_mapping_refresh_20261018',
         'mapping_robustness_audit':'../mapping_robustness_20261019/after_audit.json',
         'freshness_registry':'../artifact_freshness.json',
         'limits':['Current means derived from the recorded inputs and code, not continuous live database retrieval.',
                   'Annotation evidence tiers are retained; this does not establish biochemical accuracy for every alias or inferred enzyme.',
                   'Raw literature and the frozen 50-protein campaign retain their original source snapshots.']}
    (RUN/'validation/pipeline_refresh_summary.json').write_text(json.dumps(doc,indent=2))
    print(json.dumps({'stages':len(stages),'links':len(links),'rejected_pairs_leaked':len(leaked),'cofolds':50},indent=2))

if __name__=='__main__':main()
