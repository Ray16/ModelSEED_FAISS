"""Report independent C5/C5 discovery, post-selection overlap and modeling status."""
import argparse
from datetime import datetime,timezone
import html
import hashlib
import json
from pathlib import Path

import pandas as pd


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def read_json(path,default=None):
    return json.loads(path.read_text()) if path.exists() else ({} if default is None else default)


def render(root):
    config=read_json(root/'campaign_config.json')
    reactions=read_json(root/'reactions/summary.json')
    mapping=read_json(root/'mapping/summary.json')
    overlap=read_json(root/'novelty_comparison.json')
    progress=read_json(root/'progress.json')
    if not config or not mapping:return
    selected=pd.read_csv(root/'reactions/selected_reactions.tsv',sep='\t').fillna('')
    plot_key=digest(root/'reactions/selected_reactions.tsv')+digest(root/'campaign_config.json')
    old=read_json(root/'figure_provenance.json')
    if old.get('input_key')!=plot_key:
        from amide_synthesis.evaluation.target_campaign import figures
        figures(root,config,selected)
        write_json(root/'figure_provenance.json',dict(input_key=plot_key,
            source='Fresh reaction tables and C5/C5 target configuration; shared visualization software only'))
    validated=set()
    for path in (root/'msa_batches').glob('*/complete.json'):
        validated.update(read_json(path).get('msa_sha256',{}))
    failures=[read_json(p) for p in (root/'msa_batches').glob('*/failure.json')]
    direct_count=sum(r.get('native_reaction_annotation',0) for r in mapping['selected_reaction_counts'])
    display=selected[['selection_rank','rxn_id','amide_rank','drfp_tanimoto','chemical_resemblance']].copy()
    diagram=read_json(root/'target_and_native_reactions.json')
    names={r['rxn_id']:r['title'] for r in diagram.get('native_reactions',[])}
    profiles=read_json(root/'mapping/derived_mapping_profiles.json').get('reactions',{})
    for row in selected.to_dict('records'):
        if row['rxn_id'] not in names:
            proposed=profiles.get(row['rxn_id'],{}).get('names',[])
            names[row['rxn_id']]=(proposed[0] if proposed else row.get('name') or row['rxn_id'])
    display['native_reaction']=display.rxn_id.map(names).fillna(display.rxn_id)
    display=display[['selection_rank','rxn_id','native_reaction','amide_rank','drfp_tanimoto','chemical_resemblance']]
    for field in ['drfp_tanimoto','chemical_resemblance']:display[field]=display[field].map(lambda v:f'{v:.3f}')
    new_path=root/'candidates_absent_from_checked_prior_tables.tsv'
    extra=pd.read_csv(new_path,sep='\t').fillna('') if new_path.exists() else pd.DataFrame()
    extra_table=extra[['accession','protein_name','rxn_id','mapping_route']].to_html(index=False,escape=True) if len(extra) else '<p>Overlap audit pending.</p>'
    counts=pd.DataFrame(mapping['selected_reaction_counts'])
    source=read_json(root/'results/corpus_manifest.json')
    status=progress.get('stage','fresh_msa_preparation')
    model_text=(f"Fresh alignment retrieval: {len(validated)}/{mapping['fold_candidates']} sequences verified in this campaign. "
        f"Stage: {status}. No prior alignment files or predicted structures are reused.")
    remote=read_json(root/'aurora/execution_status.json')
    lambda_manifest=read_json(root/'lambda_fleet/manifest.json')
    lambda_state=read_json(root/'lambda_fleet/state/coordinator.json')
    if lambda_manifest:
        model_text=(f"Lambda CUDA campaign: {lambda_state.get('active_workers',0)} active workers; "
            f"stage 1 verified for {lambda_state.get('stage1_complete',0)}/{mapping['fold_candidates']}, "
            f"both stages verified for {lambda_state.get('stage2_complete',0)}/{mapping['fold_candidates']}. "
            "All selected enzymes are in the queue. Lambda13 is excluded. Fresh C5 inputs and MSAs are used.")
    elif remote:
        model_text+=' '+str(remote.get('message',remote.get('status','')))
    else:
        model_text+=' Aurora prediction submission is pending fresh alignment completion and a successful access/runtime check.'
    submission=read_json(root/'aurora/submission_plan.json')
    package_info=('''<p>All declared enzyme inputs are packaged with the new C5 substrate definitions and fresh MSAs.
<a href="aurora/launch.sh">Aurora launch script</a> · <a href="aurora/submission_plan.json">Submission plan</a> ·
<a href="aurora/manifest.json">Frozen modeling inputs</a> · <a href="local_validation.json">Local validation</a>.</p>
<p>Planned Aurora structure location: <code>'''+html.escape(submission['remote_root']+'/results/')+'</code>. No structures are claimed before the real Aurora run.</p>') if submission else ''
    if lambda_manifest:
        package_info='<p><a href="lambda_fleet/analysis/rolling/report.html">Rolling cofold analysis</a> · <a href="lambda_fleet/state/coordinator.json">Live queue status</a> · <a href="lambda_fleet/manifest.json">Lambda inputs and runtime</a> · <a href="lambda_fleet/results/">Structure output directory</a>.</p><p>The Aurora package is retained. The active provider is Lambda1/Lambda2, using stock Boltz-2 in FP32 with the established serial-sampling correction.</p>'
    chemistry=dict(free_acid='O=C([O-])CCCC(=O)[O-]',cadaverine='NCCCCC[NH3+]',
        step1='Protein + glutarate + ATP + mechanism-associated Mg',
        step2='Protein + glutaryl-AMP or glutaryl-phosphate + cadaverine; ADP retained for the phosphate route',
        donor_carbons=5,acceptor_carbons=5,samples_per_stage=50,seed=42)
    write_json(root/'target_modeling_chemistry.json',chemistry)
    esc=lambda x:html.escape(str(x))
    primary=''
    primary_root='lambda_fleet/analysis/family_landscape'
    primary_name='enzyme_landscape'
    if not (root/primary_root/(primary_name+'.svg')).exists():
        primary_root='lambda_fleet/analysis/two_step'
        primary_name='two_step_side_by_side'
    if not (root/primary_root/(primary_name+'.svg')).exists():
        primary_root='lambda_fleet/analysis/upper_left'
        primary_name='upper_left_side_by_side'
    if (root/primary_root/(primary_name+'.svg')).exists():
        primary=(f'<h2>Main visualization: integrated reaction geometry</h2>'
            f'<a href="{primary_root}/report.html"><img src="{primary_root}/{primary_name}.svg" '
            'style="max-width:1800px" alt="Candidate reaction geometry with aligned labeled panels"></a>'
            f'<p><a href="{primary_root}/report.html">Labeled candidates and metrics</a> · '
            f'<a href="{primary_root}/{primary_name}.pdf">Main figure PDF</a> · '
            '<a href="lambda_fleet/analysis/visualizations/report.html">Complete cohort and step 2 results</a></p>')
    text=f'''<!doctype html><html><head><meta charset="utf-8"><title>{esc(config['title'])}</title>
<style>body{{font:15px/1.6 system-ui;max-width:1450px;margin:30px auto;padding:0 24px;color:#183348}}h1,h2{{line-height:1.25}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:8px;border-bottom:1px solid #dce5e9;text-align:left}}th{{background:#eef4f6}}a{{color:#00687a}}img{{max-width:100%}}.status{{padding:16px;background:#edf5f6;border-left:4px solid #007f86}}.metric{{font-size:22px;font-weight:650}}</style></head><body>
<h1>{esc(config['title'])}</h1><p>Updated {datetime.now(timezone.utc).isoformat()}. The earlier 825-candidate preparation is superseded and its launcher is disabled.</p>
{primary}
<p class="metric">{mapping['fold_candidates']} independently selected sequences · {overlap.get('additional_sequences','pending')} additional to the previous selected pool · {overlap.get('additional_accessions_absent_from_wider_prior_tables','pending')} absent from the checked wider prior tables</p>
<p class="status">{esc(model_text)}</p>
{package_info}
<p><a href="mapping/fold_candidates.tsv">Full cofold candidate table</a> · <a href="mapping/fold_candidates.fasta">Cofold FASTA</a> · <a href="candidates_absent_from_checked_prior_tables.tsv">Additional candidates absent from checked prior tables</a> · <a href="comparison_after_selection.tsv">Post-selection overlap audit</a> · <a href="progress.json">Pipeline stage</a></p>
<h2>Independent discovery inputs</h2><p>This campaign rebuilt molecular normalization and reaction fingerprints from the ModelSEED source tables ({source.get('n_reactions',0):,} representable reactions; source commit {esc(source.get('commit','')[:12])}). It recomputed eligibility from product amide disconnections, free-amine reactants, macromolecular context and atom/charge balance. It did not read a previous screened reaction list, enzyme shortlist, mapping profile or alignment cache as a discovery input.</p>
<p>The C5/C5 target uses glutarate (cpd00379) and cadaverine (cpd01155), producing one amide with a retained acid and primary amine (C10H20N2O3). Four freshly balanced framings retain cofactors: ATP→AMP+PPi, ATP→ADP+Pi, direct condensation and glutaryl-CoA aminolysis. The CoA framing is a separate activated-donor alternative.</p>
<h2>Reaction selection</h2><p>{reactions.get('searchable_reactions',0):,} searchable reactions → {reactions.get('fresh_screen_eligible',0)} freshly screened amidations → 100 DRFP–Tanimoto hits → ten chemical analogues. These ten match the first C5/C5 attempt; nine overlap with the earlier adipate/HMDA panel. No requirement for a new reaction ID or a new enzyme ID influenced selection. The chemical score is a declared heuristic, not an activity probability.</p>
{display.to_html(index=False,escape=True)}<img src="target_and_native_reactions.png" alt="New C5/C5 target and five native reaction analogues">
<img src="similarity_heatmap.png" style="max-width:900px" alt="Native reaction similarities with cofactors retained">
<p><a href="reactions/target_queries.tsv">Balanced C5/C5 queries</a> · <a href="reactions/chemical_review.tsv">Chemical features and contributions</a> · <a href="reactions/manifest.json">Independent reaction provenance</a></p>
<h2>Fresh enzyme mapping and sequence selection</h2><p>Mappings were resolved from each selected reaction's own ModelSEED aliases and fresh Rhea/UniProt responses. Expansion vocabulary was derived from the current reaction-annotated reference proteins. EC numbers were scoped to the matching reaction statement, preventing another activity on the same protein from supplying the search EC. No previous selected-reaction profile was imported.</p>
<p>UniProt {esc(mapping['uniprot_release'])} yielded {mapping['unique_accessions']} unique candidate identifiers and {mapping['unique_sequences']} sequences. The retained set has {mapping['clusters95']} groups at 95% identity; both sequences require 90% aligned coverage. Curated reference sequences are retained, exact duplicates are folded once, and fragments/nonstandard sequences are excluded, giving {mapping['fold_candidates']} cofold inputs. EC/name-supported inferred candidates require at least 30% sequence identity and 80% coverage of both reference and candidate. These thresholds support hypotheses; they do not establish substrate specificity.</p>
{counts.to_html(index=False,escape=True)}
<p>The glycylglycine equation has no exact established mapping here; related L-amino-acid ligases remain explicitly exploratory. The newly searched phosphinothricin-tripeptide equation yielded {mapping['context_proteins']} curated context proteins, including carrier-dependent components, and no justified standalone free-acid/diamine ligase assignment. These records are retained in <a href="mapping/reaction_context.tsv">reaction context</a> rather than silently assigned as independent catalysts.</p>
<h2>What is additional to earlier work</h2><p>The overlap audit ran only after the independent cohort was frozen. Of {mapping['fold_candidates']} sequences, {overlap.get('seen_in_previous_selected_pool','pending')} match the previous selected ten-reaction pool or its literature references and {overlap.get('additional_sequences','pending')} are additional to that pool. Checking broader prior candidate/reference tables leaves {overlap.get('additional_accessions_absent_from_wider_prior_tables','pending')} additional accessions. These are new to the stated comparison tables, not newly evolved proteins or experimentally discovered C5/C5 activities. Many are homologs in familiar enzyme families. Expanded mapping coverage contributes to this difference; it must not be attributed solely to shortening the two monomers.</p>
{extra_table}<p><a href="novelty_comparison.json">Exact comparison scope and counts</a> · <a href="mapping/derived_mapping_profiles.json">Profiles derived within this run</a> · <a href="mapping/chemical_checks.json">Full native-equation checks</a> · <a href="mapping/query_snapshots.json">Fresh query provenance</a></p>
<h2>Cofold chemistry and completion criteria</h2><p>Stage 1 uses protein, free glutarate, ATP and mechanism-associated Mg. Stage 2 uses the activated glutaryl intermediate and cadaverine, retaining ADP for the acyl-phosphate route. Both donor and diamine are C5. Each stage requests 50 poses, seed 42, three recycles and 200 sampling steps, using the Aurora FP32 implementation. Fresh MSAs are requested in batches of up to 101 with exact query-sequence validation.</p>
<p>Every output is checked for protein identity, all requested ligand compositions, cofactors and 50 exact model/confidence slots. Stage-2 pocket restraints are derived from each enzyme's own new glutarate models. Geometry and confidence are hypotheses about structural accommodation, not measured ligation or polymerization. The forced pocket makes stage-2 accommodation conditional; hydrogen-free outputs do not identify the neutral cadaverine amine geometrically.</p>
<p><a href="target_modeling_chemistry.json">Explicit new substrate definitions</a> · <a href="mapping/manifest.json">Fresh mapping and frozen selection provenance</a></p></body></html>'''
    (root/'report.html').write_text(text)
    inputs=[Path(__file__),root/'campaign_config.json',root/'reactions/summary.json',root/'reactions/selected_reactions.tsv',
        root/'mapping/summary.json',root/'mapping/counts.tsv',root/'novelty_comparison.json',root/'progress.json',
        root/'candidates_absent_from_checked_prior_tables.tsv']
    inputs += [p for p in [root/'aurora/execution_status.json',root/'aurora/submission_plan.json',root/'local_validation.json'] if p.exists()]
    inputs += [p for p in [root/'lambda_fleet/manifest.json',root/'lambda_fleet/state/coordinator.json'] if p.exists()]
    write_json(root/'report_manifest.json',dict(status='current_status_snapshot',
        input_sha256={str(p):digest(p) for p in inputs if p.exists()},
        output_sha256={name:digest(root/name) for name in ['report.html','target_modeling_chemistry.json',
            'target_and_native_reactions.png','similarity_heatmap.png']}))
    print(root/'report.html')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True)
    a=p.parse_args();render(a.campaign.resolve())


if __name__=='__main__':main()
