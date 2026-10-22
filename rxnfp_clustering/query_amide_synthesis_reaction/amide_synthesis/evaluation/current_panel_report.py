"""Publish current panel facts from selection, mapping and actual saved structures."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
from html import escape
import json
from pathlib import Path
import shutil

from amide_synthesis.infrastructure.paths import PROJECT
from amide_synthesis.evaluation.mapping_audit_report import render
from amide_synthesis.structure_modeling.poses import inspect


def read_table(path):
    return list(csv.DictReader(Path(path).open(),delimiter='\t'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geometry_plot(references, statuses, manifest, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    items={r['accession']:r for r in manifest['inputs']}
    rows=[]
    for record in references:
        accession=record['accession'];state=statuses[accession]
        gene=record['gene_names'].split(';')[0] or record['protein_name']
        label=gene[0].upper()+gene[1:]
        if gene.lower()=='glna2':
            label+=' (Haloferax)' if record['organism'].startswith('Haloferax') else ' (Streptomyces)'
        inferred=record['mapping_route']!='reaction_annotation'
        if inferred:label+=' (candidate)'
        rows.append(dict(reaction=record['rxn_id'],accession=accession,enzyme=record['protein_name'],
                         gene=record['gene_names'],organism=record['organism'],label=label,
                         mapping_route=record['mapping_route'],phosphate=items[accession]['phosphate'],
                         status=state['status'],models=state['models'],
                         angle_deg=state.get('angle_deg'),harmonic_distance_A=state.get('harmonic_distance_A')))
    with (output/'activation_geometry.tsv').open('w') as handle:
        writer=csv.DictWriter(handle,list(rows[0]),delimiter='\t');writer.writeheader();writer.writerows(rows)
    fig,axes=plt.subplots(1,2,figsize=(10,max(4.5,.36*len(rows)+1.8)),sharey=True,layout='constrained')
    for index,row in enumerate(rows):
        if row['status']!='verified':
            for axis in axes:axis.text(.02,index,'pending',transform=axis.get_yaxis_transform(),color='#888888',fontsize=9)
            continue
        color='#0072B2' if row['phosphate']=='PA' else '#D55E00'
        inferred=row['mapping_route']!='reaction_annotation'
        for axis,key in zip(axes,['angle_deg','harmonic_distance_A']):
            value=row[key]
            axis.scatter(value,index,s=48,marker='o' if row['phosphate']=='PA' else 's',
                         facecolors='none' if inferred else color,edgecolors=color,linewidths=1.4,zorder=3)
            axis.annotate(f'{value:.1f}',(value,index),xytext=(6,-3),textcoords='offset points',fontsize=8)
    axes[0].set_yticks(range(len(rows)),[r['label'] for r in rows])
    axes[0].invert_yaxis();axes[0].set_xlim(0,198);axes[0].set_xticks([0,60,120,180])
    axes[0].set_xlabel('Proximity-weighted attack angle (°)')
    maximum=max([r['harmonic_distance_A'] for r in rows if r['status']=='verified'] or [10])
    axes[1].set_xlim(0,maximum*1.23);axes[1].set_xlabel('Harmonic attack distance (Å)')
    for axis in axes:
        axis.spines[['top','right']].set_visible(False);axis.grid(axis='x',alpha=.15);axis.set_axisbelow(True)
    completed=sum(r['status']=='verified' for r in rows)
    fig.suptitle(f'ATP–Mg–adipate cofolds · {completed}/{len(rows)} ensembles complete',fontsize=13)
    legend=[Line2D([],[],marker='o',color='#0072B2',linestyle='',label='AMP / PA route'),
            Line2D([],[],marker='s',color='#D55E00',linestyle='',label='ADP / PG route'),
            Line2D([],[],marker='s',markerfacecolor='none',color='#444444',linestyle='',label='BacD: Gly+Gly mapping unverified')]
    fig.legend(handles=legend,loc='outside lower center',ncol=3,frameon=False,fontsize=9)
    for extension in ['png','svg','pdf']:
        fig.savefig(output/('activation_geometry.'+extension),dpi=200,bbox_inches='tight')
    plt.close(fig)
    return rows


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--markdown',type=Path,default=PROJECT/'docs/COLLABORATOR_RESPONSE.md')
    args=parser.parse_args(argv)
    reaction_root=PROJECT/'results/modelseed_nylon_top10'
    root=PROJECT/'results/top10_mapping'
    audit=root/'20261020_chemical_panel'
    campaign=PROJECT/'results/top10_cofold/campaign'
    reactions=read_table(reaction_root/'top10_reactions.tsv')
    summary=json.loads((audit/'summary.json').read_text())
    references=read_table(PROJECT/'results/top10_cofold/references.tsv')
    manifest=json.loads((campaign/'manifest.json').read_text())
    statuses={}
    for item in manifest['inputs']:
        if not (campaign/'state'/(item['name']+'.done.json')).exists():
            statuses[item['accession']]={'status':'pending','models':0}
            continue
        actual=inspect(item,campaign/'outputs'/item['name'],include_near_attack=False)
        statuses[item['accession']]={'status':'verified','models':actual['n_models'],
                                   'angle_deg':actual['A_deg'],'harmonic_distance_A':actual['Dh_A']}
    completed=sum(r['status']=='verified' for r in statuses.values())
    models=sum(r['models'] for r in statuses.values())
    geometry_plot(references,statuses,manifest,PROJECT/'results/top10_cofold')
    validation=json.loads((reaction_root/'original_five_validation.json').read_text())
    from amide_synthesis.reaction_search.chemical_selection import load_policy
    weights=load_policy()['weights']
    sensitivity=validation['weight_sensitivity']
    retained_range=(min(r['previous_five_retained'] for r in sensitivity),max(r['previous_five_retained'] for r in sensitivity))
    names={r['rxn_id']:r['display_title'] for r in reactions}
    comparison_path=PROJECT/'results/top10_cofold/glnA2_comparison/comparison.json'
    comparison_note=''
    if comparison_path.exists():
        comparison=json.loads(comparison_path.read_text())
        first=comparison['ensembles']['I3R584'];second=comparison['ensembles']['Q9RDS6']
        identity=comparison['sequence_alignment']['identity_including_gaps']
        comparison_note=(f"The two GlnA2 proteins have {identity:.1%} global sequence identity. "
            f"Adipate–ATP distances are below 6 Å in {first['poses_under6A']}/50 Haloferax poses "
            f"and {second['poses_under6A']}/50 Streptomyces poses. Neither ensemble has a pose below 4 Å. "
            "This is a model-placement difference, not evidence of different measured activity. "
            "The ensembles are monomeric and differ in reuse/implementation provenance. "
            "[Pose comparison](../results/top10_cofold/glnA2_comparison/pose_comparison.png).")
    for row in summary['counts']:
        row['reaction']=names[row['rxn_id']]
    summary['limits'] += [
        'rxn09897 has no established exact Gly+Gly assignment. Related BacD/YwfE candidates remain inferred.',
        'rxn22343 matches an annotated Rhea ID, but Rhea contains a symbolic participant; full structural equivalence is not certified.',
        'Chemical resemblance and reference consistency do not validate activity or substrate promiscuity.']
    render(summary,root)
    table='| ModelSEED | Selected reaction | Annotated IDs | Expanded IDs | Exact sequences | 99% reps | 95% reps |\n|---|---|---:|---:|---:|---:|---:|\n'
    for row in summary['counts']:
        table+='| '+' | '.join(str(row[k]) for k in ['rxn_id','reaction','live_reaction_supported_accessions',
                  'expanded_candidates','expanded_exact_sequences','expanded_clusters99','expanded_clusters95'])+' |\n'
    totals=summary['totals']
    fragment_count=sum(row['fragment_expanded'] for row in summary['counts'])
    replay=PROJECT/'results/reproducibility/current_top10_20261020'
    replay_manifest=json.loads((replay/'manifest.json').read_text())
    for threshold in [99,95]:
        for extension in ['tsv','fasta']:
            name=f'representatives{threshold}.{extension}'
            source=replay/'mapping'/name
            if sha(source)!=replay_manifest['output_sha256']['mapping/'+name]:
                raise ValueError('Verified representative export changed: '+name)
            shutil.copy2(source,root/name)
    table+=f"| **Unique total** | | **{totals['unique_trusted_accessions']}** | **{totals['unique_expanded_accessions']:,}** | **{totals['expanded_unique_sequences']:,}** | **{totals['expanded_clusters99']}** | **{totals['expanded_clusters95']}** |\n"
    reference_rows=''
    for record in references:
        native=record['mapping_route']=='reaction_annotation'
        reference_rows+='<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in [record['rxn_id'],
            record['gene_names'],record['protein_name'],record['organism'],record['accession'],
            'reaction annotation' if native else 'related-reaction candidate',statuses[record['accession']]['status']])+'</tr>'
    page=(root/'report.html').read_text()
    page=page.replace('<style>','<base href="20261020_chemical_panel/"><style>',1)
    page=page.replace('src="mapping_counts.svg"','src="../mapping_counts.svg"')
    intro=(f'<h2>Current structure coverage: {completed}/{len(statuses)} proteins, {models} checked models</h2>'
           '<p>Exact/MSA/50-pose monomeric ATP–Mg–adipate ensembles. No HMDA or activated intermediate. '
           'The glycylglycine-associated BacD model is an inferred candidate, not an exact native-reaction assignment.</p>'
           '<p><a href="../../modelseed_nylon_top10/report.html">Chemical selection and native structures</a> · '
           '<a href="../../modelseed_nylon_top10/chemical_selection_components.svg">Chemical contributions</a> · '
           '<a href="../../modelseed_nylon_top10/original_five_validation.json">Previous-five validation</a> · '
           '<a href="../../top10_cofold/campaign/manifest.json">Model inputs and reuse provenance</a></p>'
           '<p><a href="../../top10_cofold/activation_geometry.png">Updated cofold geometry plot</a> · '
           '<a href="../../top10_cofold/activation_geometry.tsv">Geometry table</a> · '
           '<a href="../representatives95.fasta">95% representatives FASTA</a></p>'
           '<table><tr><th>Reaction</th><th>Gene/common names</th><th>Enzyme name</th><th>Organism</th>'
           '<th>UniProt</th><th>Evidence</th><th>Models</th></tr>'+reference_rows+'</table>')
    page=page.replace('<h1>10-reaction enzyme mapping audit</h1>','<h1>Ten-reaction enzyme candidate panel</h1>'+intro)
    (root/'report.html').write_text(page)
    draft=f'''# Amide synthetase candidates: current ten-reaction panel

Updated {datetime.now(timezone.utc).strftime('%d %B %Y, %H:%M UTC')}. Target: adipate + hexamethylenediamine (HMDA) → nylon-6,6 amide diad.

## 1. How are reactions retrieved and chemically selected?

**DRFP–Tanimoto retrieves the top 100** from 480 balanced amide-forming reactions in the 33,246-reaction searchable corpus. Cofactors remain included; ECFP4-difference + DRFP cosine breaks retrieval ties.

**Chemical selection then chooses ten**, rather than taking the first ten DRFP hits. The code checks free-acid/primary-amine amide formation and softly compares functional architecture, retained acid/amine ends, carbon spacers, backbone and substrate size. Monoacids, monoamines, ω-amino acids, AB self-condensation, rings and aromatic substrates remain eligible.

The declared weights are {weights['architecture']:.0%} architecture, {weights['retained_ends']:.0%} retained ends, {weights['spacers']:.0%} spacers, {weights['backbone']:.0%} backbone and {weights['size']:.0%} size. They are transparent chemical preferences, **not activity probabilities**. Every component and explanation is exported; original reaction IDs never control selection.

**Validation against the previous five:** {validation['previous_five_retained']}/5 remain in the ten without being forced in. Across {len(sensitivity)} tested weight perturbations, retention ranged from {retained_range[0]}/5 to {retained_range[1]}/5 (each component varied from 0.5 to 1.5 times its default, then renormalized). This checks consistency and sensitivity, not experimental accuracy.

[Chemical selection and structures](../results/modelseed_nylon_top10/report.html) · [Component visualization](../results/modelseed_nylon_top10/chemical_selection_components.png) · [Validation details](../results/modelseed_nylon_top10/original_five_validation.json).

## 2. How are reactions mapped to enzymes, and how many remain?

Map ModelSEED → Rhea → matching UniProt reaction/direction annotations, preserving their own evidence. Equivalent-equation bridges and complete Rhea/ChEBI structures are checked separately. EC/name candidates also require ≥30% sequence identity and ≥80% coverage of both proteins against an annotated reference.

Remove identical sequences, then cluster within each reaction at 99%/95% identity with ≥90% coverage of both proteins. Preserve every accession and membership; genome labels alone never merge proteins.

{table}
**Qualifications:** glycylglycine formation has no established exact reaction assignment; its 16 candidates come from related L-amino-acid ligases. BacD/YwfE normally joins alanine and anticapsin; its Gly+Gly assignment remains unverified. MurE’s Rhea entry contains a symbolic participant, limiting full structural verification. Missing stereochemistry remains flagged for citrate-derived entries. Annotation supports native activity, not adipate/HMDA activity.

{fragment_count} fragment records remain explicitly flagged in the expanded pools; they are not selected reference models.

[Evidence, named enzymes and mapping visualization](../results/top10_mapping/report.html) · [All candidate records](../results/top10_mapping/20261020_chemical_panel/reaction_candidates.tsv).

[95% representative FASTA](../results/top10_mapping/representatives95.fasta) · [Identifier/evidence table](../results/top10_mapping/representatives95.tsv).

## 3. Which enzymes are modeled?

The reference panel contains **13 proteins**: 12 with matching reaction annotations and one related-reaction BacD/YwfE candidate. It includes PuuA; GlnA2 from Haloferax mediterranei; GlnA2 and GlnA3 from Streptomyces coelicolor; DdaG; SfnaD/SfaD; SbnE; AsbA; MurE; SbnC; SbnF; AsbB; and BacD/YwfE. The two GlnA2 proteins are distinct sequences.

{comparison_note}

**Current checked coverage: {completed}/13 proteins, {models}/650 poses.** Ten existing ensembles were sequence/input-checked and reused. The three new ensembles are Haloferax GlnA2, Streptomyces GlnA3 and BacD/YwfE. Each uses exact Boltz-2, an exact-sequence MSA and 50 diffusion samples; execution uses the shared GPU reservation gate.

Complexes contain **protein + ATP + Mg + adipate**, without HMDA or activated intermediate. These are comparative acid-activation models, not proof of amidation, polymerization or correct biological assembly. The BacD model does not establish glycylglycine activity. This reference panel is separate from a larger sequence-diversity panel.

[Model inputs/provenance](../results/top10_cofold/campaign/manifest.json) · [Current coverage](../results/top10_mapping/current_panel_status.json) · [Model output folders](../results/top10_cofold/campaign/outputs).

[Updated cofold geometry plot](../results/top10_cofold/activation_geometry.png) · [Geometry table](../results/top10_cofold/activation_geometry.tsv). Rows follow the reaction panel, not predicted activity. Angle and distance remain structural descriptors.

Harmonic distance summarizes the 50 pose distances as 50 / Σ(1/d), emphasizing closer poses. The unverified-candidate label concerns enzyme mapping, independently of this distance metric.

## 4. What were Josh’s five literature enzymes?

| Enzyme | Original UniProt | Native transformation | ModelSEED |
|---|---|---|---|
| DdaG | E2JA32 | Fumarate + L-diaminopropionate → fumaroyl-DAP. | rxn46045 |
| SfaB | A0A2H4T912 | 3-Isocyanobutanoate → acyl-AMP. | rxn60314 |
| CphA1 | P73833 | Cyanophycin backbone Asp addition and Arg side-chain ligation. | rxn48822, rxn49577 |
| McbA | R4R1U5 | β-Carboline carboxylate + β-phenethylamine → marinacarboline amide. | No exact native entry located. |
| PylC | Q46E79 | Methyl-D-ornithine + L-lysine → N⁶-linked isopeptide. | rxn23366 |

All five UniProt sequences match the corresponding GenBank records in the saved audit. CphA strains/constructs sharing the SuCphA1 name are not interchangeable. DdaG and SfaB have reported adipate/HMDA activity in the supplied literature.

[Native structures and nylon comparison](../results/native_nylon_similarity/similarity_heatmap.png) · [Sequence audit](../results/native_reaction_audit/20261020T204408703497Z/manifest.json).

## 5. Does another structural statistic help predict yield?

Adding log harmonic distance to angle improved producer-classification AUROC from **0.659 to 0.703** in the 49-enzyme panel. The improvement interval spans zero; continuous-titre prediction did not improve, and unseen-enzyme transfer failed in the small literature dataset. Distance remains exploratory. Poses are not independent experiments; paralog selectivity needs experimental testing.

[Yield visualization](../results/colleague_review_20261017/figures/yield_validation.png) · [Validation audit](../results/colleague_review_20261017/validation/yield_analysis_audit.json).

## 6. Reproducibility and notes for the mapping colleague

Queries, chemical selection, mapping, sequence filtering, reference selection, model reuse and structural checks are code. Configuration and raw responses retain provenance. See [reproduction instructions](REPRODUCIBILITY.md) and the [chemical policy](../config/chemical_selection_policy.json).

For the colleague: TSV formatting is consistent, but some links contain RHEA-COMP participant/reaction-number collisions. Preserve reaction-specific evidence and catalytic roles; EC/GO/name matches alone are weaker. Unresolved links are not automatically errors. Missing database aliases and our separate rebuilt-parser bug must not be conflated with defects in the colleague’s code.

[Detailed mapping review](../MAPPING_REVIEW.md).
'''
    args.markdown.write_text(draft)
    status={'checked_utc':datetime.now(timezone.utc).isoformat(),'selected_reactions':[r['rxn_id'] for r in reactions],
            'mapping_totals':totals,'reaction_counts':summary['counts'],'completed_ensembles':completed,
            'expected_ensembles':len(statuses),'checked_models':models,'expected_models':50*len(statuses),
            'protein_status':statuses,'previous_five_retained':validation['previous_five_retained'],
            'glycylglycine_assignment':'Related-reaction candidate only; exact native assignment not established'}
    (root/'current_panel_status.json').write_text(json.dumps(status,indent=2)+'\n')
    (root/'current_report_manifest.json').write_text(json.dumps({
        'input_sha256':{str(p):sha(p) for p in [reaction_root/'top10_reactions.tsv',audit/'manifest.json',
                     campaign/'manifest.json',reaction_root/'original_five_validation.json',Path(__file__)]
                     +([comparison_path] if comparison_path.exists() else [])},
        'output_sha256':{str(p):sha(p) for p in [root/'report.html',root/'current_panel_status.json',args.markdown,
                      root/'mapping_counts.svg',root/'mapping_counts.png',root/'mapping_counts.pdf',
                      PROJECT/'results/top10_cofold/activation_geometry.png',
                      PROJECT/'results/top10_cofold/activation_geometry.svg',
                      PROJECT/'results/top10_cofold/activation_geometry.pdf',
                      PROJECT/'results/top10_cofold/activation_geometry.tsv',
                      root/'representatives95.fasta',root/'representatives95.tsv',
                      root/'representatives99.fasta',root/'representatives99.tsv']}},indent=2)+'\n')
    print(f'Published ten-reaction report: {completed}/{len(statuses)} proteins, {models} actual models checked.')


if __name__=='__main__':
    main()
