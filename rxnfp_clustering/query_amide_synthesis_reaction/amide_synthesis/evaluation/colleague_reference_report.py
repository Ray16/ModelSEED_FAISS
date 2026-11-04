"""Build a shareable Markdown reference draft from source mappings and matched assays."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import warnings

import gemmi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error,r2_score,roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import yaml

PROJECT=Path(__file__).resolve().parents[2]

SOURCE=Path('/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent/exp_data/amide_synthetase')
REFS={'DdaG':'E2JA32','SfaB':'A0A2H4T912','SuCphA1':'P73833','McbA':'R4R1U5','PylC':'Q46E79'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def js(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def tsv(path,rows):pd.DataFrame(rows).to_csv(path,sep='\t',index=False)


def stage(root,record,number):
    path=root/'results'/record['accession']/('stage'+str(number)+'.json')
    if not path.exists():return None
    data=json.loads(path.read_text());poses=data.get('poses',[])
    if (data.get('status') not in ['verified','verified_with_warnings'] or len(poses)!=50 or
        data.get('protein_sequence_sha256')!=record['sequence_sha256'] or
        data.get('source_input_config_sha256')!=record['config_sha256'] or
        data.get('source_msa_sha256')!=record['msa_sha256']):raise ValueError('Invalid modeling state: '+str(path))
    return poses


def features(distance,angle):
    d=np.asarray(distance,float);a=np.asarray(angle,float)
    if len(d)!=50 or not np.isfinite(d).all() or not np.isfinite(a).all() or (d<=0).any():raise ValueError('Expected 50 finite valid poses')
    w=1/d
    return dict(n_poses=50,weighted_angle_deg=float(np.sum(w*a)/w.sum()),harmonic_distance_A=float(50/w.sum()),
        close_poses=int((d<4).sum()),near_attack_fraction=float(((d<4)&(a>150)).mean()),
        smooth_joint_support=float((1/(1+np.exp((d-3.5)/.5))*np.exp(-.5*((a-180)/30)**2)).mean()))


def reference_geometry():
    rows=[];pose_rows=[]
    metadata=pd.read_csv(PROJECT/'results/five_tested_enzymes_cofolds/geometry_summary.tsv',sep='\t').fillna('')
    for enzyme,accession in REFS.items():
        label='CphA1' if enzyme=='SuCphA1' else enzyme
        directory=PROJECT/'results/five_tested_enzymes_cofolds/ensembles'/(label+'_'+accession)
        files=sorted(directory.glob('*.pdb'))
        if len(files)!=50:raise ValueError('Reference ensemble incomplete: '+enzyme)
        phosphate=metadata.loc[metadata.enzyme==enzyme,'phosphate'].iloc[0]
        for path in files:
            model=gemmi.read_structure(str(path))[0]
            nucleotides=[c for c in model if {'PA','PG','O3A','O3B'}<={a.name for r in c for a in r}]
            if len(nucleotides)!=1:raise ValueError('ATP identity ambiguous: '+str(path))
            atoms={a.name:a.pos for r in nucleotides[0] for a in r}
            acids=[]
            for c in model:
                heavy=[a for r in c for a in r if a.element.name!='H']
                if len(heavy)==10 and sum(a.element.name=='C' for a in heavy)==6 and sum(a.element.name=='O' for a in heavy)==4:acids.append(heavy)
            if len(acids)!=1:raise ValueError('Reference is not a unique C6 adipate complex')
            p=atoms[phosphate];leaving=atoms['O3A' if phosphate=='PA' else 'O3B']
            oxygen=min([a.pos for a in acids[0] if a.element.name=='O'],key=lambda o:o.dist(p))
            u=np.array([oxygen.x-p.x,oxygen.y-p.y,oxygen.z-p.z]);v=np.array([leaving.x-p.x,leaving.y-p.y,leaving.z-p.z])
            angle=float(np.degrees(np.arccos(np.clip(u@v/np.linalg.norm(u)/np.linalg.norm(v),-1,1))))
            pose_rows.append(dict(enzyme=enzyme,accession=accession,distance_A=oxygen.dist(p),angle_deg=angle,source=str(path)))
        data=[r for r in pose_rows if r['enzyme']==enzyme]
        old=metadata[metadata.enzyme==enzyme].iloc[0]
        rows.append(dict(enzyme=enzyme,accession=accession,substrate='adipate/HMDA benchmark; step-1 adipate complex',
            **features([r['distance_A'] for r in data],[r['angle_deg'] for r in data]),
            reported_result=old.reported_nylon_result,protocol_note=old.protocol_note))
    return pd.DataFrame(rows),pd.DataFrame(pose_rows)


def mapping_tables(campaign,out):
    mapping=campaign/'mapping'
    edges=pd.read_csv(mapping/'reaction_candidates.tsv',sep='\t').fillna('')
    members=pd.read_csv(mapping/'cluster_membership.tsv',sep='\t').fillna('')
    cohort=pd.read_csv(mapping/'fold_candidates.tsv',sep='\t').fillna('')
    selected=pd.read_csv(campaign/'reactions/selected_reactions.tsv',sep='\t').fillna('')
    rep=members.set_index('accession').representative.to_dict()
    exact=members.set_index('accession').exact_representative.to_dict()
    equations=selected.set_index('rxn_id').definition.to_dict()
    edges['cluster95_representative']=edges.accession.map(rep)
    edges['exact_sequence_representative']=edges.accession.map(exact)
    edges['retrieved_native_equation']=edges.rxn_id.map(equations).fillna('Independent substrate-annotation query')
    edges['sequence_selected_for_cofold']=edges.accession.isin(cohort.accession)
    edges['assignment_scope']=np.where(edges.mapping_route=='native_reaction_annotation','Native reaction annotation; target C5/C5 activity unestablished','Related/name/homology evidence; native equation assignment not established')
    columns=['accession','protein_name','organism','rxn_id','retrieved_native_equation','mapping_route','assignment_scope','native_reactions','reaction_evidence','reference_accession','identity','query_coverage','reference_coverage','exact_sequence_representative','cluster95_representative','sequence_selected_for_cofold']
    edges[columns].to_csv(out/'tables/enzyme_reaction_provenance.tsv',sep='\t',index=False)
    counts=[];links=[]
    for row in selected.to_dict('records'):
        group=edges[edges.rxn_id==row['rxn_id']]
        representatives=set(group.cluster95_representative)
        chosen=set(cohort.accession)&(representatives|set(group.accession))
        counts.append(dict(reaction=row['rxn_id'],native_equation=row['definition'],raw_accessions=group.accession.nunique(),
            exact_sequences=group.sequence_sha256.nunique(),clusters95=len(representatives),cofold_sequences=len(chosen),
            native_annotation_accessions=int((group.mapping_route=='native_reaction_annotation').sum()),
            inferred_or_related_accessions=int((group.mapping_route!='native_reaction_annotation').sum())))
        for accession in sorted(chosen):
            direct=group[group.accession==accession]
            support=group[group.cluster95_representative==accession]
            links.append(dict(cofold_accession=accession,reaction=row['rxn_id'],native_equation=row['definition'],
                link_type='direct_source_assignment' if len(direct) else 'source_assignment_of_95percent_cluster_member',
                direct_mapping_routes=';'.join(sorted(set(direct.mapping_route))),supporting_source_accessions=';'.join(sorted(set(support.accession)|set(direct.accession)))))
    counts=pd.DataFrame(counts);counts.to_csv(out/'tables/reaction_95_counts.tsv',sep='\t',index=False)
    tsv(out/'tables/cofold_sequence_reaction_links.tsv',links)
    if counts.cofold_sequences.sum()!=len(cohort):raise ValueError('Reaction allocation does not reconcile with cohort')
    return counts


def assay_analysis(out):
    panel=pd.read_csv(SOURCE/'assay_data/panel_activity.csv')
    poses=pd.read_csv(SOURCE/'structural_proxies/tables/continuous_pose_geometry.csv')
    poses=poses[poses.group=='calibration']
    geometry=pd.DataFrame([dict(id=name,**features(g.distance_A,g.angle_deg)) for name,g in poses.groupby('id')])
    data=panel.merge(geometry,on='id',validate='one_to_one')
    if len(data)!=49 or not set(['AS'+str(i) for i in range(1,49)])<=set(data.id):raise ValueError('Matched assay panel incomplete')
    data['titre_ugml']=data.activity_N66L1_ugml_avg
    data['log_distance']=np.log(data.harmonic_distance_A)
    checks=[]
    for row in panel[panel.sequence.notna()].itertuples():
        path=SOURCE/'results/modeling/boltz2/step1_1mg/inputs'/('step1_'+row.id+'.yaml')
        doc=yaml.safe_load(path.read_text());seq=next(e['protein']['sequence'] for e in doc['sequences'] if 'protein' in e)
        checks.append(dict(id=row.id,input_sequence_matches_assay=seq==row.sequence,source=str(path)))
    if not all(r['input_sequence_matches_assay'] for r in checks):raise ValueError('Assay/model sequence mismatch')
    tsv(out/'tables/panel_sequence_checks.tsv',checks)
    data.drop(columns=[c for c in data if c.startswith('rep_') or c=='sequence']).to_csv(out/'tables/matched_assay_geometry.tsv',sep='\t',index=False)
    correlations=[];models=[];predictions=[];rng=np.random.default_rng(20261103)
    for label,frame in [('48_homologs',data[data.id!='SfaB'].reset_index(drop=True)),('49_including_SfaB',data.reset_index(drop=True))]:
        y=frame.titre_ugml.to_numpy();target=np.log10(y+.01)
        for feature in ['weighted_angle_deg','harmonic_distance_A','near_attack_fraction','smooth_joint_support']:
            x=frame[feature].to_numpy();rho,p=stats.spearmanr(x,y)
            bootstrap=[]
            for _ in range(2000):
                ix=rng.integers(0,len(y),len(y))
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore');value=stats.spearmanr(x[ix],y[ix]).statistic
                if np.isfinite(value):bootstrap.append(value)
            correlations.append(dict(panel=label,feature=feature,n=len(y),spearman_rho=float(rho),nominal_p=float(p),
                ci95_low=float(np.quantile(bootstrap,.025)),ci95_high=float(np.quantile(bootstrap,.975))))
        configurations={'mean_baseline':[], 'angle_only':['weighted_angle_deg'],'distance_only':['log_distance'],'angle_plus_distance':['weighted_angle_deg','log_distance']}
        for name,columns in configurations.items():
            pred=np.zeros(len(y))
            for i in range(len(y)):
                train=np.arange(len(y))!=i
                if not columns:pred[i]=target[train].mean()
                else:
                    model=make_pipeline(StandardScaler(),Ridge(alpha=1.0));model.fit(frame.loc[train,columns],target[train]);pred[i]=model.predict(frame.loc[[i],columns])[0]
            rho,p=stats.spearmanr(y,pred)
            models.append(dict(panel=label,model=name,n=len(y),loocv_rmse=float(np.sqrt(mean_squared_error(target,pred))),
                loocv_r2=float(r2_score(target,pred)),loocv_spearman=float(rho) if columns else None,nominal_p=float(p) if columns else None,
                strong_producer_auroc=float(roc_auc_score(y>=.5,pred)) if columns else None))
            predictions.extend(dict(panel=label,model=name,id=frame.iloc[i].id,observed_titre_ugml=y[i],observed_log10=target[i],predicted_log10=pred[i]) for i in range(len(y)))
    corr=pd.DataFrame(correlations)
    for label in corr.panel.unique():
        ix=corr.index[corr.panel==label];p=corr.loc[ix,'nominal_p'].to_numpy();order=np.argsort(p)
        q=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1];values=np.empty_like(q);values[order]=np.minimum(q,1);corr.loc[ix,'BH_q']=values
    corr.to_csv(out/'tables/feature_correlations.tsv',sep='\t',index=False)
    tsv(out/'tables/cross_validation.tsv',models);tsv(out/'tables/held_out_predictions.tsv',predictions)
    js(out/'tables/panel_diagnostics.json',dict(n_homologs=48,n_with_SfaB_control=49,excluded_blank='BL21',
        response='N66L1 product titre in micrograms/mL; not percent molar yield',
        zero_mean_titres=int((data.titre_ugml==0).sum()),strong_producers=data.loc[data.titre_ugml>=.5,'id'].tolist(),
        titre_range=[float(data.titre_ugml.min()),float(data.titre_ugml.max())],
        shapiro_log_response_p=float(stats.shapiro(np.log10(data.titre_ugml+.01)).pvalue),
        model_response='log10(titre + 0.01); pseudocount and ridge alpha=1 fixed before analysis',
        pose_unit='50 stochastic predictions per enzyme, not 50 independent experiments',
        bootstrap_unit='enzyme',bootstrap_replicates=2000,model_selection_on_outcomes=False))
    return data,corr,pd.DataFrame(models),pd.DataFrame(predictions)


def savefig(fig,out,name):
    for ext in ['png','svg','pdf']:fig.savefig(out/'figures'/(name+'.'+ext),dpi=300,bbox_inches='tight')
    plt.close(fig)


def figures(out,current,refs,panel,predictions):
    plt.rcParams.update({'font.size':9,'svg.fonttype':'none','pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(10.3,4.2),layout='constrained')
    for ax in axes:
        ax.axhline(150,ls='--',lw=.7,c='#a6b2ba');ax.axvline(4,ls='--',lw=.7,c='#a6b2ba')
        ax.set(xlabel='Harmonic attack distance (Å)',ylabel='Proximity-weighted attack angle (°)',ylim=(70,185));ax.spines[['right','top']].set_visible(False)
    axes[0].scatter(current.harmonic_distance_A,current.weighted_angle_deg,s=22,c='#167b92',alpha=.65)
    axes[0].set_title(f'A  Current C5/C5 candidates (n={len(current)})',loc='left',weight='bold')
    axes[1].scatter(panel.harmonic_distance_A,panel.weighted_angle_deg,s=18,c='#c2cbd0',alpha=.7,label='AS1–AS48 / SfaB panel')
    colors=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00']
    offsets=[(-4,-14),(5,7),(4,8),(4,-14),(4,8)]
    for (_,row),color,offset in zip(refs.iterrows(),colors,offsets):
        axes[1].scatter(row.harmonic_distance_A,row.weighted_angle_deg,s=100,marker='D' if row.enzyme=='PylC' else '*',c=color,edgecolor='#183348',lw=.5,zorder=4)
        axes[1].annotate(row.enzyme,(row.harmonic_distance_A,row.weighted_angle_deg),xytext=offset,textcoords='offset points',fontsize=9)
    axes[1].set_title('B  C6/C6 reference: all five tested enzymes',loc='left',weight='bold')
    for ax in axes:ax.set_xlim(left=2)
    fig.supxlabel('Targets are separate panels. PylC is not an interpretable negative control; historical reference protocols vary.',fontsize=8)
    savefig(fig,out,'attack_angle_vs_distance_with_five_controls')
    fig,axes=plt.subplots(1,2,figsize=(9.8,4),layout='constrained');f=panel[panel.id!='SfaB']
    im=axes[0].scatter(f.harmonic_distance_A,f.weighted_angle_deg,c=f.titre_ugml,cmap='viridis',norm=LogNorm(vmin=f.titre_ugml.min(),vmax=f.titre_ugml.max()),s=35,edgecolor='#526574',lw=.3)
    offsets={'AS9':(13,5),'AS17':(-12,12),'AS23':(9,-13)}
    for _,r in f[f.id.isin(offsets)].iterrows():axes[0].annotate(r.id,(r.harmonic_distance_A,r.weighted_angle_deg),xytext=offsets[r.id],textcoords='offset points',weight='bold',fontsize=8)
    axes[0].set(xlabel='Harmonic attack distance (Å)',ylabel='Weighted attack angle (°)',title='A  48 homologs; color = measured titre')
    fig.colorbar(im,ax=axes[0],label='N66L1 titre (µg/mL), log color scale',fraction=.05)
    for model,color in [('angle_only','#0072B2'),('angle_plus_distance','#D55E00')]:
        g=predictions[(predictions.panel=='48_homologs')&(predictions.model==model)]
        axes[1].scatter(g.observed_log10,g.predicted_log10,s=26,c=color,alpha=.7,label=model.replace('_',' '))
    axes[1].plot([-2,1.1],[-2,1.1],c='#a6b2ba',lw=.8,ls='--')
    axes[1].set(xlabel='Observed log10(titre + 0.01)',ylabel='Held-out prediction',title='B  Leave-one-enzyme-out prediction');axes[1].legend(frameon=False,fontsize=8)
    for ax in axes:ax.spines[['right','top']].set_visible(False)
    savefig(fig,out,'matched_panel_angle_distance_titre')


def markdown_table(frame):return frame.to_markdown(index=False,floatfmt='.3f').replace('nan','—')


def build(campaign,out):
    out.mkdir(parents=True,exist_ok=True)
    for name in ['tables','figures']:(out/name).mkdir(exist_ok=True)
    counts=mapping_tables(campaign,out)
    refs,refposes=reference_geometry();refs.to_csv(out/'tables/five_reference_enzymes.tsv',sep='\t',index=False);refposes.to_csv(out/'tables/five_reference_pose_geometry.tsv',sep='\t',index=False)
    panel,corr,cv,pred=assay_analysis(out)
    run=campaign/'lambda_fleet';manifest=json.loads((run/'manifest.json').read_text());current=[];source_states={}
    for record in manifest['inputs']:
        first=stage(run,record,1)
        if first:
            current.append(dict(accession=record['accession'],family=record['family'],**features([p['distance_A'] for p in first],[p['angle_deg'] for p in first])))
            path=run/'results'/record['accession']/'stage1.json';source_states[str(path)]=sha(path)
    current=pd.DataFrame(current);current.to_csv(out/'tables/c5_geometry_snapshot.tsv',sep='\t',index=False)
    figures(out,current,refs,panel,pred)
    primary=corr[corr.panel=='48_homologs'];validation=cv[cv.panel=='48_homologs']
    angle=primary[primary.feature=='weighted_angle_deg'].iloc[0];joint=validation[validation.model=='angle_plus_distance'].iloc[0];alone=validation[validation.model=='angle_only'].iloc[0]
    table=counts[['reaction','native_equation','raw_accessions','exact_sequences','clusters95','cofold_sequences','native_annotation_accessions']].copy()
    table['native_equation']=table.native_equation.map(lambda s:re.sub(r'\[0\]|\(1\)\s*','',s))
    refdisplay=refs[['enzyme','accession','weighted_angle_deg','harmonic_distance_A','close_poses']]
    corrdisplay=primary[['feature','spearman_rho','ci95_low','ci95_high','nominal_p','BH_q']]
    cvdisplay=validation[['model','loocv_rmse','loocv_r2','loocv_spearman','strong_producer_auroc']]
    now=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    text=f'''# Enzyme provenance, retained homologs and geometry–activity calibration

_Draft for colleague review · {now} · Current C5/C5 discovery plus a separate C6/C6 experimental reference_

## Main findings

The current glutarate–cadaverine discovery retains **570 modeling sequences** from 694 candidate identifiers and 574 pooled 95%-identity clusters. Every modeling sequence can now be traced to its retrieved native-reaction analogue and the annotation or homology evidence supporting that link. The full cohort is being cofolded; novelty relative to prior work does not control modeling admission.[^mapping]

All five enzymes tested by Josh's group—DdaG, SfaB, SuCphA1, McbA and PylC—are highlighted in Figure 1B using their available **adipate/HMDA reference ensembles**. Figure 1A shows the current C5/C5 candidates separately. Historical C6 models are not C5 predictions. PylC's construct failed its native-activity control, so it is not treated as a valid negative enzyme.[^five][^paper]

For the requested AS1–AS48 calibration, angle and distance show **limited evidence of quantitative titre prediction**. The proximity-weighted angle has Spearman ρ={angle.spearman_rho:.3f} (95% bootstrap CI {angle.ci95_low:.3f} to {angle.ci95_high:.3f}; nominal p={angle.nominal_p:.3f}). The joint angle/distance model has held-out R²={joint.loocv_r2:.3f}, versus {alone.loocv_r2:.3f} for angle alone. These results do not validate a yield predictor for the new C5/C5 pair.[^assay]

## 1. Track the source reaction for every enzyme and homolog

The sequence-to-reaction mapping is an evidence graph, not a claim that every homolog has experimentally established native specificity. The [full provenance table](tables/enzyme_reaction_provenance.tsv) preserves every source edge, including multiple edges for one sequence. It includes the retrieved ModelSEED equation, UniProt native catalytic statements, evidence route, alignment reference, identity/coverage, exact-sequence representative and 95% cluster representative. The [cofold-to-reaction table](tables/cofold_sequence_reaction_links.tsv) links each actual modeled sequence to its source reaction(s), distinguishing its own source assignment from the assignment of a cluster member.[^mapping]

```mermaid
flowchart LR
    A["New C5/C5 queries"] --> B["100 retrieved amidations"]
    B --> C["10 chemical analogues"]
    C --> D["Fresh reaction-scoped protein mapping"]
    D --> E["694 identifiers → 574 clusters at 95%"]
    E --> F["570 sequences → new C5/C5 cofolds"]
    G["AS1–AS48 measured C6/C6 titres"] --> H["Matched C6 geometry calibration"]
    H -. "Reference interpretation only" .-> F
```

**Native-reaction annotation** means the sequence has an annotation scoped to the corresponding Rhea reaction. **Inferred or related** means EC/name/homology evidence, without an established exact assignment to the retrieved equation. Neither category establishes activity on glutarate plus cadaverine. The glycylglycine branch is specifically exploratory: its related L-amino-acid ligases must not be renamed experimentally established glycylglycine synthases.[^mapping]

## 2. How many homologs remain at 95% identity?

The table reports the clustering actually used in this campaign: identical sequences are collapsed, followed by greedy 95% sequence-identity clustering with at least 90% aligned coverage of both sequences. Clustering is pooled across the candidate set. “Native annotated” is an accession count before clustering, not the number of experimentally proven target catalysts.[^mapping]

{markdown_table(table)}

The nine mapped native-reaction groups reconcile to 574 clusters and 570 cofold sequences. Five fragment clusters in the AsbA group are excluded (26 → 21); one additional curated L-amino-acid-ligase reference is protected after clustering (10 → 11). The targeted cadaverine query also supports two proteins already represented in the glutamyl-amine group and is not counted as an extra native reaction. The phosphinothricin-tripeptide equation (`rxn13979`) supplies no justified standalone free-acid/diamine catalyst: three carrier-dependent context records are retained separately.[^mapping]

The equations and all per-reaction counts are in [reaction_95_counts.tsv](tables/reaction_95_counts.tsv). This gives the number of homolog representatives to consider cofolding per reaction without losing source attribution or counting the same substrate-query support twice.

## 3. Include the five tested enzymes in angle versus distance

![Current C5 candidate geometry and separate C6 reference panel with all five tested enzymes](figures/attack_angle_vs_distance_with_five_controls.png)

**Figure 1.** Each point summarizes 50 predicted poses. Panel A is the current C5/C5 snapshot ({len(current)} verified stage-1 ensembles); panel B is the matched C6/C6 reference space, with all five tested enzymes highlighted. Distances and angles for the five references were recomputed directly from their retained coordinate ensembles using the same formulas as the assay-panel analysis. The 4 Å and 150° lines are geometric guides, not validated activity decision boundaries. Reference ensembles have differing historical protonation, metal and numerical protocols; they are illustrative controls rather than a calibrated cross-family ranking.[^five]

{markdown_table(refdisplay)}

DdaG and SfaB have reported nylon-diad activity; SuCphA1 and McbA had no detectable products under the reported assay conditions. PylC remains uninterpretable because the soluble construct failed the native-activity control. These controls demonstrate why a single geometric score should not be assumed to transfer quantitatively across enzyme families.[^paper]

**C5 reference coverage:** the five reference enzymes do not yet have verified stage-1 C5 ensembles in the current snapshot. They are deliberately not placed on the current C5 axes using C6 coordinates. If the desired figure is a single, fully matched C5 comparison containing all five controls, fresh C5 cofolds for the missing references are an additional work item; this draft supplies the available reference comparison explicitly.

## 4. Does angle plus distance correlate with the homolog titres?

The supplied file contains **48 sequenced homologs (AS1–AS48), one SfaB control and one BL21 blank**. The primary analysis uses the 48 homologs; a sensitivity analysis includes SfaB (n=49). All 48 homolog input sequences match the assay sequences exactly. The measured outcome is **N66L1 product titre in µg/mL for adipate/HMDA**, not percent molar yield. Therefore these rows are not pooled with literature percentage yields or with current C5 predictions.[^assay]

![Matched panel geometry colored by measured titre and held-out prediction comparison](figures/matched_panel_angle_distance_titre.png)

**Figure 2.** Left: angle–distance geometry for the 48 homologs, colored by measured titre; AS9, AS17 and AS23 are the three ≥0.5 µg/mL producers. Right: held-out predictions from angle-only and joint angle/distance models. Each held-out row is one enzyme; its 50 stochastic poses are not independent experimental samples.

### Definitions and statistical approach

For distance dᵢ from an adipate oxygen to the reactive ATP phosphorus and attack angle θᵢ, the reported proximity-weighted angle is θw = Σ(θᵢ/dᵢ) / Σ(1/dᵢ). Absolute proximity is represented separately by the harmonic distance dH = 50 / Σ(1/dᵢ). A high θw can occur even if every pose is distant, so it cannot replace dH or the count of close poses.

The prespecified strict joint fraction is the proportion with d < 4 Å and θ > 150°. A smooth joint descriptor uses sigmoid((3.5−d)/0.5) × exp[−0.5((θ−180)/30)²], averaged over poses. No threshold, pseudocount or ridge parameter was optimized on these outcomes. Rank correlations use enzyme-level bootstrap confidence intervals (2,000 resamples) and Benjamini–Hochberg adjustment across the four feature tests. Because titres are highly skewed and have only three strong producers, rank-based effect estimates and held-out performance are emphasized rather than a normal-response significance claim.

### Direct feature–titre associations: 48 homologs

{markdown_table(corrdisplay)}

### Joint angle/distance: leave-one-enzyme-out prediction

The response is log10(titre + 0.01), with 0.01 µg/mL fixed in advance. Predictors are standardized inside each training fold; ridge alpha is fixed at 1. The joint model uses θw and log(dH). The mean-only model supplies an honest training-fold-mean RMSE baseline. Its rank correlation and AUROC are omitted because leave-one-out means vary mechanically with the excluded outcome. R² uses the conventional overall-response mean denominator; negative values do not imply negative biological yields.

{markdown_table(cvdisplay)}

The [49-enzyme sensitivity results](tables/cross_validation.tsv) and [all feature correlations](tables/feature_correlations.tsv) retain the SfaB-control analysis. Complete observed and held-out values are in [held_out_predictions.tsv](tables/held_out_predictions.tsv). The descriptive associations and any AUROC values are exploratory: only three strong producers are available, homologs can remain dependent, and leave-one-enzyme-out evaluation does not establish generalization to a new enzyme family or substrate pair. High geometry scores identify candidates for testing; these data do not support reliable quantitative yield prediction.

## Review comments and next decisions

- Confirm whether Figure 1's separate substrate-matched reference panels meet the intended comparison, or whether all five controls should receive new C5 cofolds for one matched C5 panel.
- Decide whether homolog sampling should cover all 570 retained sequences (the current queue) or a prospectively declared, reaction-balanced subset for a subsequent experimental round. Do not retrospectively select a favorable assay subset to improve correlation.
- Confirm any assay-specific detection limits before replacing very small measured titres with censored values. This draft retains the supplied means and does not treat reconstructed literature replicates as new observations.
- Keep the C6 calibration and C5 discovery conclusions separate until matched C5 activity measurements are available.

## Source and data references

[^mapping]: Frozen current-campaign mapping: `mapping/reaction_candidates.tsv`, `cluster_membership.tsv`, `fold_candidates.tsv` and reaction-selection tables under `results/glutarate_cadaverine_fresh_20261102/`. Portable exports are included in this draft's `tables/` directory; exact source paths and SHA256 hashes are recorded in `manifest.json`.
[^five]: Retained ensembles under `results/five_tested_enzymes_cofolds/ensembles/` and their protocol notes. Recomputed per-pose reference measurements are in [five_reference_pose_geometry.tsv](tables/five_reference_pose_geometry.tsv); enzyme summaries and protocol qualifications are in [five_reference_enzymes.tsv](tables/five_reference_enzymes.tsv).
[^assay]: Local source: `Enzyme_agent/exp_data/amide_synthetase/assay_data/panel_activity.csv`; matched geometry: `structural_proxies/tables/continuous_pose_geometry.csv`. [Matched enzyme-level dataset](tables/matched_assay_geometry.tsv), [sequence checks](tables/panel_sequence_checks.tsv), and [diagnostics](tables/panel_diagnostics.json) accompany this draft. This is the AS1–AS48 panel confirmed by the requester.
[^paper]: Qian et al., *Improved nylon polymerization using amide diads*, Green Chemistry (2026), [doi:10.1039/D6GC00885B](https://doi.org/10.1039/D6GC00885B). The paper is used for reference-enzyme assay interpretation, not as a source of new C5 measurements for the homolog panel.
'''
    target=out/'DRAFT.md'
    if target.exists():target=out/'DRAFT.generated.md'
    target.write_text(text)
    sources=[campaign/'mapping/reaction_candidates.tsv',campaign/'mapping/cluster_membership.tsv',campaign/'mapping/fold_candidates.tsv',
        campaign/'reactions/selected_reactions.tsv',SOURCE/'assay_data/panel_activity.csv',SOURCE/'structural_proxies/tables/continuous_pose_geometry.csv',
        PROJECT/'results/five_tested_enzymes_cofolds/manifest.json',Path(__file__)]
    js(out/'manifest.json',dict(status='reference_draft',created_utc=now,report=target.name,
        source_sha256={str(p):sha(p) for p in sources},c5_snapshot_stage_files=source_states,
        outputs_sha256={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'},
        target_matching='C5 discovery and C6 assay calibration remain separate',experimental_unit='enzyme',
        no_running_gpu_jobs_modified=True))
    print('REPORT',target);print('COUNTS',table.to_json(orient='records'))
    print('CORRELATIONS',primary.to_json(orient='records'));print('CV',validation.to_json(orient='records'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();build(a.campaign.resolve(),a.output.resolve())


if __name__=='__main__':main()
