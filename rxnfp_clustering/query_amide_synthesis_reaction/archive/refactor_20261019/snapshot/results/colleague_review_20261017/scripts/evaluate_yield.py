"""Exploratory, enzyme-level validation of geometry against measured titre/yield.

All transforms and ridge alpha=1 are fixed, without optimizing on these outcomes.
Poses are repeated predictions, never independent experimental samples.
"""
from artifacts import artifact, relative

from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.special import expit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut

RUN=Path(__file__).resolve().parents[1]
SOURCE=Path('/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent/exp_data/amide_synthetase')

def pose_features(g,dc='distance_A',ac='angle_deg',target=180):
    d=g[dc].to_numpy(float);a=g[ac].to_numpy(float)
    if len(g)!=50 or not np.isfinite(d).all() or not np.isfinite(a).all() or (d<=0).any():
        raise ValueError('Expected 50 valid poses')
    w=1/d
    return dict(n_poses=len(g),angle_deg=float(np.sum(w*a)/w.sum()),harmonic_distance_A=float(len(d)/w.sum()),
                log_harmonic_distance=float(np.log(len(d)/w.sum())),mean_angle_deg=float(a.mean()),
                distance_support=float(expit((3.5-d)/.5).mean()),
                joint_near_attack_fraction=float(((d<4)&(np.abs(a-target)<30)).mean()),
                joint_smooth_support=float((expit((3.5-d)/.5)*np.exp(-.5*((a-target)/30)**2)).mean()))

def evaluate(frame, response, configurations, prefix, groups=None):
    y=np.asarray(response,float);rows=[];preds=[]
    for name,cols in configurations.items():
        X=frame[cols].to_numpy(float)
        splitter=LeaveOneOut() if groups is None else LeaveOneGroupOut()
        splits=splitter.split(X,y,groups)
        pred=np.zeros(len(y))
        for train,test in splits:
            model=make_pipeline(StandardScaler(),Ridge(alpha=1.0))
            model.fit(X[train],y[train]);pred[test]=model.predict(X[test])
        r,p=spearmanr(y,pred)
        row=dict(model=name,n=len(y),features=';'.join(cols),oof_RMSE=float(np.sqrt(mean_squared_error(y,pred))),oof_R2=float(r2_score(y,pred)),oof_spearman=float(r),nominal_p=float(p))
        if 'activity_ugml' in frame:
            binary=frame.activity_ugml>=.5
            row['oof_AUROC_strong_producer']=float(roc_auc_score(binary,pred))
        rows.append(row)
        for i,v in enumerate(pred):preds.append(dict(model=name,row=i,observed=y[i],predicted=v))
    pd.DataFrame(rows).to_csv(artifact(prefix+'_cv.tsv'),sep='\t',index=False)
    pd.DataFrame(preds).to_csv(artifact(prefix+'_oof_predictions.tsv'),sep='\t',index=False)
    return rows

def main():
    geom=SOURCE/'structural_proxies/tables/continuous_pose_geometry.csv'
    panelpath=SOURCE/'assay_data/panel_activity.csv'
    panel=pd.read_csv(panelpath).rename(columns={'activity_N66L1_ugml_avg':'activity_ugml'})
    pg=pd.read_csv(geom);pg=pg[pg.group=='calibration']
    features=pd.DataFrame([dict(id=a,**pose_features(g)) for a,g in pg.groupby('id')])
    data=features.merge(panel[['id','activity_ugml','sequence']],on='id',validate='one_to_one')
    data.to_csv(artifact('panel_geometry_and_titre.tsv'),sep='\t',index=False)
    configs={'angle_only':['angle_deg'],'angle_plus_distance':['angle_deg','log_harmonic_distance'],
             'angle_plus_distance_support':['angle_deg','distance_support'],
             'angle_plus_near_attack_fraction':['angle_deg','joint_near_attack_fraction'],
             'joint_smooth_support':['joint_smooth_support']}
    # Titre has a long right tail and a nonzero measurement floor; 0.01 ug/ml is fixed,
    # not selected for fit. No pose or technical replicate appears as a new row.
    results=evaluate(data,np.log10(data.activity_ugml+.01),configs,'panel_leave_one_enzyme_out')
    heldout=pd.read_csv(artifact('panel_leave_one_enzyme_out_oof_predictions.tsv'),sep='\t')
    baseline=heldout[heldout.model=='angle_only'].sort_values('row').predicted.to_numpy()
    combined=heldout[heldout.model=='angle_plus_distance'].sort_values('row').predicted.to_numpy()
    binary=(data.activity_ugml.to_numpy()>=.5);paired=[]
    rng_auc=np.random.default_rng(20261017)
    for _ in range(3000):
        ix=rng_auc.integers(0,len(data),len(data))
        if len(np.unique(binary[ix]))<2:continue
        paired.append(roc_auc_score(binary[ix],combined[ix])-roc_auc_score(binary[ix],baseline[ix]))
    (artifact('paired_auroc_uncertainty.json')).write_text(json.dumps(dict(delta_AUROC=float(roc_auc_score(binary,combined)-roc_auc_score(binary,baseline)),fixed_oof_prediction_bootstrap95=list(np.quantile(paired,[.025,.975])),n_bootstrap=len(paired),unit='enzyme',limitation='Conditional resampling of fixed held-out predictions; does not repeat model fitting. Only three positive enzymes, so uncertainty remains large.'),indent=2))
    correlations=[]
    rng=np.random.default_rng(20261017)
    for col in ['angle_deg','harmonic_distance_A','distance_support','joint_near_attack_fraction','joint_smooth_support']:
        r,p=spearmanr(data[col],data.activity_ugml);boot=[]
        for _ in range(2000):
            ix=rng.integers(0,len(data),len(data));b=spearmanr(data[col].to_numpy()[ix],data.activity_ugml.to_numpy()[ix]).statistic
            if np.isfinite(b):boot.append(b)
        correlations.append(dict(feature=col,spearman=float(r),nominal_p=float(p),bootstrap95_low=float(np.quantile(boot,.025)),bootstrap95_high=float(np.quantile(boot,.975))))
    pd.DataFrame(correlations).to_csv(artifact('panel_feature_correlations.tsv'),sep='\t',index=False)

    litroot=SOURCE/'literature_cofolding/fig4_amine_matrix'
    s1=pd.read_csv(litroot/'step1_acid/fig4_amine_matrix_step1_nac_models.csv')
    s2=pd.read_csv(litroot/'step2_amine/fig4_amine_matrix_step2_nac_models.csv')
    f1=pd.DataFrame([dict(enzyme=e,acid_code=a,**{'s1_'+k:v for k,v in pose_features(g,'attack_distance','inline_angle').items()}) for (e,a),g in s1.groupby(['enzyme','diacid_code'])])
    f2=pd.DataFrame([dict(enzyme=e,acid_code=a,amine_code=n,**{'s2_'+k:v for k,v in pose_features(g,'attack_distance','bd_angle',107).items()}) for (e,a,n),g in s2.groupby(['enzyme','diacid_code','amine_code'])])
    yieldpath=SOURCE/'literature_activity/fig2c_quantified/diads_fig2c_quantified_yield.csv'
    yields=pd.read_csv(yieldpath);yields=yields[yields.is_mean.astype(str).str.lower().eq('true')].copy()
    # Local source header says Chen, but PDF title page credits Qian; record correction.
    yields['paper']='Qian et al., bioRxiv 2025.02.28.640842, supplied 2025-09-04 version'
    yields['acid_code']=yields.acid_name.map({'adipic':'A','glutaric':'G','succinic':'S'})
    yields['amine_code']=yields.amine_name.map({'cadaverine':'C','hexamethylenediamine':'M','HMD':'M'})
    lit=yields.merge(f1,on=['enzyme','acid_code']).merge(f2,on=['enzyme','acid_code','amine_code'])
    lit.to_csv(artifact('literature_yield_and_geometry.tsv'),sep='\t',index=False)
    conf={'activation_angle':['s1_angle_deg'],
          'activation_angle_plus_distance':['s1_angle_deg','s1_log_harmonic_distance'],
          'two_step_angles':['s1_angle_deg','s2_angle_deg'],
          'two_step_angles_plus_distances':['s1_angle_deg','s1_log_harmonic_distance','s2_angle_deg','s2_log_harmonic_distance'],
          'two_step_smooth_support':['s1_joint_smooth_support','s2_joint_smooth_support']}
    litresults={}
    for label,groups in [('leave_one_acid_out',lit.acid_code),('leave_one_enzyme_out',lit.enzyme),('leave_one_substrate_pair_out',lit.acid_code+lit.amine_code)]:
        litresults[label]=evaluate(lit,np.log10(lit.value+1),conf,'literature_'+label,groups=groups)
    audit=dict(panel_n=len(data),panel_missing_geometry=sorted(set(panel.id)-set(data.id)),panel_strong_producers=int((data.activity_ugml>=.5).sum()),literature_n=len(lit),literature_independent_enzymes=int(lit.enzyme.nunique()),literature_independent_acids=int(lit.acid_code.nunique()),literature_response='digitized mean molar yield (%); not raw observations; inactive enzymes lack matching poses and are excluded',panel_response='log10(titre ug/ml + 0.01)',literature_transform='log10(yield percentage points + 1)',ridge_alpha=1,poses_per_row=50,parameters_fitted_to_outcomes=False,features_examined=len(configs),bootstrap_unit='enzyme',caveats=['Exploratory comparison of several prespecified features; no multiple-testing adjustment or independent prospective validation.','Homologous panel proteins can remain dependent; leave-one-enzyme-out does not establish transfer across enzyme families.','Literature has only two active enzymes; substrate pairs, not poses, are the experimental units.','No pooled fit across panel titre and literature percent yield.','Previously selected score was chosen among 81 settings on the same 48-protein panel; its high training AUROC is not independent validation.','Step 2 uses an activated intermediate and a separate pose ensemble; two-step support is not a measured joint trajectory.'],source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [geom,panelpath,yieldpath]},panel_cv=results,literature_cv=litresults)
    (artifact('yield_analysis_audit.json')).write_text(json.dumps(audit,indent=2))
    print(pd.DataFrame(results).to_string(index=False));print('Literature matched',len(lit));print(pd.DataFrame(litresults['leave_one_enzyme_out']).to_string(index=False))

if __name__=='__main__':main()
