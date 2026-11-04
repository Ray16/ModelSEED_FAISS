"""Incrementally summarize verified enzyme ensembles without waiting for the fleet."""
import argparse
import csv
from datetime import datetime,timezone
import html
import json
import math
import os
from pathlib import Path
import statistics
import time


def write_json(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def table(path,rows,fields):
    tmp=path.with_suffix('.tmp')
    with tmp.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,delimiter='\t',extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    tmp.replace(path)


def stage(root,record,number):
    path=root/'results'/record['accession']/('stage'+str(number)+'.json')
    if not path.exists():return None
    state=json.loads(path.read_text());poses=state.get('poses',[])
    if (state.get('status') not in ('verified','verified_with_warnings') or len(poses)!=50 or
        len({p['model'] for p in poses})!=50 or state.get('accession')!=record['accession'] or
        state.get('stage')!=number or state.get('protein_sequence_sha256')!=record['sequence_sha256'] or
        state.get('source_input_config_sha256')!=record['config_sha256'] or
        state.get('source_msa_sha256')!=record['msa_sha256']):
        raise ValueError('Invalid input identity or ensemble state: '+str(path))
    for p in poses:
        if not math.isfinite(float(p['distance_A'])) or float(p['distance_A'])<=0 or not 0<=float(p['angle_deg'])<=180:
            raise ValueError('Invalid geometry in '+str(path))
    return poses


def summarize(root):
    manifest=json.loads((root/'manifest.json').read_text());records=manifest['inputs']
    out=root/'analysis/rolling';out.mkdir(parents=True,exist_ok=True)
    rows=[];poses=[];first_count=second_count=0
    for record in records:
        row=dict(accession=record['accession'],family=record['family'],reaction=record['reaction'],
            stage1_status='pending',stage2_status='pending',step1_engaged_poses='',
            step1_candidate_angle_deg='',step1_weighted_angle_deg='',step1_near_attack_fraction='',
            step2_valid_poses='',step2_pocket_violations='',step2_amine_reachable_fraction='',step2_near_attack_fraction='')
        first,second=stage(root,record,1),stage(root,record,2)
        if first:
            first_count+=1;row['stage1_status']='verified'
            engaged=[float(p['angle_deg']) for p in first if float(p['distance_A'])<4]
            weights=[1/float(p['distance_A']) for p in first]
            row.update(step1_engaged_poses=len(engaged),step1_candidate_angle_deg=statistics.mean(engaged) if len(engaged)>=5 else '',
                step1_weighted_angle_deg=sum(float(p['angle_deg'])*w for p,w in zip(first,weights))/sum(weights),
                step1_near_attack_fraction=sum(float(p['distance_A'])<4 and float(p['angle_deg'])>150 for p in first)/50)
        if second:
            if not first:raise ValueError('Stage 2 exists without verified stage 1')
            second_count+=1;row['stage2_status']='verified'
            valid=[p for p in second if p.get('pocket_satisfied',True)]
            row.update(step2_valid_poses=len(valid),step2_pocket_violations=50-len(valid),
                step2_amine_reachable_fraction=sum(float(p['distance_A'])<3.5 for p in valid)/len(valid) if valid else '',
                step2_near_attack_fraction=sum(float(p['distance_A'])<3.5 and 90<=float(p['angle_deg'])<=120 for p in valid)/len(valid) if valid else '')
        for number,ensemble in [(1,first),(2,second)]:
            if ensemble:
                poses.extend(dict(accession=record['accession'],family=record['family'],stage=number,**p) for p in ensemble)
        rows.append(row)
    fields=list(rows[0]);table(out/'candidates.tsv',rows,fields)
    table(out/'poses.tsv',poses,['accession','family','stage','model','path','distance_A','angle_deg','pocket_satisfied','pocket_min_distance_A'])
    final=(root/'analysis/final/complete.json').exists()
    summary=dict(status='final_analysis_complete' if final else 'provisional',updated_utc=datetime.now(timezone.utc).isoformat(),
        total_candidates=len(records),stage1_verified=first_count,stage2_verified=second_count,
        verified_structures=(first_count+second_count)*50,
        qualification='Geometry and confidence are structural hypotheses; incomplete stages remain missing')
    write_json(out/'summary.json',summary)
    visible=[r for r in rows if r['stage1_status']=='verified']
    visible.sort(key=lambda r:(r['family'],r['accession']))
    columns=['accession','family','stage1_status','stage2_status','step1_engaged_poses','step1_candidate_angle_deg',
        'step1_near_attack_fraction','step2_amine_reachable_fraction','step2_near_attack_fraction']
    header='<tr>'+''.join('<th>'+html.escape(c)+'</th>' for c in columns)+'</tr>'
    body=''.join('<tr>'+''.join('<td>'+html.escape(f'{r[c]:.3f}' if isinstance(r[c],float) else str(r[c]))+'</td>' for c in columns)+'</tr>' for r in visible)
    text=f'''<!doctype html><meta charset="utf-8"><title>Rolling C5 cofold analysis</title>
<style>body{{font:15px/1.5 system-ui;margin:30px;color:#183348}}table{{border-collapse:collapse;font-size:12px}}td,th{{padding:7px;border-bottom:1px solid #ddd;text-align:left}}th{{background:#eef4f6}}</style>
<h1>Rolling glutarate–cadaverine cofold analysis</h1><p><b>{html.escape(summary['status'])}</b> · Updated {summary['updated_utc']}</p>
<p>Stage 1: {first_count}/{len(records)} verified · Both stages: {second_count}/{len(records)} verified · {summary['verified_structures']:,} structures verified.</p>
<p><a href="candidates.tsv">All candidates, including pending stages</a> · <a href="poses.tsv">Verified per-pose geometry</a> · <a href="summary.json">Status JSON</a></p>
<p>Each accepted stage has 50 models and passes producer QC. Stage-1 engagement uses distance &lt;4 Å; its candidate angle needs at least five engaged poses. Stage-2 reachability uses N–acyl-C distance &lt;3.5 Å; the stricter secondary angle interval is 90–120°. Pocket-restraint violations are retained but excluded from stage-2 summaries. Missing stages remain missing. Results are provisional while the queue runs; geometry does not measure enzyme activity or calibrate comparisons across enzyme families.</p>
<table>{header}{body}</table>'''
    tmp=out/'report.tmp';tmp.write_text(text);tmp.replace(out/'report.html')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--watch',action='store_true');p.add_argument('--interval',type=int,default=60)
    a=p.parse_args();root=a.run_dir.resolve();out=root/'analysis/rolling';out.mkdir(parents=True,exist_ok=True)
    deadline=time.time()+14*86400
    while True:
        try:
            result=summarize(root);print(json.dumps(result),flush=True)
            # Refresh the campaign landing report as status and ensembles change.
            try:
                from amide_synthesis.evaluation.fresh_campaign import render
                render(root.parent)
            except Exception as error:write_json(out/'landing_report_error.json',dict(error=repr(error)))
            if not a.watch or result['status']=='final_analysis_complete':return
        except Exception as error:
            write_json(out/'error.json',dict(status='analysis_error',error=repr(error),updated_utc=datetime.now(timezone.utc).isoformat()))
            raise
        if time.time()>deadline:raise RuntimeError('Rolling analysis watch deadline reached')
        time.sleep(max(10,a.interval))


if __name__=='__main__':main()
