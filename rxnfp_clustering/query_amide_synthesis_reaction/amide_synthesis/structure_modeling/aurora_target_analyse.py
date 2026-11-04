"""Validate complete target-campaign output and export geometry without activity claims."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def analyse(root,out):
    manifest=json.loads((root/'manifest.json').read_text())
    records=manifest['inputs']
    if len(records)!=manifest['sequence_count'] or len({r['accession'] for r in records})!=len(records):
        raise ValueError('Unexpected cohort identity/count')
    if out.exists():raise FileExistsError('Preserve the completed analysis snapshot')
    candidates,poses=[],[]
    for record in records:
        stages=[]
        for stage in (1,2):
            path=root/'results'/record['accession']/('stage%d.json'%stage)
            data=json.loads(path.read_text());items=data.get('poses',[])
            if (data.get('status') not in ('verified','verified_with_warnings') or
                data.get('accession')!=record['accession'] or data.get('stage')!=stage or
                data.get('protein_sequence_sha256')!=record['sequence_sha256'] or
                len(items)!=50 or len({p['model'] for p in items})!=50):
                raise ValueError('Invalid verified stage state: '+str(path))
            if ('config_sha256' in record and
                (data.get('source_input_config_sha256')!=record['config_sha256'] or
                 data.get('source_msa_sha256')!=record['msa_sha256'])):
                raise ValueError('Stage output belongs to a different substrate/configuration or alignment: '+str(path))
            for item in items:
                distance,angle=float(item['distance_A']),float(item['angle_deg'])
                if not math.isfinite(distance) or distance<=0 or not math.isfinite(angle) or not 0<=angle<=180:
                    raise ValueError('Invalid geometry: '+str(path))
                pdb=Path(item['path']);confidence=pdb.with_name('confidence_'+pdb.stem+'.json')
                if digest(pdb)!=item['pdb_sha256'] or digest(confidence)!=item['confidence_sha256']:
                    raise ValueError('Verified output changed: '+str(pdb))
                conf=json.loads(confidence.read_text())
                poses.append(dict(accession=record['accession'],family=record['family'],stage=stage,
                    model=item['model'],distance_A=distance,angle_deg=angle,
                    pocket_satisfied=item.get('pocket_satisfied',''),
                    pocket_min_distance_A=item.get('pocket_min_distance_A',''),
                    confidence_score=conf.get('confidence_score',''),iptm=conf.get('iptm',''),
                    ligand_iptm=conf.get('ligand_iptm',''),
                    pair_chains_iptm=json.dumps(conf['pair_chains_iptm'],sort_keys=True) if 'pair_chains_iptm' in conf else ''))
            stages.append(items)
        first,second=stages
        engaged=[float(p['angle_deg']) for p in first if float(p['distance_A'])<4.0]
        valid=[p for p in second if p.get('pocket_satisfied',True)]
        inverse=[1/float(p['distance_A']) for p in first]
        candidates.append(dict(accession=record['accession'],reaction=record['reaction'],family=record['family'],
            mg_count=record['mg_count'],step1_engaged_poses=len(engaged),
            step1_candidate_angle_deg=statistics.mean(engaged) if len(engaged)>=5 else '',
            step1_weighted_angle_deg=sum(float(p['angle_deg'])*w for p,w in zip(first,inverse))/sum(inverse),
            step1_near_attack_fraction=sum(float(p['distance_A'])<4 and float(p['angle_deg'])>150 for p in first)/50,
            step2_valid_poses=len(valid),step2_pocket_violations=50-len(valid),
            step2_amine_reachable_fraction=sum(float(p['distance_A'])<3.5 for p in valid)/len(valid) if valid else '',
            step2_near_attack_fraction=sum(float(p['distance_A'])<3.5 and 90<=float(p['angle_deg'])<=120 for p in valid)/len(valid) if valid else '',
            target_activity_status='structure_based_hypothesis'))
    tmp=out.with_name(out.name+'.building.'+str(os.getpid()));tmp.mkdir(parents=True)
    def table(name,rows):
        with (tmp/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');writer.writeheader();writer.writerows(rows)
    table('candidates.tsv',candidates);table('poses.tsv',poses)
    summary=dict(status='complete',candidates=len(candidates),poses=len(poses),
        manifest_sha256=digest(root/'manifest.json'),
        step1_engaged_candidates=sum(r['step1_engaged_poses']>=5 for r in candidates),
        step2_pocket_violations=sum(r['step2_pocket_violations'] for r in candidates),
        limits=['Geometry and confidence do not establish activity on glutarate/cadaverine.',
                'Step-2 accommodation is conditional on per-enzyme forced pocket restraints.',
                'No old adipate/HMDA predictions or numerical baseline is merged.',
                'Across-family activity rankings are not calibrated.'])
    (tmp/'complete.json').write_text(json.dumps(summary,indent=2)+'\n');tmp.rename(out)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();print(json.dumps(analyse(a.root,a.output),indent=2))


if __name__=='__main__':main()
