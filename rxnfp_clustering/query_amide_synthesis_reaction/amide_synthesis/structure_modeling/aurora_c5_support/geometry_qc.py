from ligand_qc import validate_ligands
"""Aurora XPU pilot: one protein/config per MPI rank, two poses each."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback




PYTHON='/lus/flare/projects/FRAME-IDP/rzhu/Software/envs/boltz-2026.1/bin/python'
THREADS=['OMP_NUM_THREADS','OMP_THREAD_LIMIT','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(path,obj):
    temp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    temp.write_text(json.dumps(obj,indent=2)+'\n');temp.replace(path)


def angle(a,b,c):
    u=[a.x-b.x,a.y-b.y,a.z-b.z];v=[c.x-b.x,c.y-b.y,c.z-b.z]
    den=math.sqrt(sum(x*x for x in u)*sum(x*x for x in v))
    if den==0:raise ValueError('Degenerate angle')
    return math.degrees(math.acos(max(-1,min(1,sum(x*y for x,y in zip(u,v))/den))))


def step2_reactive(model):
    nitrogen=None;acyl=None;carbonyl=None
    for chain in model:
        heavy=[a for r in chain for a in r if a.element.name!='H']
        nc=sum(a.element.name=='C' for a in heavy);nn=sum(a.element.name=='N' for a in heavy)
        if len(heavy)==7 and nc==5 and nn==2:nitrogen=[a for a in heavy if a.element.name=='N']
        if any(a.element.name=='P' for a in heavy) and ((nc==5 and nn==0) or nc==15):
            ps=[a for a in heavy if a.element.name=='P'];ox=[a for a in heavy if a.element.name=='O']
            bridges=[o for o in ox if any(o.pos.dist(p.pos)<=1.9 for p in ps)]
            for carbon in [a for a in heavy if a.element.name=='C']:
                neigh=[o for o in ox if o.pos.dist(carbon.pos)<=1.8]
                if len(neigh)>=2 and any(carbon.pos.dist(o.pos)<=1.8 for o in bridges):
                    term=[o for o in neigh if not any(o.pos.dist(p.pos)<=1.9 for p in ps)]
                    if term:acyl,carbonyl=carbon,term[0];break
    if nitrogen is None or acyl is None:raise ValueError('Missing reactive acyl intermediate/cadaverine')
    chosen=min(nitrogen,key=lambda n:n.pos.dist(acyl.pos))
    return chosen.pos.dist(acyl.pos),angle(chosen.pos,acyl.pos,carbonyl.pos)


def inspect_outputs(item,config,folder):
    import gemmi
    prediction=folder/('boltz_results_'+item['name'])/'predictions'/item['name']
    expected={item['name']+'_model_'+str(i)+'.pdb' for i in range(50)}
    actual={p.name for p in prediction.glob(item['name']+'_model_*.pdb')}
    if actual!=expected:raise ValueError('Expected 50 exact model slots, got '+repr(sorted(actual)))
    protein=next(x['protein'] for x in config['sequences'] if 'protein' in x)
    mg_expected=sum(x.get('ligand',{}).get('ccd')=='MG' for x in config['sequences'])
    poses=[]
    for filename in sorted(expected):
        pocket_info={}
        path=prediction/filename;confidence=path.with_name('confidence_'+path.stem+'.json')
        json.loads(confidence.read_text());model=gemmi.read_structure(str(path))[0]
        validate_ligands(model,config)
        seq=''.join(gemmi.find_tabulated_residue(r.name).one_letter_code for r in model['A'])
        if hashlib.sha256(seq.encode()).hexdigest()!=item['protein_sequence_sha256']:raise ValueError('Protein identity mismatch')
        if sum(a.element.name=='Mg' for c in model for r in c for a in r)!=mg_expected:raise ValueError('Mg count mismatch')
        if item['stage']==1:
            nuc={a.name:a.pos for r in model['B'] for a in r if a.element.name!='H'}
            if not {'PA','PG','O3A','O3B'}<=set(nuc):raise ValueError('ATP atoms missing')
            acid_id='E' if mg_expected==2 else 'D';acid=[a for r in model[acid_id] for a in r if a.element.name!='H']
            if len(acid)!=9 or sum(a.element.name=='C' for a in acid)!=5 or sum(a.element.name=='O' for a in acid)!=4:
                raise ValueError('Glutarate composition mismatch')
            phosphorus=nuc['PG' if mg_expected==2 else 'PA'];leaving=nuc['O3B' if mg_expected==2 else 'O3A']
            oxygen=min([a for a in acid if a.element.name=='O'],key=lambda a:a.pos.dist(phosphorus))
            distance=oxygen.pos.dist(phosphorus);theta=angle(oxygen.pos,phosphorus,leaving)
        else:
            if mg_expected == 2:
                adp=[a for r in model['D'] for a in r if a.element.name!='H']
                if tuple(sum(a.element.name==e for a in adp) for e in ('C','N','O','P')) != (10,5,10,2): raise ValueError('ADP composition mismatch')
            cadaverine=[a for r in model['C'] for a in r if a.element.name!='H']
            if len(cadaverine)!=7 or sum(a.element.name=='C' for a in cadaverine)!=5 or sum(a.element.name=='N' for a in cadaverine)!=2:
                raise ValueError('cadaverine composition mismatch')
            intermediate=[a for r in model['B'] for a in r if a.element.name!='H']
            signature=tuple(sum(a.element.name==e for a in intermediate) for e in ('C','N','O','P'))
            if signature!=((5,0,7,1) if mg_expected==2 else (15,5,10,1)):
                raise ValueError('Acyl intermediate composition mismatch: '+repr(signature))
            distance,theta=step2_reactive(model)
            pocket=config['constraints'][0]['pocket'];ids={int(x[1]) for x in pocket['contacts']}
            close=min(a.pos.dist(b.pos) for residue in model['A'] if residue.seqid.num in ids
                      for a in residue if a.element.name!='H' for b in cadaverine)
            pocket_info={'pocket_min_distance_A':close,'pocket_satisfied':close<=float(pocket['max_distance'])+0.05}
        poses.append({'model':filename,'pdb_sha256':sha(path),'confidence_sha256':sha(confidence),
                      'distance_A':distance,'angle_deg':theta,**pocket_info})
    return poses
