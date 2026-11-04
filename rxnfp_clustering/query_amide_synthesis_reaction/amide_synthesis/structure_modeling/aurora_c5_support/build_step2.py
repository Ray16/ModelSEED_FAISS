"""Build the exact established step-2 pocket from a validated step-1 ensemble."""
import json,hashlib
from pathlib import Path
import gemmi,yaml

GLUTAROYL_AMP='Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)([O-])OC(=O)CCCC(=O)[O-])[C@@H](O)[C@H]1O'
GLUTAROYL_PHOSPHATE='O=C([O-])CCCC(=O)OP(=O)([O-])[O-]'
CADAVERINE='NCCCCC[NH3+]'

def build(config,poses,expected_count=50):
    if len(poses)!=expected_count: raise ValueError('Expected %d verified step-1 poses, got %d'%(expected_count,len(poses)))
    mg=sum(x.get('ligand',{}).get('ccd')=='MG' for x in config['sequences'])
    if mg not in (1,2): raise ValueError('Invalid route Mg count')
    acid_id='E' if mg==2 else 'D'
    best=None
    for path in sorted(map(Path,poses)):
        model=gemmi.read_structure(str(path))[0]
        p=[a.pos for r in model['B'] for a in r if a.element.name=='P']
        acid=[a for r in model[acid_id] for a in r if a.element.name!='H']
        if len(acid)!=9 or sum(a.element.name=='C' for a in acid)!=5 or sum(a.element.name=='O' for a in acid)!=4: raise ValueError('Invalid glutarate in '+str(path))
        oxy=[a.pos for a in acid if a.element.name=='O']
        if not p: raise ValueError('Missing ATP phosphorus in '+str(path))
        d=min(o.dist(x) for o in oxy for x in p)
        if best is None or d<best[0]: best=(d,path,model,acid)
    distance,path,model,acid=best
    scored=[]
    for residue in model['A']:
        atoms=[a.pos for a in residue if a.element.name!='H']
        if atoms:
            d=min(a.dist(b.pos) for a in atoms for b in acid)
            if d<=7.0:scored.append((d,residue.seqid.num))
    contacts=sorted(num for _,num in sorted(scored)[:20])
    if not contacts:raise ValueError('No protein pocket contacts: '+str(path))
    protein=next(x['protein'] for x in config['sequences'] if 'protein' in x)
    ligands=[{'ligand':{'id':'B','smiles':GLUTAROYL_PHOSPHATE if mg==2 else GLUTAROYL_AMP}}, {'ligand':{'id':'C','smiles':CADAVERINE}}]
    ids=iter('DEFG')
    if mg==2:ligands.append({'ligand':{'id':next(ids),'ccd':'ADP'}})
    ligands += [{'ligand':{'id':next(ids),'ccd':'MG'}} for _ in range(mg)]
    doc={'version':1,'sequences':[{'protein':protein}]+ligands,'constraints':[{'pocket':{'binder':'C','contacts':[['A',n] for n in contacts],'max_distance':10.0,'force':True}}]}
    return doc,{'selected_pose':str(path),'selected_pose_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'closest_atp_p_glutarate_o_A':distance,'pocket_residues':contacts}
