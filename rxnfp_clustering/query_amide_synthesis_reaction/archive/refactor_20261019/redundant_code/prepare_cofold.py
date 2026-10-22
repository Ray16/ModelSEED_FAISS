"""Freeze the audited 50-protein activation campaign; adapt proven fleet code."""
from artifacts import artifact, relative

from pathlib import Path
import csv,json,hashlib,shutil,datetime
import yaml
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

REVIEW=Path(__file__).resolve().parents[1];RUN=REVIEW/'cofold';PROJECT=REVIEW.parents[1]
PREVIOUS=PROJECT/'results/boltz2_activation_20261016/scripts'

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    rows=list(csv.DictReader((artifact('selected50.tsv')).open(),delimiter='\t'))
    audit={(r['accession'],r['reaction']):r for r in csv.DictReader((artifact('mapping_link_audit.tsv')).open(),delimiter='\t')}
    inputs=[];acid='[O-]C(=O)CCCCC(=O)[O-]'
    mol=Chem.MolFromSmiles(acid);assert Chem.GetFormalCharge(mol)==-2 and rdMolDescriptors.CalcMolFormula(mol)=='C6H8O4-2'
    assert len(rows)==50 and len({r['sequence'] for r in rows})==50
    for r in rows:
        a=r['accession'];ev=audit[(a,r['reaction'])]
        assert ev['eligible_for_panel']=='True' and ev['direct_rhea']=='True'
        seq=r['sequence'];h=hashlib.sha1(seq.encode()).hexdigest()[:16]
        old=RUN/'msa'/(h+'.csv');new=RUN/'msa_delta'/(h+'.csv')
        # The initial batched request already covers shared sequences. The second
        # request contains only newly selected sequences, without duplicate requests.
        delta='>'+a+'\n' in (RUN/'msa_delta.fasta').read_text()
        msa=new if delta else old
        name=a+'_2mg';p=RUN/'inputs'/(name+'.yaml')
        sequences=[{'protein':{'id':'A','sequence':seq,'msa':str(msa.resolve())}},
                   {'ligand':{'id':'B','ccd':'ATP'}},{'ligand':{'id':'C','ccd':'MG'}},
                   {'ligand':{'id':'D','ccd':'MG'}},{'ligand':{'id':'E','smiles':acid}}]
        p.write_text(yaml.safe_dump({'version':1,'sequences':sequences},sort_keys=False))
        family={'rxn00851':'Ddl','rxn02010':'MurE_Lys','rxn02011':'MurE_DAP','rxn00646':'GshA','rxn00351':'GshB'}[r['reaction']]
        inputs.append(dict(accession=a,name=name,reaction=r['reaction'],family=family,mechanism_family=family,phosphate='PG',leaving_oxygen='O3B',mg_count=2,enzyme_chains=1,active_sites_modeled=1,sequence_sha256=r['sequence_sha256'],sequence_length=len(seq),msa=str(msa.resolve()),input=str(p.resolve()),input_sha256=digest(p),acid_chain='E',acid_smiles=acid,metal_assignment_certainty='family_structural_transfer',metal_assignment_basis='Two-metal ATP-dependent acyl-phosphate modeling hypothesis, transferred from Ddl/Mur/glutathione-ligase structures; not a titration of each homolog.',annotation_source=str((REVIEW/'evidence/uniprot_mapping_snapshot'/(a+'.json')).resolve()),annotation_sha256=digest(REVIEW/'evidence/uniprot_mapping_snapshot'/(a+'.json'))))
    manifest=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='50 audited original-file mapped proteins, ten per qualifying reaction; one-chain adipate activation complexes; no HMDA or activated intermediate in these inputs',inputs=inputs,settings=dict(mode='exact',seed=42,diffusion_samples=50,max_parallel_samples=41,actual_stock_sample_chunks=[5]*10,recycling_steps=3,sampling_steps=200,num_workers=0,preprocessing_threads=1,output_format='pdb'),ligands={'ATP':'Boltz CCD ATP, same cached representation as historical comparison (formal charge 0 recorded previously)','MG':'Boltz CCD Mg2+; two ions per modeled active site','adipate':'C6H8O4 dianion, formal charge -2'},expected_models=2500,sequence_verification='Exact match to current reviewed UniProt sequence and original mapping; no fragments or explicitly bifunctional proteins selected',interpretation='Geometry measures acid activation at Pgamma. It does not establish amine acceptance, catalytic turnover, expression, full oligomeric assembly or nylon yield.',sources={str(p.resolve()):digest(p) for p in [artifact('selected50.tsv'),artifact('mapping_link_audit.tsv'),PROJECT/'mapping_policy.json']},metal_sources=['https://www.rcsb.org/structure/2DLN','https://www.rcsb.org/structure/4C12','https://www.rcsb.org/structure/1GSH'])
    (RUN/'manifest.json').write_text(json.dumps(manifest,indent=2))
    # Preserve the prior implementation verbatim alongside the adapted producer.
    for name in ['analyze.py','fleet.py','verify_runtime.py']:
        src=(PREVIOUS/name).read_text();(RUN/'evidence'/('previous_'+name)).write_text(src)
        if name=='fleet.py':
            src=src.replace('resumable 17-input','resumable 50-input').replace('amide17_20261016','amide50_20261017').replace('len(rows)==17',"len(rows)==len(manifest['inputs'])")
            src=src.replace('len(cpus)//8','max(0,(len(cpus)-2)//8)')
            src=src.replace("if max(os.getloadavg()) + count*8 > os.cpu_count(): count = 0","if max(os.getloadavg()) + count*8 + 2 > os.cpu_count(): count = 0")
            src=src.replace("preflight = record(status=", "import tempfile\n    with tempfile.TemporaryDirectory(prefix='amide50_preflight_') as t:\n        Path(t,'writecheck').write_text('ok')\n    preflight = record(status=")
            # Every done marker is rechecked against actual output and immutable inputs.
            src=src.replace("if done.exists(): continue", "if done.exists():\n                inspect(item,RUN/'outputs'/name)\n                continue",1)
        if name=='analyze.py':
            src=src.replace('import csv','import hashlib\nimport csv',1)
            src=src.replace("    models = sorted", "    if hashlib.sha256(Path(item['input']).read_bytes()).hexdigest()!=item['input_sha256']:\n        raise ValueError('Input identity changed')\n    models = sorted",1)
            src=src.replace("'mean_angle_deg': float(a.mean())}","'mean_angle_deg': float(a.mean()), 'near_attack_fraction': float(((d<4)&(a>150)).mean())}")
        (RUN/'scripts'/name).write_text(src)
    (RUN/'evidence/implementation_hashes.json').write_text(json.dumps({str(p):digest(p) for p in (RUN/'scripts').glob('*.py')},indent=2))
    print('Frozen 50 audited inputs; expected 2,500 poses; MSA-gated fleet ready')

if __name__=='__main__':main()
