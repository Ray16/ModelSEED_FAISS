"""Compare two saved reference ensembles without generating new predictions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from Bio import Align
from Bio.Align import substitution_matrices
import numpy as np

from amide_synthesis.structure_modeling.poses import inspect


def compare(campaign, references, accessions, output):
    manifest=json.loads((campaign/'manifest.json').read_text())
    records={r['accession']:r for r in csv.DictReader(references.open(),delimiter='\t')}
    aligner=Align.PairwiseAligner()
    aligner.mode='global';aligner.substitution_matrix=substitution_matrices.load('BLOSUM62')
    aligner.open_gap_score=-10;aligner.extend_gap_score=-.5
    sequences=[records[a]['sequence'] for a in accessions]
    alignment=aligner.align(*sequences)[0];matches=0;paired=0
    for (i,j),(k,l) in zip(*alignment.aligned):
        matches+=sum(a==b for a,b in zip(sequences[0][i:j],sequences[1][k:l]));paired+=j-i
    result={'sequence_alignment':{'method':'global BLOSUM62, gap open -10, extension -0.5',
        'lengths':dict(zip(accessions,map(len,sequences))),'identical_residues':int(matches),
        'alignment_columns':int(alignment.length),'paired_residues':int(paired),
        'identity_including_gaps':float(matches/alignment.length)},'ensembles':{},'settings':manifest['settings']}
    series={}
    for accession in accessions:
        item=next(r for r in manifest['inputs'] if r['accession']==accession)
        folder=campaign/'outputs'/item['name']
        actual=inspect(item,folder,include_near_attack=False)
        distance=np.array([r['distance_A'] for r in actual['poses']]);angle=np.array([r['angle_deg'] for r in actual['poses']])
        confidence=[json.loads(p.read_text()) for p in sorted(folder.glob('by_seed/*/s42/confidence_*.json'))]
        with Path(item['msa']).open() as handle:msa_rows=sum(1 for _ in handle)-1
        result['ensembles'][accession]={'gene':records[accession]['gene_names'],'organism':records[accession]['organism'],
            'models_checked':actual['n_models'],'weighted_angle_deg':actual['A_deg'],'harmonic_distance_A':actual['Dh_A'],
            'distance_quantiles_min_10_50_90_max':np.quantile(distance,[0,.1,.5,.9,1]).tolist(),
            'angle_quantiles_min_10_50_90_max':np.quantile(angle,[0,.1,.5,.9,1]).tolist(),
            'poses_under4A':int((distance<4).sum()),'poses_under6A':int((distance<6).sum()),
            'median_ptm':float(np.median([r['ptm'] for r in confidence])),
            'median_ligand_iptm':float(np.median([r['ligand_iptm'] for r in confidence])),
            'msa_rows':msa_rows,'mg_count':item['mg_count'],'phosphate':item['phosphate'],
            'reused_from_previous_campaign':accession in manifest.get('reused_models',{})}
        series[accession]=(distance,angle)
    result['interpretation_limits']=['Shared gene names do not establish orthology or equal substrate specificity.',
        'Pose separation is within these 50-sample ensembles, not an independent activity experiment.',
        'Both models are monomeric. Biological assembly and native active-site placement need separate checks.',
        'A reused older ensemble and a freshly generated ensemble are not a matched implementation/independent-seed replication.',
        'High global confidence does not establish productive placement of adipate or completed amidation.']
    output.mkdir(parents=True,exist_ok=True)
    (output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axis=plt.subplots(figsize=(7.5,4.7),layout='constrained')
    for accession,color in zip(accessions,['#0072B2','#D55E00']):
        organism=records[accession]['organism'].split(' (')[0]
        distance,angle=series[accession]
        axis.scatter(distance,angle,s=24,alpha=.6,color=color,label=organism)
        row=result['ensembles'][accession]
        axis.scatter(row['harmonic_distance_A'],row['weighted_angle_deg'],s=100,color=color,marker='D',edgecolor='black')
    axis.set(xlabel='Adipate oxygen–ATP Pγ distance (Å)',ylabel='Attack angle (°)',ylim=(0,185),
             title='Two GlnA2 proteins · 50 poses each')
    axis.spines[['top','right']].set_visible(False);axis.legend(frameon=False)
    fig.supxlabel('Dots: poses. Diamonds: weighted-angle / harmonic-distance summaries. No activity inference.',fontsize=9)
    for extension in ['png','svg','pdf']:fig.savefig(output/('pose_comparison.'+extension),dpi=180)
    plt.close(fig)
    inputs=[campaign/'manifest.json',references,Path(__file__)]
    (output/'manifest.json').write_text(json.dumps({'input_sha256':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        'output_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file() and p.name!='manifest.json'}},indent=2)+'\n')
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True);p.add_argument('--references',type=Path,required=True)
    p.add_argument('--first',required=True);p.add_argument('--second',required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv)
    result=compare(args.campaign,args.references,[args.first,args.second],args.output)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
