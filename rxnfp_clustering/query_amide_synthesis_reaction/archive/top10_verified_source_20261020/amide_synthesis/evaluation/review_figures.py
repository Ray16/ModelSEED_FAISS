"""Standalone figures and an explicit alignment matrix for the selected panel."""

from amide_synthesis.infrastructure.paths import PROJECT
from amide_synthesis.infrastructure.review import artifact, relative, REVIEW, require_unfrozen_review

from pathlib import Path
import sys,json
import pandas as pd
import numpy as np
from Bio.Align import PairwiseAligner, substitution_matrices
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
RUN=REVIEW
sys.path.insert(0,str(PROJECT.parents[0]/'_lib'))
import _style
import matplotlib.pyplot as plt

def save(fig,name):
    for ext in ['png','pdf']:fig.savefig(artifact(name+'.'+ext),dpi=300,bbox_inches='tight')
    plt.close(fig)

def main():
    data=pd.read_csv(artifact('selected50.tsv'),sep='\t');cache=artifact('selected50_global_identity.tsv')
    if cache.exists() and list(pd.read_csv(cache,sep='\t',index_col=0).index)==data.accession.tolist():
        mat=pd.read_csv(cache,sep='\t',index_col=0).to_numpy()
    else:
        aligner=PairwiseAligner();aligner.mode='global';aligner.substitution_matrix=substitution_matrices.load('BLOSUM62');aligner.open_gap_score=-10;aligner.extend_gap_score=-.5
        mat=np.eye(len(data));seqs=data.sequence.tolist()
        for i,a in enumerate(seqs):
            for j in range(i):
                al=aligner.align(a,seqs[j])[0];ct=al.counts();value=ct.identities/al.shape[1]
                mat[i,j]=mat[j,i]=value
            print('aligned row',i+1,flush=True)
        pd.DataFrame(mat,index=data.accession,columns=data.accession).to_csv(cache,sep='\t')
    # Cluster within each reaction, preserving the five reaction blocks for inspection.
    order=[]
    for r,g in data.groupby('reaction',sort=False):
        ix=g.index.to_numpy();z=linkage(squareform(np.clip(1-mat[np.ix_(ix,ix)],0,1),checks=False),method='average')
        order.extend(ix[leaves_list(z)])
    labels=[data.iloc[i].reaction+' | '+data.iloc[i].accession for i in order]
    fig,ax=plt.subplots(figsize=(28,27));im=ax.imshow(mat[np.ix_(order,order)],vmin=0,vmax=1,cmap='viridis')
    ax.set_xticks(range(50),labels,rotation=90);ax.set_yticks(range(50),labels)
    for x in [9.5,19.5,29.5,39.5]:ax.axvline(x,color='white',lw=2);ax.axhline(x,color='white',lw=2)
    ax.set_title('Global protein sequence identity; clustered within reaction groups')
    fig.colorbar(im,ax=ax,shrink=.7,label='Identical residues / global alignment columns');save(fig,'selected50_sequence_identity')
    counts=pd.read_csv(artifact('balanced_top5_reactions.tsv'),sep='\t')
    fig,ax=plt.subplots(figsize=(14,8));x=np.arange(5);w=.2
    for i,(col,label) in enumerate([('mapped_accessions','Raw mapped IDs'),('supported_accessions','Supported full proteins'),('clusters95','95% identity clusters')]):
        ax.bar(x+(i-1)*w,counts[col],width=w,label=label)
    ax.set_xticks(x,counts.rxn_id);ax.set_ylabel('Protein count');ax.legend();save(fig,'mapping_counts')
    cv=pd.read_csv(artifact('panel_leave_one_enzyme_out_cv.tsv'),sep='\t')
    part=cv[cv.model.isin(['angle_only','angle_plus_distance'])]
    fig,axes=plt.subplots(1,2,figsize=(16,7))
    labs=['Angle','Angle + distance']
    axes[0].bar(labs,part.oof_AUROC_strong_producer,color=['#0072B2','#D55E00']);axes[0].set_ylim(0,1);axes[0].axhline(.5,color='black',ls='--');axes[0].set_ylabel('Held-out AUROC (3 strong producers)')
    axes[1].bar(labs,part.oof_R2,color=['#0072B2','#D55E00']);axes[1].axhline(0,color='black',ls='--');axes[1].set_ylabel('Held-out R-squared for log titre');save(fig,'yield_validation')
    (artifact('alignment_method.json')).write_text(json.dumps(dict(aligner='Biopython PairwiseAligner',mode='global',substitution='BLOSUM62',open_gap=-10,extend_gap=-.5,identity='exact residue matches divided by all alignment columns including gaps',display='average linkage within each of the five reaction groups',n_sequences=len(data)),indent=2))

if __name__=='__main__':main()
