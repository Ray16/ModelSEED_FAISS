"""Audit the uncapped user mapping, rank chemistry, cluster and select a balanced panel.

Run with the modelseed Python. No external data are silently added to the mapping.
"""
from artifacts import artifact, relative

from pathlib import Path
import sys, csv, json, hashlib, subprocess, collections, os
import re
import numpy as np
import pandas as pd

RUN = Path(__file__).resolve().parents[1]
PROJECT = RUN.parents[1]
ROOT = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT))
from amide_synthesis.substrate_distance import score_reaction, nylon66_architecture_score
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
MM = '/homes/rzhu/miniforge3/envs/seqhom/bin/mmseqs'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fasta(path, seqs):
    path.write_text(''.join(f'>{a}\n{s}\n' for a,s in seqs.items()))

def cluster(path, threshold, stem):
    stem=stem+'_'+sha(path)[:10]
    prefix = path.parent/stem
    if not prefix.with_name(prefix.name+'_cluster.tsv').exists():
        cmd=[MM,'easy-cluster',str(path),str(prefix),str(path.parent/('tmp_'+stem)),
             '--min-seq-id',str(threshold),'-c','0.9','--cov-mode','0',
             '--alignment-mode','3','--seq-id-mode','0','--cluster-mode','2',
             '--threads','4','-s','7.5','-v','1']
        with (path.parent/(stem+'.log')).open('w') as f:
            subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
    return dict((m,r) for r,m in csv.reader(prefix.with_name(prefix.name+'_cluster.tsv').open(),delimiter='\t'))

def main():
    csv.field_size_limit(10**8)
    (RUN/'archive/pre_annotation_clusters').mkdir(exist_ok=True)
    audit_path=artifact('mapping_link_audit.tsv')
    audit=pd.read_csv(audit_path,sep='\t').fillna('') if audit_path.exists() else None
    supported={(r.accession,r.reaction) for r in audit.itertuples() if r.eligible_for_panel} if audit is not None else None
    single_site={(r.accession,r.reaction) for r in audit.itertuples() if not re.search('bifunctional|multifunctional',r.protein_name,re.I)} if audit is not None else None
    links=collections.defaultdict(dict);raw=collections.Counter(); conflicts=[]
    source=PROJECT/'uniprot-msd-links.tsv'
    for line in source.open():
        a,rxns,s=line.rstrip('\n').split('\t'); s=''.join(s.split()).upper().rstrip('*')
        for r in set(rxns.split(',')):
            raw[r]+=1
            if a in links[r] and links[r][a]!=s: conflicts.append((r,a))
            links[r][a]=s
    hits=pd.read_csv(PROJECT/'results/amide_forming_neighbours.tsv',sep='\t').fillna('')
    # Preserve every query's cosine rather than choosing whichever query gives a large number.
    cos=hits.pivot_table(index='rxn_id',columns='query',values='cosine_similarity',aggfunc='max')
    hits=hits.sort_values(['substrate_distance','rxn_id'],ascending=[False,True]).drop_duplicates('rxn_id').copy()
    target=pd.read_csv(PROJECT/'results/queries.tsv',sep='\t').set_index('query').loc['direct_condensation','rxn_smiles']
    reference,_=score_reaction(target)
    rows=[]
    for _,h in hits.iterrows():
        score,feat=score_reaction(h.rxn_smiles)
        row=h.to_dict(); row.pop('query',None);row.pop('rank',None)
        row.update(feat or {});row['chemistry_score']=score
        # Nylon-6,6 is an AA/BB coupling. The historical score gave AB self-
        # condensation more credit than the exact target; keep it for comparison
        # but make the architecture-specific score explicit and target=1.0.
        row['nylon66_score']=nylon66_architecture_score(score,feat)
        row['fraction_of_target_score']=score/reference
        row['mapped_accessions']=len(links[h.rxn_id]); row['mapping_rows']=raw[h.rxn_id]
        row['exact_unique_sequences']=len(set(links[h.rxn_id].values())-{''})
        row['atp_dependent']='ATP[0]' in h.definition and ('ADP[0]' in h.definition or 'AMP[0]' in h.definition)
        row['mechanism']='acyl_phosphate' if row['atp_dependent'] and 'ADP[0]' in h.definition else ('acyl_adenylate' if row['atp_dependent'] else 'other')
        for q,v in cos.loc[h.rxn_id].items():row['cosine_'+q]=v
        rows.append(row)
    ranked=pd.DataFrame(rows).sort_values(['chemistry_score','rxn_id'],ascending=[False,True]).reset_index(drop=True)
    ranked.insert(0,'chemistry_rank',np.arange(1,len(ranked)+1))
    ranked.head(5).to_csv(artifact('strict_top5_legacy.tsv'),sep='\t',index=False)
    ranked=ranked.sort_values(['nylon66_score','rxn_id'],ascending=[False,True]).reset_index(drop=True)
    ranked.insert(0,'nylon66_rank',np.arange(1,len(ranked)+1))
    ranked.to_csv(artifact('all_screened_reactions.tsv'),sep='\t',index=False)
    ranked.head(100).to_csv(artifact('top100_reactions.tsv'),sep='\t',index=False)
    ranked.head(5).to_csv(artifact('strict_top5_reactions.tsv'),sep='\t',index=False)
    pd.read_csv(PROJECT/'results/neighbours_labelled.tsv',sep='\t').groupby('query').head(100).to_csv(artifact('top100_fingerprint_per_query.tsv'),sep='\t',index=False)

    # Mapping provenance does not establish a genome ID. Sequence clustering therefore
    # removes near duplicates irrespective of organism, without claiming identical loci.
    pool=ranked[(ranked.atp_dependent)&(ranked.exact_unique_sequences>=10)]
    counts=[]; chosen=[]; selected=[];used_sequences=set();membership=[]
    for _,rxn in pool.iterrows():
        r=rxn.rxn_id
        if audit is not None and r not in set(audit[audit.in_targeted_audit].reaction):
            raise RuntimeError('Need to extend annotation audit before selecting '+r)
        group_root=RUN/('sequences/clusters' if audit is not None else 'archive/pre_annotation_clusters');group_root.mkdir(exist_ok=True)
        p=group_root/r;p.mkdir(exist_ok=True)
        seqs=links[r];byseq=collections.defaultdict(list)
        for a,s in seqs.items():
            if s:byseq[s].append(a)
        def representative(ids):
            valid=[a for a in ids if supported is None or (a,r) in supported]
            return sorted(valid or ids)[0]
        exact={representative(ids):s for s,ids in byseq.items()}
        fasta(p/'exact_unique.fasta',exact)
        # Exclude noncanonical sequence characters and implausibly short/long fragments
        # for the folded panel, while retaining them in the full counts and membership.
        eligible={a:s for a,s in exact.items() if 100<=len(s)<=1500 and set(s)<=set('ACDEFGHIKLMNPQRSTVWY') and (supported is None or (a,r) in supported) and (single_site is None or (a,r) in single_site)}
        fasta(p/'eligible.fasta',eligible)
        if not eligible:continue
        c95=cluster(p/'eligible.fasta',.95,'identity95')
        reps={a:eligible[a] for a in sorted(set(c95.values()))}
        fasta(p/'nonredundant95.fasta',reps)
        c50=cluster(p/'nonredundant95.fasta',.5,'identity50')
        row={k:rxn[k] for k in ['nylon66_rank','nylon66_score','chemistry_rank','rxn_id','reaction_name','chemistry_score','fraction_of_target_score','definition','mechanism']}
        row.update(mapping_rows=raw[r],mapped_accessions=len(seqs),supported_accessions=sum(supported is None or (a,r) in supported for a in seqs),exact_unique_sequences=len(exact),eligible_exact_sequences=len(eligible),clusters95=len(reps),clusters50=len(set(c50.values())))
        counts.append(row)
        if len(reps)<10:continue
        # Cover 50% identity clusters before taking additional members. Among available
        # representatives, farthest-point sampling on 3-mer Jaccard spreads diversity.
        kmers={a:{s[i:i+3] for i in range(len(s)-2)} for a,s in reps.items()}
        available={a for a,s in reps.items() if s not in used_sequences};picks=[];diversity_used=collections.Counter()
        while len(picks)<10 and available:
            def priority(a):
                ds=min((1-len(kmers[a]&kmers[b])/len(kmers[a]|kmers[b]) for b in picks),default=1)
                return (-diversity_used[c50[a]],ds,-abs(len(reps[a])-np.median([len(s) for s in reps.values()])),a)
            a=max(available,key=priority);picks.append(a);available.remove(a);diversity_used[c50[a]]+=1
        if len(picks)<10:continue
        chosen.append(row)
        for a in picks:
            s=reps[a];used_sequences.add(s)
            selected.append(dict(reaction=r,accession=a,sequence=s,length=len(s),sequence_sha256=hashlib.sha256(s.encode()).hexdigest(),cluster50=c50[a],cluster95=a,mechanism=rxn.mechanism,nylon66_score=rxn.nylon66_score,nylon66_rank=rxn.nylon66_rank,chemistry_score=rxn.chemistry_score,chemistry_rank=rxn.chemistry_rank))
        fasta(p/'selected10.fasta',{a:reps[a] for a in picks})
        for s,ids in byseq.items():
            rep=representative(ids)
            for a in sorted(ids):membership.append(dict(reaction=r,accession=a,length=len(s),annotation_supported=(supported is None or (a,r) in supported),exact_representative=rep,cluster95=c95.get(rep,''),cluster50=c50.get(c95.get(rep,''),''),selected=a in picks,sequence_sha256=hashlib.sha256(s.encode()).hexdigest(),sequence=s))
        if len(chosen)==5:break
    pd.DataFrame(counts).to_csv(artifact('mapping_counts_considered.tsv'),sep='\t',index=False)
    pd.DataFrame(chosen).to_csv(artifact('balanced_top5_reactions.tsv'),sep='\t',index=False)
    pd.DataFrame(membership).to_csv(artifact('all_mapping_members_and_clusters.tsv'),sep='\t',index=False)
    pd.DataFrame(selected).to_csv(artifact('selected50.tsv'),sep='\t',index=False)
    fasta(artifact('selected50.fasta'),{s['accession']:s['sequence'] for s in selected})
    audit=dict(target_chemistry_score=reference,screened_unique_reactions=len(ranked),mapping_sha256=sha(source),source_screen_sha256=sha(PROJECT/'results/amide_forming_neighbours.tsv'),conflicting_accessions=conflicts,selection_policy='highest chemistry score among ATP-dependent screened reactions with >=10 globally distinct 95%-identity representatives; 90% reciprocal coverage; 50%-identity cluster coverage then 3-mer max-min diversity',clustering_software=subprocess.check_output([MM,'version'],text=True).strip(),cluster_parameters={'identity_redundancy':.95,'reciprocal_coverage':.9,'identity_diversity':.5,'threads':4},selected_proteins=len(selected),group_counts=dict(collections.Counter(s['reaction'] for s in selected)),selected_unique_sequences=len(used_sequences),provenance_caveat='Input table merges direct Rhea and GO-assisted links; no per-link provenance or genome IDs are present. No cap or inherited homologs used.')
    (artifact('selection_audit.json')).write_text(json.dumps(audit,indent=2))
    print(pd.DataFrame(chosen).to_string(index=False),flush=True)
    print(json.dumps(audit,indent=2),flush=True)

if __name__=='__main__':main()
