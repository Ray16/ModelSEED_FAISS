"""Rebuild a provenance-preserving mapping from current reviewed UniProt records.

Exact RHEA:n reaction identifiers only. RHEA-COMP:n, ChEBI and generic GO do
not generate links. Homology-supported UniProt reaction annotations are kept
in their own evidence tier. Does not overwrite the supplied table.
"""
from artifacts import artifact, relative

from pathlib import Path
import csv,gzip,json,re,collections,hashlib,datetime,time
import requests
import sys

RUN=Path(__file__).resolve().parents[1];PROJECT=RUN.parents[1];ROOT=PROJECT.parents[1]
sys.path.insert(0,str(PROJECT))
from amide_synthesis.mapping_quality import reaction_statements

def main():
    csv.field_size_limit(10**8)
    cache=RUN/'evidence/uniprot_reviewed_rhea_snapshot.tsv.gz'
    params={'query':'(reviewed:true) AND (rhea:*)','format':'tsv','fields':'accession,reviewed,protein_name,organism_id,sequence,cc_catalytic_activity'}
    url='https://rest.uniprot.org/uniprotkb/stream'
    if not cache.exists():
        print('Downloading reviewed Rhea snapshot',flush=True)
        with requests.get(url,params=params,stream=True,timeout=(30,180)) as r:
            r.raise_for_status()
            (RUN/'evidence/global_mapping_download.json').write_text(json.dumps(dict(url=r.url,headers=dict(r.headers),retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),query=params),indent=2))
            tmp=cache.with_suffix('.partial.gz');n=0
            with gzip.open(tmp,'wb',compresslevel=3) as f:
                for b in r.iter_content(1024*1024):
                    f.write(b);n+=len(b)
                    if n//(25*1024*1024)!=(n-len(b))//(25*1024*1024):print('Downloaded',round(n/1e6),'MB uncompressed',flush=True)
            tmp.replace(cache)
    reverse=collections.defaultdict(set);obsolete=set();reaction_names={}
    files=[]
    for p in sorted((ROOT/'ModelSEEDDatabase/Biochemistry').glob('reaction_*.tsv')):
        files.append(p)
        for d in csv.DictReader(p.open(),delimiter='\t'):
            if d.get('is_obsolete') not in ('0','false','False'):obsolete.add(d['id']);continue
            reaction_names[d['id']]=d['name']
            for group in re.findall(r'(?:^|\|)Rhea: ([^|]+)',d.get('aliases','')):
                for rid in group.split(';'):
                    rid=rid.strip()
                    if re.fullmatch(r'\d+',rid):reverse[rid].add(d['id'])
    old={}
    for line in (PROJECT/'uniprot-msd-links.tsv').open():
        a,rs,s=line.rstrip('\n').split('\t');old[a]=(set(rs.split(',')),s)
    dest=artifact('uniprot-msd-direct-reviewed.tsv');experimental=artifact('uniprot-msd-experimental-reviewed.tsv')
    evidence=artifact('direct_mapping_evidence.tsv.gz')
    stats=collections.Counter();oldpairs=sum(len(rs) for rs,s in old.values());lost=[];retained_by_reaction=collections.Counter();newset=set()
    reaction_counts=collections.defaultdict(collections.Counter)
    with gzip.open(cache,'rt') as src,dest.open('w') as out,experimental.open('w') as exp,gzip.open(evidence,'wt',compresslevel=3) as ev:
        fields=['accession','taxon_id','modelseed_reaction','rhea_reaction','evidence_tier','evidence','protein_name','sequence_sha256','alias_fanout','source']
        w=csv.DictWriter(ev,fields,delimiter='\t');w.writeheader()
        for d in csv.DictReader(src,delimiter='\t'):
            stats['source_proteins']+=1;a=d['Entry'];s=d['Sequence'];mapped=set();expmap=set()
            for statement in reaction_statements(d['Catalytic activity']):
                ridset=[statement['rhea_id']];evtext=statement['evidence'];tier=statement['evidence_tier']
                for rid in ridset:
                    for rxn in sorted(reverse.get(rid,())):
                        mapped.add(rxn);newset.add(rxn)
                        if tier=='experimental_direct':expmap.add(rxn)
                        w.writerow(dict(accession=a,taxon_id=d['Organism (ID)'],modelseed_reaction=rxn,rhea_reaction='RHEA:'+rid,evidence_tier=tier,evidence=evtext,protein_name=d['Protein names'],sequence_sha256=hashlib.sha256(s.encode()).hexdigest(),alias_fanout=len(reverse[rid]),source='uniprot_reaction_xref_to_modelseed_rhea_alias'))
                        stats['evidence_rows']+=1
            if mapped:
                out.write(a+'\t'+','.join(sorted(mapped))+'\t'+s+'\n');stats['mapped_proteins']+=1;stats['mapped_pairs']+=len(mapped)
            if expmap:
                exp.write(a+'\t'+','.join(sorted(expmap))+'\t'+s+'\n');stats['experimental_proteins']+=1;stats['experimental_pairs']+=len(expmap)
            for rxn in mapped:
                reaction_counts[rxn]['direct_accessions']+=1
                reaction_counts[rxn]['experimental_accessions']+=rxn in expmap
            if a in old:
                rs,oldseq=old.pop(a);stats['old_pairs_retained']+=len(rs & mapped);stats['new_pairs_added_existing_accessions']+=len(mapped-rs)
                for rxn in rs & mapped:retained_by_reaction[rxn]+=1
                for rxn in rs-mapped:lost.append((a,rxn,'not_in_current_direct_alias_mapping'))
                stats['old_sequences_changed']+=oldseq!=s
            elif mapped:stats['new_pairs_new_accessions']+=len(mapped)
            if stats['source_proteins']%25000==0:print(dict(stats),flush=True)
    for a,(rs,s) in old.items():
        for rxn in rs:lost.append((a,rxn,'accession_absent_from_current_reviewed_rhea_query'))
    with (artifact('old_links_not_retained.tsv')).open('w') as f:
        w=csv.writer(f,delimiter='\t');w.writerow(['accession','reaction','reason']);w.writerows(lost)
    with (artifact('direct_mapping_reaction_counts.tsv')).open('w') as f:
        w=csv.writer(f,delimiter='\t');w.writerow(['reaction','name','direct_accessions','experimental_accessions','old_pairs_retained'])
        for rxn,c in sorted(reaction_counts.items()):w.writerow([rxn,reaction_names[rxn],c['direct_accessions'],c['experimental_accessions'],retained_by_reaction[rxn]])
    result=dict(stats,old_pairs=oldpairs,old_pairs_not_retained=len(lost),mapped_reactions=len(newset),source_cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),mapping_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),snapshot_query=params,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),validation_scope='Reaction-annotation support; reviewed does not mean every assignment was experimentally measured. Experimental tier requires ECO:0000269 on the specific reaction statement. Reactions without local Rhea aliases remain unresolved; GO is not substituted.',namespace_rule='Only exact Rhea:RHEA:<integer> reaction xrefs; RHEA-COMP identifiers never create reaction links.',source_files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    assert result['old_pairs_retained']+len(lost)==oldpairs
    (artifact('direct_mapping_rebuild_summary.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
