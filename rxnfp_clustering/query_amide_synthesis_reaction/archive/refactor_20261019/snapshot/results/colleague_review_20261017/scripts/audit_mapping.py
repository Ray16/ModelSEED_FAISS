"""Validate mapping links against namespace-aware current UniProt annotations.

Audits all members of the leading eight sufficiently populated ATP reactions,
plus a uniform reservoir sample of 200 mapping PAIRS (not 200 chosen proteins).
Absence of direct annotation is unresolved, not automatically a false positive.
"""
from artifacts import artifact, relative

from pathlib import Path
import csv,json,random,re,concurrent.futures,time,hashlib,collections
import requests
import pandas as pd

RUN=Path(__file__).resolve().parents[1];PROJECT=RUN.parents[1];ROOT=PROJECT.parents[1]
CACHE=RUN/'evidence/uniprot_mapping_snapshot';CACHE.mkdir(exist_ok=True)

def fetch(chunk):
    missing=[a for a in chunk if not (CACHE/(a+'.json')).exists()]
    if not missing:return
    for attempt in range(4):
        try:
            q=' OR '.join('accession:'+a for a in missing)
            r=requests.get('https://rest.uniprot.org/uniprotkb/search',params={'query':q,'format':'json','size':500},timeout=90);r.raise_for_status()
            got=r.json()['results']
            for entry in got:
                a=entry['primaryAccession'];(CACHE/(a+'.json')).write_text(json.dumps(entry))
                for alias in entry.get('secondaryAccessions',[]):
                    if alias in missing:(CACHE/(alias+'.json')).write_text(json.dumps(entry))
            for a in missing:
                if not (CACHE/(a+'.json')).exists():(CACHE/(a+'.json')).write_text(json.dumps({'unavailable':True,'requested':a}))
            return
        except requests.RequestException:
            if attempt==3:raise
            time.sleep(2**attempt)

def main():
    csv.field_size_limit(10**8);rng=random.Random(20261017);sample=[];n=0;links={};rows=0
    for l in (PROJECT/'uniprot-msd-links.tsv').open():
        a,rxns,s=l.rstrip('\n').split('\t');rows+=1;links[a]=(set(rxns.split(',')),s)
        for r in sorted(set(rxns.split(','))):
            n+=1
            if len(sample)<200:sample.append((a,r))
            else:
                j=rng.randrange(n)
                if j<200:sample[j]=(a,r)
    ranked=pd.read_csv(artifact('all_screened_reactions.tsv'),sep='\t')
    targeted=ranked[(ranked.atp_dependent)&(ranked.exact_unique_sequences>=10)].head(8)
    wanted=set(targeted.rxn_id)
    pairs=[(a,r) for a,(rs,s) in links.items() for r in sorted(rs & wanted)]
    allpairs=sorted(set(pairs+sample));accessions=sorted(set(a for a,r in allpairs))
    print('fetching',len(accessions),'proteins;',len(pairs),'targeted pairs;',len(sample),'uniform sampled pairs',flush=True)
    chunks=[accessions[i:i+40] for i in range(0,len(accessions),40)]
    with concurrent.futures.ThreadPoolExecutor(3) as pool:
        for i,_ in enumerate(pool.map(fetch,chunks)):print('batch',i+1,'/',len(chunks),flush=True)
    metadata={}
    for f in sorted((ROOT/'ModelSEEDDatabase/Biochemistry').glob('reaction_*.tsv')):
        for r in csv.DictReader(f.open(),delimiter='\t'):metadata[r['id']]=r
    records=[]
    for a,r in allpairs:
        d=json.loads((CACHE/(a+'.json')).read_text());m=metadata.get(r,{})
        aliases=set(re.findall(r'Rhea: ([^|]+)',m.get('aliases','')))
        expected={s.strip() for a_ in aliases for s in a_.split(';')}
        true=set();comp=set();all_ec=set();reaction_names=[]
        for c in d.get('comments',[]):
            if c.get('commentType')!='CATALYTIC ACTIVITY':continue
            rx=c['reaction'];reaction_names.append(rx.get('name',''))
            if rx.get('ecNumber'):all_ec.add(rx['ecNumber'])
            for ref in rx.get('reactionCrossReferences',[]):
                if re.fullmatch(r'RHEA:\d+',ref['id']):true.add(ref['id'].split(':')[1])
                elif re.fullmatch(r'RHEA-COMP:\d+',ref['id']):comp.add(ref['id'].split(':')[1])
        desc=d.get('proteinDescription',{});name=''
        for key in ['recommendedName','submittedName']:
            descs=desc.get(key,[]);descs=descs if isinstance(descs,list) else [descs]
            for x in descs:
                name=name or x.get('fullName',{}).get('value','')
                all_ec|={x['value'] for x in x.get('ecNumbers',[])}
        exp_ec=set(m.get('ec_numbers','').split('|'))-{'','null'}
        direct=bool(expected & true);collision=bool(expected & comp) and not direct
        ec_support=bool(exp_ec & all_ec)
        noncatalytic=bool(re.search('regulatory subunit|light chain|modifier subunit|non.catalytic',name,re.I))
        seq=links[a][1];current=d.get('sequence',{}).get('value','');same=seq==current
        evidence='unavailable' if d.get('unavailable') else ('noncatalytic_subunit' if noncatalytic else ('direct_rhea' if direct else ('ec_supported' if ec_support else ('rhea_component_collision' if collision else 'unresolved'))))
        fragment=desc.get('flag','')=='Fragment'
        supported=(direct or ec_support) and not noncatalytic and same and not fragment
        records.append(dict(accession=a,primary_accession=d.get('primaryAccession',''),reaction=r,protein_name=name,organism=d.get('organism',{}).get('scientificName',''),taxon_id=d.get('organism',{}).get('taxonId',''),expected_rhea=';'.join(sorted(expected)),observed_rhea=';'.join(sorted(true)),observed_rhea_comp=';'.join(sorted(comp)),observed_ec=';'.join(sorted(all_ec)),direct_rhea=direct,ec_supported=ec_support,component_collision=collision,noncatalytic_subunit=noncatalytic,fragment=fragment,sequence_matches_current=same,evidence=evidence,eligible_for_panel=supported,in_uniform_sample=(a,r) in sample,in_targeted_audit=r in wanted,sequence_sha256=hashlib.sha256(seq.encode()).hexdigest(),catalytic_reactions=' | '.join(reaction_names)))
    df=pd.DataFrame(records);df.to_csv(artifact('mapping_link_audit.tsv'),sep='\t',index=False)
    report=[]
    for r,g in df[df.in_targeted_audit].groupby('reaction'):
        report.append(dict(reaction=r,raw_accessions=len(g),direct_rhea=int(g.direct_rhea.sum()),ec_supported=int(g.ec_supported.sum()),component_collisions=int(g.component_collision.sum()),noncatalytic_subunits=int(g.noncatalytic_subunit.sum()),fragments=int(g.fragment.sum()),sequence_mismatches=int((~g.sequence_matches_current).sum()),supported_for_panel=int(g.eligible_for_panel.sum()),unresolved=int((g.evidence=='unresolved').sum())))
    pd.DataFrame(report).to_csv(artifact('mapping_quality_by_reaction.tsv'),sep='\t',index=False)
    sampledf=df[df.in_uniform_sample]
    doc=dict(mapping_rows=rows,unique_accessions=len(links),unique_pairs=n,unique_reactions=len(set(r for rs,s in links.values() for r in rs)),audited_targeted_reactions=sorted(wanted),audited_accessions=len(accessions),uniform_pair_sample_n=len(sampledf),uniform_pair_sample_seed=20261017,uniform_sample_evidence_counts=sampledf.evidence.value_counts().to_dict(),uniform_sample_direct_rhea_fraction=float(sampledf.direct_rhea.mean()),uniform_sample_supported_fraction=float(sampledf.eligible_for_panel.mean()),interpretation='Annotation concordance, not experimental catalytic validation. Unresolved links are not false positives. Component-number collisions indicate a namespace failure pattern, but exact upstream source cause cannot be proved from the final merged file alone.',snapshot_date='2026-10-17',mapping_sha256=hashlib.sha256((PROJECT/'uniprot-msd-links.tsv').read_bytes()).hexdigest())
    (artifact('mapping_quality_summary.json')).write_text(json.dumps(doc,indent=2));print(pd.DataFrame(report).to_string(index=False),flush=True);print(json.dumps(doc,indent=2),flush=True)

if __name__=='__main__':main()
