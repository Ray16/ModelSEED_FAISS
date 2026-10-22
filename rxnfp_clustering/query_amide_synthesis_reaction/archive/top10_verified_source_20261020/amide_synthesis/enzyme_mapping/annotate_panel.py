"""Add reaction-specific evidence and organism metadata without changing sequences."""

from amide_synthesis.infrastructure.paths import PROJECT
from pathlib import Path
import csv,json,hashlib,collections,re
from amide_synthesis.infrastructure.review import artifact, REVIEW, require_unfrozen_review
RUN=REVIEW

def main():
    require_unfrozen_review()
    source=artifact('selected50.tsv');rows=list(csv.DictReader(source.open(),delimiter='\t'))
    audit={(r['accession'],r['reaction']):r for r in csv.DictReader(artifact('mapping_link_audit.tsv').open(),delimiter='\t')}
    extra=['protein_name','organism','taxon_id','gene','reviewed','sequence_status','reaction_evidence_tier','reaction_evidence']
    for r in rows:
        a=audit[(r['accession'],r['reaction'])]
        d=json.loads((RUN/'evidence/uniprot_mapping_snapshot'/(r['accession']+'.json')).read_text())
        expected=set(a['expected_rhea'].split(';'));evidence=[]
        for comment in d.get('comments',[]):
            reaction=comment.get('reaction',{})
            ids={x['id'].split(':')[1] for x in reaction.get('reactionCrossReferences',[]) if re.fullmatch(r'RHEA:\d+',x['id'])}
            if ids & expected:evidence.extend(reaction.get('evidences',[]))
        codes={e.get('evidenceCode') for e in evidence}
        tier='experimental_direct' if 'ECO:0000269' in codes else 'homology_supported_direct' if 'ECO:0000250' in codes else 'computational_direct' if codes & {'ECO:0000255','ECO:0000256'} else 'curated_direct'
        r.update(protein_name=a['protein_name'],organism=a['organism'],taxon_id=a['taxon_id'],gene=';'.join(g.get('geneName',{}).get('value','') for g in d.get('genes',[])),reviewed=d.get('entryType',''),sequence_status=d.get('proteinDescription',{}).get('flag','Complete'),reaction_evidence_tier=tier,reaction_evidence=json.dumps(evidence,separators=(',',':')))
    with source.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)
    manifest=RUN/'cofold/manifest.json';doc=json.loads(manifest.read_text())
    assert {(r['accession'],r['sequence_sha256'],r['reaction']) for r in rows}=={(r['accession'],r['sequence_sha256'],r['reaction']) for r in doc['inputs']}
    doc['sources'][str(source.resolve())]=hashlib.sha256(source.read_bytes()).hexdigest();manifest.write_text(json.dumps(doc,indent=2))
    summary={'selected_evidence_tiers':dict(collections.Counter(r['reaction_evidence_tier'] for r in rows)),'sequence_and_reaction_identity_unchanged':True}
    (RUN/'sequences/selected50_annotation_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
