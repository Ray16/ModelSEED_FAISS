"""Prepare a traceable reference panel from an audited reaction/protein pool.

Keep every exact reaction-annotated reference. For a reaction without one,
select one related-reaction candidate by evidence, review status and accession;
retain its inferred label. Reuse models only after exact sequence/input checks.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

from amide_synthesis.enzyme_mapping.uniprot_snapshot import reaction_annotations
from amide_synthesis.structure_modeling.campaign import prepare
from amide_synthesis.structure_modeling.poses import inspect


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table(path):
    return list(csv.DictReader(Path(path).open(), delimiter='\t'))


def write_table(path, rows):
    with Path(path).open('w') as handle:
        writer = csv.DictWriter(handle, list(rows[0]), delimiter='\t')
        writer.writeheader(); writer.writerows(rows)


def stage(audit, reactions, profiles, previous, root):
    root.mkdir(parents=True, exist_ok=True)
    if (root/'reference_selection.json').exists():
        raise FileExistsError('Reference selection already exists; use prepare to resume')
    counts = table(audit/'counts.tsv')
    candidates = table(audit/'reaction_candidates.tsv')
    specs = json.loads(profiles.read_text())['reactions']
    configurations = {r['rxn_id']:r for r in table(reactions)}
    selected = [r for r in candidates if r['mapping_route']=='reaction_annotation']
    choices = []
    for count in counts:
        rid = count['rxn_id']
        if int(count['live_reaction_supported_accessions']):
            continue
        pool = [r for r in candidates if r['rxn_id']==rid and r['fragment']=='False']
        raw = []
        for page in sorted((audit/'raw').glob(rid+'_page*.json')):
            if page.name.endswith('.meta.json'):
                continue
            raw.extend(json.loads(page.read_text())['results'])
        by_id = {record['primaryAccession']:record for record in raw}
        def key(record):
            annotations = reaction_annotations(by_id[record['accession']], specs[rid].get('related_reference_rhea', []))
            experimental = any(r['evidence_tier']=='experimental_direct' for r in annotations)
            return not experimental, record['reviewed']!='True', record['accession']
        if not pool:
            raise ValueError('No traceable candidate for reaction '+rid)
        chosen = min(pool, key=key)
        selected.append(chosen)
        choices.append({'rxn_id':rid,'accession':chosen['accession'],
                        'reason':'Related-reaction experimental evidence, then reviewed status, then accession',
                        'mapping_route':chosen['mapping_route'],
                        'qualification':specs[rid].get('qualification','Candidate inference, not exact reaction support')})
    unique = {}
    for record in selected:
        unique.setdefault(record['sequence_sha256'], record)
    previous_manifest = json.loads((previous/'manifest.json').read_text())
    existing = {r['accession']:r for r in previous_manifest['inputs']}
    msa = root/'msa'; msa.mkdir(exist_ok=True)
    reuse = {}
    for record in unique.values():
        item = existing.get(record['accession'])
        if not item:
            continue
        import yaml
        sequence = yaml.safe_load(Path(item['input']).read_text())['sequences'][0]['protein']['sequence']
        if sequence != record['sequence']:
            raise ValueError('Current/campaign sequence mismatch: '+record['accession'])
        actual = inspect(item, previous/'outputs'/item['name'], include_near_attack=False)
        destination = msa/(hashlib.sha1(sequence.encode()).hexdigest()[:16]+'.csv')
        source = Path(item['msa'])
        first = next(csv.DictReader(source.open()))
        if ''.join(c for c in first['sequence'] if not c.islower()).replace('-','') != sequence:
            raise ValueError('Existing MSA differs from its exact sequence')
        shutil.copy2(source, destination)
        reuse[record['accession']] = {'source_campaign':str(previous.resolve()), 'source_item':item,
                                      'actual_models_checked':actual['n_models'], 'msa_sha256':sha(source)}
    write_table(root/'references.tsv', selected)
    fasta = root/'references.fasta'
    fasta.write_text(''.join(f">{r['accession']}\n{r['sequence']}\n" for r in unique.values()))
    rows = []
    for record in unique.values():
        route = configurations[record['rxn_id']]['target_query']
        if route not in ['atp_amp_ligase','atp_adp_ligase']:
            raise ValueError('Declare a mechanism for this non-ATP reaction: '+record['rxn_id'])
        adp = route=='atp_adp_ligase'
        rows.append({'accession':record['accession'],'sequence':record['sequence'],
            'mg_count':2 if adp else 1,'phosphate':'PG' if adp else 'PA',
            'family':route,'msa':str((msa/(hashlib.sha1(record['sequence'].encode()).hexdigest()[:16]+'.csv')).resolve()),
            'metal_assignment_basis':'Comparative acid-activation protocol: 2 Mg for ADP/PG route, 1 Mg for AMP/PA; family-transfer hypothesis, not a universal native metal count'})
    write_table(root/'selection.tsv',rows)
    result={'scope':'All reaction-annotated references plus one explicitly inferred candidate per unmapped reaction',
            'unique_proteins':len(unique),'reaction_protein_pairs':len(selected),'candidate_choices':choices,
            'reused':reuse,'new_proteins':[r['accession'] for r in unique.values() if r['accession'] not in reuse],
            'input_sha256':{str(p.resolve()):sha(p) for p in [audit/'manifest.json',reactions,profiles,previous/'manifest.json']},
            'selection_sha256':sha(root/'selection.tsv'), 'scope_limit':'Monomeric protein + ATP + Mg + adipate; no HMDA/activated intermediate or biological-assembly certification'}
    (root/'reference_selection.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def prepare_and_reuse(root):
    selection=json.loads((root/'reference_selection.json').read_text())
    if sha(root/'selection.tsv') != selection['selection_sha256']:
        raise ValueError('Panel selection changed')
    for name,digest in selection['input_sha256'].items():
        if sha(name)!=digest:
            raise ValueError('Panel source changed: '+name)
    campaign=root/'campaign'
    manifest=prepare(root/'selection.tsv',campaign)
    reused={}
    for item in manifest['inputs']:
        source=selection['reused'].get(item['accession'])
        if not source:
            continue
        old=source['source_item']
        for key in ['mg_count','phosphate','acid_chain','sequence_length']:
            if item[key] != old[key]:
                raise ValueError('Reuse input mismatch: '+item['accession']+' '+key)
        if item['msa_sha256'] != source['msa_sha256']:
            raise ValueError('Reuse MSA changed')
        destination=campaign/'outputs'/item['name']
        destination.symlink_to(Path(source['source_campaign'])/'outputs'/old['name'],target_is_directory=True)
        checked=inspect(item,destination)
        (campaign/'state'/(item['name']+'.done.json')).write_text(json.dumps(checked,indent=2)+'\n')
        reused[item['accession']]=source
    manifest['reused_models']=reused
    manifest['reference_selection']=str((root/'reference_selection.json').resolve())
    manifest['producer_code_sha256']={str(p.resolve()):sha(p) for p in [Path(__file__),Path(__file__).with_name('fleet.py'),Path(__file__).with_name('poses.py')]}
    (campaign/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return {'campaign':str(campaign),'ensembles':len(manifest['inputs']),'reused_ensembles':len(reused),
            'missing_ensembles':len(manifest['inputs'])-len(reused)}


def freeze_sources(root):
    """Keep durable copies of the exact upstream versions used for preparation."""
    if (root/'campaign/manifest.json').exists():
        raise ValueError('Cannot revise provenance after campaign preparation')
    path=root/'reference_selection.json'
    selection=json.loads(path.read_text())
    evidence=root/'evidence';evidence.mkdir(exist_ok=True)
    frozen={};original={}
    for index,(name,digest) in enumerate(selection['input_sha256'].items()):
        source=Path(name)
        if sha(source)!=digest:
            raise ValueError('Source changed before freezing: '+name)
        target=evidence/(str(index)+'_'+source.name)
        shutil.copy2(source,target)
        frozen[str(target.resolve())]=digest
        original[str(target.resolve())]=name
    selection['input_sha256']=frozen
    selection['original_source_paths']=original
    path.write_text(json.dumps(selection,indent=2)+'\n')
    return {'frozen_inputs':len(frozen)}


def launch_when_ready(root):
    """Explicit launch action, on the preflighted node, with a bounded MSA wait."""
    import socket,time
    if socket.gethostname()!='lambda13':
        raise RuntimeError('Launch on the preflighted lambda13 host')
    deadline=time.time()+10800
    while not (root/'msa/index.tsv').exists():
        (root/'launch_status.json').write_text(json.dumps({'status':'waiting_for_batched_msa',
                                                          'host':socket.gethostname(),'pid':__import__('os').getpid()})+'\n')
        if time.time()>deadline:
            raise RuntimeError('Batched MSA wait timed out; no GPUs reserved')
        time.sleep(10)
    if not (root/'campaign/manifest.json').exists():
        prepare_and_reuse(root)
    (root/'launch_status.json').write_text(json.dumps({'status':'coordinator_started',
                                                      'host':socket.gethostname(),'pid':__import__('os').getpid()})+'\n')
    from amide_synthesis.structure_modeling.fleet import main as fleet
    return fleet(['coordinator','--run-dir',str(root/'campaign')])


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['stage','freeze-sources','prepare','launch'])
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--audit',type=Path)
    p.add_argument('--reactions',type=Path)
    p.add_argument('--profiles',type=Path)
    p.add_argument('--previous',type=Path)
    args=p.parse_args(argv)
    result=(stage(args.audit,args.reactions,args.profiles,args.previous,args.root) if args.mode=='stage' else
            freeze_sources(args.root) if args.mode=='freeze-sources' else
            launch_when_ready(args.root.resolve()) if args.mode=='launch' else prepare_and_reuse(args.root))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
