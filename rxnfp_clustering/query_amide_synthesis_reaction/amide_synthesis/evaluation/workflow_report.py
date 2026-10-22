"""Generate stage documentation and current counts from saved artifacts."""
from pathlib import Path
import datetime
import json

from amide_synthesis.stages import STAGES, PHASES, dependencies
from amide_synthesis.infrastructure.paths import PROJECT, RES

DIAGRAM = '''flowchart TD
    accTitle: Reaction Search to Structural Evaluation
    accDescr: Similarity and chemistry identify reactions. Evidence-preserving mapping supplies proteins. Evidence convergence and balanced reaction coverage create separate modeling cohorts, whose acid-activation geometry is evaluated against measured controls.
    query["Target reaction and mechanistic query variants"] --> reactions["Similarity search and free-amine acylation screen"]
    reactions --> mapping["Reaction-protein mapping with evidence"]
    mapping --> candidates["Merged candidates and annotations"]
    candidates --> shortlist["Bench eligibility and diversity selection"]
    candidates --> convergence["Evidence convergence: 17-protein cohort"]
    mapping --> balanced["Reaction coverage and sequence diversity: 50-protein cohort"]
    convergence --> modeling["Protein-ATP-Mg-adipate ensembles"]
    balanced --> modeling
    modeling --> geometry["Assembly checks and activation geometry"]
    geometry --> evaluation["Experimental controls and assay priorities"]
'''


def stage_rows():
    return [dict(name=s.name, phase=s.phase, module='amide_synthesis.'+s.module,
        method=s.description, inputs=list(s.inputs), outputs=list(s.outputs),
        tracked=s.tracked, source_dependencies=list(s.source_dependencies), dependencies=list(dependencies(s.name)), network=s.needs_uniprot)
        for s in STAGES]


def describe(format='markdown'):
    if format=='mermaid': return DIAGRAM
    if format=='json': return json.dumps({'phases':PHASES, 'stages':stage_rows()}, indent=2)
    rows=['# Pipeline stage index', '', 'Generated from `amide_synthesis/stages.py`. The detailed methods are in [PIPELINE.md](../PIPELINE.md).', '', '```mermaid', DIAGRAM.rstrip(), '```', '', '| Phase | Stage | Method | Canonical module |', '|---|---|---|---|']
    rows += [f'| {s.phase} | `{s.name}` | {s.description} | `amide_synthesis.{s.module}` |' for s in STAGES]
    rows += ['', 'The 17- and 50-protein cohorts are separate selection policies. Historical campaign inputs remain frozen.']
    return '\n'.join(rows)


def snapshot():
    from amide_synthesis.infrastructure.freshness import status
    def read(name):
        path=RES/name
        return json.loads(path.read_text()) if path.exists() else None
    counts={name:read(name+'_summary.json') for name in
            ['screen','precomputed_link','master_candidates','shortlist','converged']}
    stages=stage_rows()
    for row in stages:
        row['status']=status(row['name']) if row['tracked'] else 'untracked'
    campaigns=[]
    for folder,policy in [('boltz2_activation_20261016','evidence_convergence'),
                          ('colleague_review_20261017/cofold','balanced_reaction_diversity')]:
        root=RES/folder; manifest=root/'manifest.json'
        if not manifest.exists():continue
        source=json.loads(manifest.read_text())
        campaigns.append(dict(path=str(root.relative_to(PROJECT)), selection_policy=policy,
            expected_ensembles=len(source['inputs']), completion_records=len(list((root/'state').glob('*.done.json'))),
            manifest=str(manifest.relative_to(PROJECT)), frozen=True,
            interpretation='Acid activation only; HMDA and activated intermediate absent'))
    return dict(generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                phases=PHASES,stages=stages,counts=counts,campaigns=campaigns,
                limitations='Counts describe saved source snapshots; annotation support and structural predictions are not biochemical validation.')


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='write a current workflow manifest; otherwise print it')
    args=parser.parse_args(argv); result=snapshot(); text=json.dumps(result,indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text)
    else:print(text,end='')
    return 0
