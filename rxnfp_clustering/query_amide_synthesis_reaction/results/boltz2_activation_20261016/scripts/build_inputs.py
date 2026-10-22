"""Build the 17-member adipate activation campaign from saved candidates."""
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

RUN = Path(__file__).resolve().parents[1]
ENZYME = Path('/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent')
REASONING = ENZYME / 'structural_modeling/modeling_reasoning'
sys.path.insert(0, str(REASONING))
import resolve_handoff
from chemistry import characterize_smiles

ADIPATE = '[O-]C(=O)CCCCC(=O)[O-]'
REFERENCES = {
    'NIS': ['https://www.rcsb.org/structure/3TO3', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC9356276/'],
    'Mur': ['https://www.rcsb.org/structure/4C12', 'https://doi.org/10.1074/jbc.M113.508135'],
    'PylC': ['https://www.uniprot.org/uniprotkb/Q46E79/entry', 'https://doi.org/10.1016/j.jmb.2012.09.007'],
    'LdmS': ['https://pmc.ncbi.nlm.nih.gov/articles/PMC9486568/'],
    'GS': ['https://www.uniprot.org/uniprotkb/Q9RDS6/entry', 'https://www.rcsb.org/structure/2GLS'],
    'GspS': ['https://pmc.ncbi.nlm.nih.gov/articles/PMC1698887/'],
    'DdaG': ['https://pmc.ncbi.nlm.nih.gov/articles/PMC2783456/'],
}

def main():
    queries = {r['accession']: r for r in json.loads((RUN / 'evidence/family_cofactor_queries.json').read_text())}
    rows = list(csv.DictReader((RUN / 'candidates.tsv').open(), delimiter='\t'))
    facts = characterize_smiles(ADIPATE)
    mol = Chem.MolFromSmiles(ADIPATE)
    assert Chem.GetFormalCharge(mol) == -2 and rdMolDescriptors.CalcMolFormula(mol) == 'C6H8O4-2'
    rule = json.loads((REASONING / 'knowledge/family_metal.json').read_text())
    inputs = []
    for row in rows:
        accession = row['uniprot']
        query = queries[accession]
        mode = 'acyl_adenylate' if query['phosphate'] == 'PA' else 'acyl_phosphate'
        family = resolve_handoff._family_per_site({'modeling_input': {'activation': {'activation_mode': mode}}}, rule)
        pfams = row['pfam'].split(';')
        if 'PF06276' in pfams:
            group, note, certainty = 'NIS', 'One Mg at the ATP site in the closest characterized AsbB structure. NIS metal occupancy varies: DesD structures also show two. One is the primary hypothesis here, not a proven universal count.', 'family_transfer_with_known_variation'
        elif 'PF08245' in pfams:
            group, note, certainty = 'Mur', 'Two-metal acyl-phosphate mechanism; same-family MurE ADP/product structure 4C12 contains two Mg ions at the active site.', 'family_structural_support'
        elif accession == 'Q46E79':
            group, note, certainty = 'PylC', 'UniProt explicitly states two magnesium ions per subunit; ATP-grasp acyl-phosphate mechanism.', 'direct_annotation_and_structure'
        elif accession == 'Q2FWC5':
            group, note, certainty = 'LdmS', 'The primary paper states two catalytic Mg ions. Crystallization Na substitutes for the second Mg in 7R8P/7R8Q; the raw one-Mg tally is not used.', 'direct_primary_paper'
        elif 'PF00120' in pfams:
            group, note, certainty = 'GS', 'Two-metal glutamine-synthetase fold; explicit two-Mg annotation for GlnA2 and family transfer for PuuA.', 'direct_annotation_or_family_transfer'
        elif 'PF03738' in pfams:
            group, note, certainty = 'GspS', 'Two catalytic magnesium ions described in the E. coli GspS primary structural study; direct binding-label cross-check agrees.', 'direct_or_close_family_structure'
        else:
            group, note, certainty = 'DdaG', 'AMP-forming adenylating ligase. One Mg follows the existing family/mechanism modeling policy; the DdaG paper establishes AMP formation but not a metal-count titration.', 'family_modeling_hypothesis'
        count = family['per_site_metal_count']
        seq = row['sequence']
        seqhash = hashlib.sha1(seq.encode()).hexdigest()[:16]
        msa = RUN / 'msa' / (seqhash + '.csv')
        chains = [{'protein': {'id': 'A', 'sequence': seq, 'msa': str(msa)}},
                  {'ligand': {'id': 'B', 'ccd': 'ATP'}}]
        for i in range(count):
            chains.append({'ligand': {'id': chr(67 + i), 'ccd': 'MG'}})
        acid_chain = chr(67 + count)
        chains.append({'ligand': {'id': acid_chain, 'smiles': ADIPATE}})
        name = f'{accession}_{count}mg'
        path = RUN / 'inputs' / (name + '.yaml')
        path.write_text(yaml.safe_dump({'version': 1, 'sequences': chains}, sort_keys=False))
        inputs.append({'accession': accession, 'name': name, 'gene': row['gene'], 'family': row['family'],
                       'mechanism_family': group, 'phosphate': query['phosphate'], 'leaving_oxygen': query['leaving_oxygen'],
                       'mg_count': count, 'active_sites_modeled': 1, 'enzyme_chains': 1,
                       'metal_assignment_basis': note, 'metal_assignment_certainty': certainty,
                       'metal_sources': REFERENCES[group], 'family_reason_result': family,
                       'uniprot_cross_check': query['metal_helper_result'],
                       'sequence_sha256': hashlib.sha256(seq.encode()).hexdigest(), 'sequence_length': len(seq),
                       'msa': str(msa), 'input': str(path), 'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                       'acid_chain': acid_chain, 'acid_smiles': ADIPATE})
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'scope': '17 saved converged candidates; first-step activation of adipate; one catalytic protein chain per input',
                'inputs': inputs, 'adipate_rdkit_check': facts,
                'settings': {'mode': 'exact', 'seed': 42, 'diffusion_samples': 50, 'recycling_steps': 3, 'sampling_steps': 200,
                             'max_parallel_samples': 41, 'actual_stock_sample_chunks': [5]*10, 'num_workers': 0,
                             'preprocessing_threads': 1, 'output_format': 'pdb'},
                'sampling_note': 'Pinned upstream uses chunk(D % M + 1). D=50,M=41 yields ten chunks of five. This selects a bounded existing stock execution path without modifying the shared sampler.',
                'ligands': {'ATP': 'Boltz CCD ATP; cached RDKit formal charge 0, matching the historical campaign representation; atom names PA/PG/O3A/O3B preserved',
                            'MG': 'Boltz CCD MG, verified formal charge +2', 'adipate': 'C6H8O4, formal charge -2'},
                'geometry_proxy': {'name': 'inverse-distance-weighted mean attack angle', 'formula': 'A=sum(theta_i/d_i)/sum(1/d_i), i=1..50',
                                   'distance_support': 'Dh=50/sum(1/d_i), plus min/median distance and fraction d<4 A',
                                   'angle': 'donor carboxylate O - reaction-specific ATP P - leaving oxygen', 'restraints': 'none',
                                   'interpretation': 'structural triage; transfer across families has not been experimentally validated'},
                'historical_outputs': '11 of the 17 have historical ensembles. They are preserved separately; exact ligand-charge/input provenance was not established, so this charge-explicit campaign is not pooled with them.',
                'source_hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [RUN/'candidates.tsv', REASONING/'knowledge/family_metal.json', REASONING/'resolve_handoff.py']}}
    (RUN/'manifest.json').write_text(json.dumps(manifest, indent=2))
    with (RUN/'metal_assignments.tsv').open('w') as f:
        fields = ['accession','gene','mechanism_family','phosphate','mg_count','metal_assignment_certainty','metal_assignment_basis','metal_sources']
        w = csv.DictWriter(f, fieldnames=fields, delimiter='\t', extrasaction='ignore'); w.writeheader(); w.writerows(inputs)
    print('Prepared', len(inputs), 'inputs;', sum(i['mg_count']==1 for i in inputs), 'one-Mg;', sum(i['mg_count']==2 for i in inputs), 'two-Mg')

if __name__ == '__main__':
    main()
