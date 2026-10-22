"""Query saved candidates through the existing enzyme annotation/metal helpers."""
import concurrent.futures
import csv
import hashlib
import importlib.util
import json
import sys
import os
import threading
import urllib.request
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
ENZYME = Path('/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent')
sys.path.insert(0, str(ENZYME))
from enzyme_discovery_agent.tools.database import get_protein_domains
from structural_modeling.proxy_policy import mechanism_from_reaction

spec = importlib.util.spec_from_file_location('cofactor_stoichiometry', ENZYME / 'structural_modeling/scripts/cofactor_stoichiometry.py')
metal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metal)

def fetch(url, timeout=30):
    path = RUN / 'evidence' / ('query_' + hashlib.sha256(url.encode()).hexdigest() + '.json')
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            pass
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            data = json.load(response)
        tmp = path.with_suffix(f'.{os.getpid()}.{threading.get_ident()}.tmp')
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(path)
        return data
    except Exception as exc:
        print('QUERY_FAILED', url, type(exc).__name__, str(exc), flush=True)
        return None

metal._fetch_json = fetch

def query(row):
    accession = row['uniprot']
    completed = RUN / 'evidence' / f'{accession}_cofactor_query.json'
    if completed.exists():
        return json.loads(completed.read_text())
    domain_file = RUN / 'evidence' / f'{accession}_domains.json'
    domains = json.loads(domain_file.read_text()) if domain_file.exists() else get_protein_domains(accession)
    domain_file.write_text(json.dumps(domains, indent=2))
    # Use the candidate's own catalytic Pfam signature, avoiding accessory CHAP.
    pfams = row['pfam'].split(';')
    catalytic = next((p for p in ['PF03738', 'PF00120', 'PF06276', 'PF27279', 'PF30763', 'PF08245'] if p in pfams), pfams[0])
    family_query = 'xref:pfam-' + catalytic
    result = metal.derive_metal_cofactor(accession, family_query=family_query, fetch=fetch)
    coordinate = mechanism_from_reaction(row['catalytic_activity'])
    if coordinate is None:
        raise ValueError('No unambiguous nucleotide-product coordinate for ' + accession)
    data = {'accession': accession, 'family': row['family'], 'family_query': family_query,
            'metal_helper_result': result, 'domains': domains,
            'phosphate': coordinate.phosphate, 'leaving_oxygen': 'O3A' if coordinate.phosphate == 'PA' else 'O3B',
            'mechanism': coordinate.mechanism, 'coordinate_basis': 'candidate catalytic-activity equation',
            'source_catalytic_activity': row['catalytic_activity']}
    (RUN / 'evidence' / f'{accession}_cofactor_query.json').write_text(json.dumps(data, indent=2))
    print(accession, row['family'], coordinate.phosphate, result, flush=True)
    return data

if __name__ == '__main__':
    rows = list(csv.DictReader((RUN / 'candidates.tsv').open(), delimiter='\t'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(query, rows))
    (RUN / 'evidence' / 'family_cofactor_queries.json').write_text(json.dumps(results, indent=2))
    print('COMPLETE', len(results), flush=True)
