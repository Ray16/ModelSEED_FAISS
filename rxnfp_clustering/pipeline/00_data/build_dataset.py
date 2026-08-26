#!/usr/bin/env python
"""Build the benchmark dataset from ModelSEEDDatabase.

Output: dataset.parquet with one row per usable reaction:
  rxn_id, ec (first full 4-level EC), ec1, ec2, ec3, n_ec, rxn_smiles, n_cpd

A reaction is usable iff:
  - not obsolete
  - has >=1 full 4-level EC number
  - every participating compound has a parseable SMILES

Reaction SMILES = reactants '>>' products, each compound repeated by its
(rounded, abs, capped-at-4) stoichiometric coefficient, canonicalized by RDKit.
Cofactors are NOT stripped (RXNFP was trained on full reaction SMILES; difference
fingerprints cancel species appearing identically on both sides).
"""
import csv, glob, os, sys
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
csv.field_size_limit(10**7)

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
DATA = os.path.join(HERE, "data"); RESULTS = os.path.join(HERE, "results"); CACHE = os.path.join(HERE, "cache")
DB = os.path.join(HERE, "..", "ModelSEEDDatabase", "Biochemistry")

def load_smiles():
    smi = {}
    for f in sorted(glob.glob(os.path.join(DB, "compound_*.tsv"))):
        for row in csv.DictReader(open(f), delimiter='\t'):
            s = row.get('smiles', '')
            if s and s != 'null':
                smi[row['id']] = s
    # canonicalize once; drop unparseable
    canon = {}
    for cid, s in smi.items():
        m = Chem.MolFromSmiles(s)
        if m is not None:
            canon[cid] = Chem.MolToSmiles(m)
    return canon

def first_full_ec(e):
    if not e or e == 'null':
        return None, 0
    ecs = []
    for tok in e.replace('|', ';').split(';'):
        tok = tok.strip()
        p = tok.split('.')
        if len(p) == 4 and all(x.isdigit() for x in p):
            ecs.append(tok)
    if not ecs:
        return None, 0
    return ecs[0], len(set(ecs))

def parse_stoich(s):
    """Return list of (coef:int, cpd_id) with coef sign; skip malformed."""
    out = []
    for part in s.split(';'):
        if not part.strip():
            continue
        f = part.split(':')
        if len(f) >= 2 and f[1]:
            try:
                coef = float(f[0])
            except ValueError:
                continue
            out.append((coef, f[1]))
    return out

def build_rxn_smiles(stoich, smi):
    react, prod = [], []
    for coef, cid in stoich:
        if cid not in smi:
            return None
        mult = min(4, max(1, int(round(abs(coef)))))
        (react if coef < 0 else prod).extend([smi[cid]] * mult)
    if not react or not prod:
        return None
    return ".".join(react) + ">>" + ".".join(prod)

def main():
    smi = load_smiles()
    print(f"compounds with parseable SMILES: {len(smi)}", file=sys.stderr)
    rows = []
    for f in sorted(glob.glob(os.path.join(DB, "reaction_*.tsv"))):
        for row in csv.DictReader(open(f), delimiter='\t'):
            if row.get('is_obsolete') == '1':
                continue
            ec, n_ec = first_full_ec(row.get('ec_numbers', ''))
            if not ec:
                continue
            stoich = parse_stoich(row.get('stoichiometry', ''))
            if not stoich:
                continue
            rs = build_rxn_smiles(stoich, smi)
            if rs is None:
                continue
            p = ec.split('.')
            rows.append(dict(
                rxn_id=row['id'], ec=ec,
                ec1=p[0], ec2='.'.join(p[:2]), ec3='.'.join(p[:3]),
                n_ec=n_ec, n_cpd=len(stoich), rxn_smiles=rs))
    df = pd.DataFrame(rows).drop_duplicates('rxn_id').reset_index(drop=True)
    out = os.path.join(DATA, "dataset.parquet")
    df.to_parquet(out, index=False)
    print(f"reactions: {len(df)}  distinct EC: {df.ec.nunique()}  "
          f"EC3: {df.ec3.nunique()}  EC2: {df.ec2.nunique()}  EC1: {df.ec1.nunique()}")
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
