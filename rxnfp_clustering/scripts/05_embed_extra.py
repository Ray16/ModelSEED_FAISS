#!/usr/bin/env python
"""Extra structural / physicochemical reaction embeddings (CPU, RDKit + MHFP).

  ecfp2_diff     Morgan r=1 count difference, 2048   (radius sweep)
  ecfp6_diff     Morgan r=3 count difference, 2048   (radius sweep)
  avalon_diff    Avalon 2048 count difference
  torsion_diff   Topological-torsion 2048 count difference
  secfp_diff     MHFP/SECFP 2048 binary difference   (minhash family)
  descriptors_diff  ~200 RDKit 2D descriptors, product-sum minus reactant-sum
"""
import os, sys, time
import numpy as np, pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator as fpg
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from mhfp.encoder import MHFPEncoder
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
DATA = os.path.join(HERE, "data"); RESULTS = os.path.join(HERE, "results"); CACHE = os.path.join(HERE, "cache")
EMB = os.path.join(HERE, "embeddings"); os.makedirs(EMB, exist_ok=True)
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, file=sys.stderr, flush=True)

df = pd.read_parquet(os.path.join(DATA, "dataset.parquet"))
rxns = df.rxn_smiles.tolist(); N = len(rxns)
_mc = {}
def mol(s):
    if s not in _mc: _mc[s] = Chem.MolFromSmiles(s)
    return _mc[s]
def sides(r): l, r2 = r.split(">>"); return l.split("."), r2.split(".")
def want(n): return not os.path.exists(os.path.join(EMB, n + ".npy"))
def save(n, X): np.save(os.path.join(EMB, n+".npy"), X.astype(np.float32)); log(f"saved {n} {X.shape}")

morgan1 = fpg.GetMorganGenerator(radius=1, fpSize=2048)
morgan3 = fpg.GetMorganGenerator(radius=3, fpSize=2048)
torsion = fpg.GetTopologicalTorsionGenerator(fpSize=2048)

def count_vec(smis, gen, dim):
    v = np.zeros(dim, np.float32)
    for s in smis:
        m = mol(s)
        if m is not None: v += gen.GetCountFingerprintAsNumPy(m).astype(np.float32)
    return v
def avalon_count(smis):
    v = np.zeros(2048, np.float32)
    for s in smis:
        m = mol(s)
        if m is None: continue
        cv = pyAvalonTools.GetAvalonCountFP(m, nBits=2048)
        for i, c in cv.GetNonzeroElements().items(): v[i] += c
    return v
_mhfp = MHFPEncoder(n_permutations=2048)
def secfp_vec(smis):
    v = np.zeros(2048, np.float32)
    for s in smis:
        try:
            fp = _mhfp.secfp_from_smiles(s, length=2048)
            v = np.maximum(v, np.array(fp, np.float32))
        except Exception: pass
    return v

def build_diff(fn, dim):
    X = np.zeros((N, dim), np.float32)
    for i, r in enumerate(rxns):
        L, R = sides(r); X[i] = fn(R) - fn(L)
        if i % 3000 == 0: log(f"  {i}/{N}")
    return X

if want("ecfp2_diff"): log("ecfp2..."); save("ecfp2_diff", build_diff(lambda s: count_vec(s, morgan1, 2048), 2048))
if want("ecfp6_diff"): log("ecfp6..."); save("ecfp6_diff", build_diff(lambda s: count_vec(s, morgan3, 2048), 2048))
if want("torsion_diff"): log("torsion..."); save("torsion_diff", build_diff(lambda s: count_vec(s, torsion, 2048), 2048))
if want("avalon_diff"): log("avalon..."); save("avalon_diff", build_diff(avalon_count, 2048))
if want("secfp_diff"): log("secfp..."); save("secfp_diff", build_diff(secfp_vec, 2048))

# ---- RDKit 2D descriptors difference ----
if want("descriptors_diff"):
    log("descriptors...")
    names = [n for n, _ in Descriptors._descList]
    calc = MolecularDescriptorCalculator(names)
    D = len(names)
    def desc_vec(smis):
        v = np.zeros(D, np.float32)
        for s in smis:
            m = mol(s)
            if m is None or m.GetNumHeavyAtoms() == 0:  # skip lone protons etc.
                continue
            try:
                d = np.array(calc.CalcDescriptors(m), np.float32)
            except Exception:
                continue
            d[~np.isfinite(d)] = 0.0
            v += d
        return v
    X = build_diff(desc_vec, D)
    X[~np.isfinite(X)] = 0.0
    save("descriptors_diff", X)

log("extra structural embeddings done")
