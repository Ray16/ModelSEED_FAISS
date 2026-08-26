#!/usr/bin/env python
"""Compute only the RDKit-based fingerprints (parallel to slow DRFP)."""
import os, sys, time
import numpy as np, pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import rdFingerprintGenerator as fpg
from rdkit.DataStructs import ConvertToNumpyArray
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
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

morgan = fpg.GetMorganGenerator(radius=2, fpSize=2048)
atompair = fpg.GetAtomPairGenerator(fpSize=2048)
rdkgen = fpg.GetRDKitFPGenerator(fpSize=2048)

def count_vec(smis, gen, dim):
    v = np.zeros(dim, np.float32)
    for s in smis:
        m = mol(s)
        if m is not None: v += gen.GetCountFingerprintAsNumPy(m).astype(np.float32)
    return v
def bit_vec(smis, gen, dim):
    v = np.zeros(dim, np.float32)
    for s in smis:
        m = mol(s)
        if m is not None: v = np.maximum(v, gen.GetFingerprintAsNumPy(m).astype(np.float32))
    return v
def maccs_count(smis):
    v = np.zeros(167, np.float32)
    for s in smis:
        m = mol(s)
        if m is None: continue
        a = np.zeros(167, np.float32); ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(m), a); v += a
    return v

def build_diff(gen, dim, kind="count"):
    X = np.zeros((N, dim), np.float32)
    for i, r in enumerate(rxns):
        L, R = sides(r)
        X[i] = (maccs_count(R)-maccs_count(L)) if kind=="maccs" else (count_vec(R,gen,dim)-count_vec(L,gen,dim))
        if i % 3000 == 0: log(f"  {i}/{N}")
    return X
def build_concat(gen, dim):
    X = np.zeros((N, 2*dim), np.float32)
    for i, r in enumerate(rxns):
        L, R = sides(r); X[i,:dim]=bit_vec(L,gen,dim); X[i,dim:]=bit_vec(R,gen,dim)
        if i % 3000 == 0: log(f"  {i}/{N}")
    return X
def save(n, X):
    if os.path.exists(os.path.join(EMB, n+".npy")): log(f"{n} exists, skip"); return
    np.save(os.path.join(EMB, n+".npy"), X.astype(np.float32)); log(f"saved {n} {X.shape}")

for n, fn in [("ecfp4_diff", lambda: build_diff(morgan,2048)),
              ("ecfp4_concat", lambda: build_concat(morgan,2048)),
              ("maccs_diff", lambda: build_diff(None,167,"maccs")),
              ("atompair_diff", lambda: build_diff(atompair,2048)),
              ("rdk_diff", lambda: build_diff(rdkgen,2048))]:
    if os.path.exists(os.path.join(EMB, n+".npy")): log(f"{n} exists, skip"); continue
    log(f"{n}..."); save(n, fn())
log("rdkit fps done")
