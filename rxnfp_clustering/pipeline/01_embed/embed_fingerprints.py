#!/usr/bin/env python
"""Compute reaction embeddings with many fingerprint methods.

Each method -> embeddings/<method>.npy  (float32, row order == dataset.parquet).
Difference fingerprints: sum(product mol fps) - sum(reactant mol fps).
Concat fingerprints: [reactant-side binary FP ; product-side binary FP].

Methods:
  drfp          DRFP 2048 (reaction difference fp, Probst et al.)
  ecfp4_diff    Morgan r=2 count difference, 2048
  ecfp4_concat  Morgan r=2 binary, reactant|product concat, 2*2048
  maccs_diff    MACCS-166 count difference
  atompair_diff AtomPair folded 2048 count difference
  rdk_diff      RDKit topological folded 2048 count difference
  ecfp4_drfp    concat(ecfp4_diff, drfp)   <- ECFP4+DRFP combo
  rxnfp         pretrained rxn-BERT, 256   (separate script 02b)
"""
import os, sys, time
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import rdFingerprintGenerator as fpg
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
DATA = os.path.join(HERE, "data"); RESULTS = os.path.join(HERE, "results"); CACHE = os.path.join(HERE, "cache")
EMB = os.path.join(HERE, "embeddings"); os.makedirs(EMB, exist_ok=True)

def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, file=sys.stderr, flush=True)

df = pd.read_parquet(os.path.join(DATA, "dataset.parquet"))
rxns = df.rxn_smiles.tolist()
N = len(rxns)
log(f"{N} reactions")

# cache canonical mols per unique compound smiles
_mol_cache = {}
def mol(s):
    m = _mol_cache.get(s)
    if m is None and s not in _mol_cache:
        m = Chem.MolFromSmiles(s)
        _mol_cache[s] = m
    return m

def sides(rxn):
    l, r = rxn.split(">>")
    return l.split("."), r.split(".")

# ---- generators ----
morgan = fpg.GetMorganGenerator(radius=2, fpSize=2048)
atompair = fpg.GetAtomPairGenerator(fpSize=2048)
rdkgen = fpg.GetRDKitFPGenerator(fpSize=2048)

def count_vec(smis, gen, dim):
    v = np.zeros(dim, dtype=np.float32)
    for s in smis:
        m = mol(s)
        if m is None: continue
        v += gen.GetCountFingerprintAsNumPy(m).astype(np.float32)
    return v

def bit_vec(smis, gen, dim):
    v = np.zeros(dim, dtype=np.float32)
    for s in smis:
        m = mol(s)
        if m is None: continue
        v = np.maximum(v, gen.GetFingerprintAsNumPy(m).astype(np.float32))
    return v

def maccs_count(smis):
    v = np.zeros(167, dtype=np.float32)
    for s in smis:
        m = mol(s)
        if m is None: continue
        arr = np.zeros(167, dtype=np.float32)
        from rdkit.DataStructs import ConvertToNumpyArray
        ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(m), arr)
        v += arr
    return v

def build_diff(gen, dim, kind="count"):
    X = np.zeros((N, dim), dtype=np.float32)
    for i, rxn in enumerate(rxns):
        L, R = sides(rxn)
        if kind == "count":
            X[i] = count_vec(R, gen, dim) - count_vec(L, gen, dim)
        elif kind == "maccs":
            X[i] = maccs_count(R) - maccs_count(L)
        if i % 2000 == 0: log(f"  {i}/{N}")
    return X

def build_concat(gen, dim):
    X = np.zeros((N, 2*dim), dtype=np.float32)
    for i, rxn in enumerate(rxns):
        L, R = sides(rxn)
        X[i, :dim] = bit_vec(L, gen, dim)
        X[i, dim:] = bit_vec(R, gen, dim)
        if i % 2000 == 0: log(f"  {i}/{N}")
    return X

def save(name, X):
    np.save(os.path.join(EMB, name + ".npy"), X.astype(np.float32))
    log(f"saved {name}: {X.shape}")

def want(name):
    return not os.path.exists(os.path.join(EMB, name + ".npy"))

# ---- DRFP ----
if want("drfp"):
    from drfp import DrfpEncoder
    log("DRFP...")
    X = np.array(DrfpEncoder.encode(rxns, n_folded_length=2048), dtype=np.float32)
    save("drfp", X)

# ---- Morgan / ECFP4 ----
if want("ecfp4_diff"):
    log("ECFP4 diff...");  save("ecfp4_diff", build_diff(morgan, 2048, "count"))
if want("ecfp4_concat"):
    log("ECFP4 concat..."); save("ecfp4_concat", build_concat(morgan, 2048))
if want("maccs_diff"):
    log("MACCS diff...");  save("maccs_diff", build_diff(None, 167, "maccs"))
if want("atompair_diff"):
    log("AtomPair diff..."); save("atompair_diff", build_diff(atompair, 2048, "count"))
if want("rdk_diff"):
    log("RDKit diff..."); save("rdk_diff", build_diff(rdkgen, 2048, "count"))

# ---- ECFP4 + DRFP combo ----
if want("ecfp4_drfp"):
    a = np.load(os.path.join(EMB, "ecfp4_diff.npy"))
    b = np.load(os.path.join(EMB, "drfp.npy"))
    save("ecfp4_drfp", np.hstack([a, b]))

log("done (rxnfp handled by embed_rxnfp.py)")
