#!/usr/bin/env python
"""Mordred descriptor reaction-difference embedding — the *richer physchem* axis.

  mordred_diff   1613 2D descriptors, Sum(products) - Sum(reactants)

Motivation: `descriptors_diff` (217 RDKit 2D descriptors) already ties for the
top of the clustering plateau (EC3 AMI 0.624). Mordred exposes ~7x more 2D
descriptors (topological/connectivity/autocorrelation/information-content
families RDKit lacks). If any *off-the-shelf* representation can beat the
plateau, a much larger descriptor basis in the family that already wins is the
best-motivated candidate.

Construction mirrors 05_embed_extra.py's descriptors_diff exactly: per-mol raw
descriptor vector (non-finite -> 0, lone protons / 0-heavy-atom skipped),
reaction = Sum(products) - Sum(reactants), non-finite -> 0, saved raw. The
downstream sweep (07/08) StandardScales it (mordred_diff added to CONTINUOUS)
then TruncatedSVD -> 256-d cosine, so absolute descriptor scale is handled
downstream, identically to descriptors_diff.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from mordred import Calculator, descriptors

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data"); EMB = os.path.join(HERE, "embeddings")
os.makedirs(EMB, exist_ok=True)
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, file=sys.stderr, flush=True)

if os.path.exists(os.path.join(EMB, "mordred_diff.npy")):
    log("mordred_diff.npy exists, skipping"); sys.exit(0)

df = pd.read_parquet(os.path.join(DATA, "dataset.parquet"))
rxns = df.rxn_smiles.tolist(); N = len(rxns)

calc = Calculator(descriptors, ignore_3D=True)
D = len(calc.descriptors)
log(f"{D} Mordred 2D descriptors")

def unique_mols():
    s = set()
    for r in rxns:
        L, R = r.split(">>"); s.update(L.split(".")); s.update(R.split("."))
    return sorted(s)

umols = unique_mols(); log(f"{len(umols)} unique molecules")

def mol_vec(m):
    """Raw Mordred vector for one mol; non-finite / error entries -> 0."""
    if m is None or m.GetNumHeavyAtoms() == 0:
        return np.zeros(D, np.float32)
    try:
        res = calc(m)
        v = np.array([float(x) if _num(x) else 0.0 for x in res.values()], np.float32)
    except Exception:
        return np.zeros(D, np.float32)
    v[~np.isfinite(v)] = 0.0
    return v

def _num(x):
    try:
        f = float(x); return f == f  # not NaN, and castable
    except Exception:
        return False

vecs = {}
t = time.time()
for i, s in enumerate(umols):
    vecs[s] = mol_vec(Chem.MolFromSmiles(s))
    if i % 500 == 0:
        log(f"  mols {i}/{len(umols)} ({time.time()-t:.0f}s)")

X = np.zeros((N, D), np.float32); mask = np.ones(N, bool)
for i, r in enumerate(rxns):
    L, R = r.split(">>"); lmols = L.split("."); rmols = R.split(".")
    X[i] = sum(vecs[s] for s in rmols) - sum(vecs[s] for s in lmols)
X[~np.isfinite(X)] = 0.0
np.save(os.path.join(EMB, "mordred_diff.npy"), X)
np.save(os.path.join(EMB, "mordred_diff_mask.npy"), mask)
log(f"saved mordred_diff.npy {X.shape}, mask {mask.sum()}/{N} (100%) OK")
