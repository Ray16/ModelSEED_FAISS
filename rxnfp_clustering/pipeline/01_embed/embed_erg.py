#!/usr/bin/env python
"""ErG (extended reduced graph) reaction-difference fingerprint — the
pharmacophore-topology axis (Stiefl et al. 2006; RDKit rdReducedGraphs).

ErG abstracts each atom to a pharmacophore type (donor/acceptor/aromatic/
hydrophobic/...) collapsed onto a reduced topological graph, then fingerprints
that graph. This is a fundamentally different similarity axis from ECFP-style
circular substructure fingerprints (which are exact-atom / exact-bond) and
from physchem descriptors (which are global scalars): ErG asks "does this
molecule have the same *arrangement of pharmacophore features*", independent
of the exact atoms carrying them.

  erg_diff   315-dim float, Sum(products) - Sum(reactants)

No install needed (pure RDKit). Per-unique-SMILES cache; on molecule-parse
failure, that reaction gets a zero row and erg_diff_mask[i] = False.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from rdkit import Chem
from rdkit.Chem import rdReducedGraphs

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
DATA = os.path.join(HERE, "data"); RESULTS = os.path.join(HERE, "results"); CACHE = os.path.join(HERE, "cache")
EMB = os.path.join(HERE, "embeddings"); os.makedirs(EMB, exist_ok=True)
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, file=sys.stderr, flush=True)

df = pd.read_parquet(os.path.join(DATA, "dataset.parquet"))
rxns = df.rxn_smiles.tolist(); N = len(rxns)
DIM = 315

def unique_mols():
    s = set()
    for r in rxns:
        L, R = r.split(">>"); s.update(L.split(".")); s.update(R.split("."))
    return sorted(s)

def embed_erg(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        fp = rdReducedGraphs.GetErGFingerprint(mol)
        return np.asarray(fp, dtype=np.float32)
    except Exception:
        return None

if os.path.exists(os.path.join(EMB, "erg_diff.npy")):
    log("erg_diff.npy already exists, skipping"); sys.exit(0)

umols = unique_mols(); log(f"{len(umols)} unique molecules")
vecs = {}
t = time.time()
for i, s in enumerate(umols):
    vecs[s] = embed_erg(s)
    if i % 2000 == 0: log(f"  mols {i}/{len(umols)} ({time.time()-t:.0f}s)")
n_fail = sum(v is None for v in vecs.values())
log(f"molecule embedding done: {n_fail}/{len(umols)} failed to parse/fingerprint")

X = np.zeros((N, DIM), np.float32)
mask = np.ones(N, bool)
for i, r in enumerate(rxns):
    L, R = r.split(">>")
    lmols = L.split("."); rmols = R.split(".")
    if any(vecs[s] is None for s in lmols) or any(vecs[s] is None for s in rmols):
        mask[i] = False
        continue
    vr = sum(vecs[s] for s in rmols)
    vl = sum(vecs[s] for s in lmols)
    X[i] = vr - vl

np.save(os.path.join(EMB, "erg_diff.npy"), X)
np.save(os.path.join(EMB, "erg_diff_mask.npy"), mask)
log(f"saved erg_diff.npy {X.shape}, mask: {mask.sum()}/{N} ({100*mask.mean():.1f}%) OK")
