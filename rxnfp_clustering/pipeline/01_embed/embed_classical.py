#!/usr/bin/env python
"""Four untried *pure-RDKit* reaction-difference fingerprints, produced in one
pass over the unique-molecule cache. These complete the classical fingerprint
space the Part-I/II sweep left open and test one theoretically EC-aligned axis:

  fcfp4_diff      2048d  Functional-Class Morgan (useFeatures=True), radius 2.
                         Atoms are abstracted to pharmacophore roles (donor/
                         acceptor/aromatic/halogen/basic/acidic) rather than
                         exact element+degree as in standard ECFP. EC subclasses
                         are functional-transformation labels, so a functional-
                         class circular FP is the untried substructure axis with
                         the strongest a-priori reason to separate EC better than
                         exact-atom ECFP.
  toptorsion_diff 2048d  Topological Torsion (Nilakantan 1987): every 4-atom
                         linear path, hashed. A path/torsion axis distinct from
                         circular (ECFP) and atom-pair.
  avalon_diff     2048d  Avalon substructure fingerprint (Gedeck 2006) — a
                         curated enumerated-feature FP, different feature basis
                         from Morgan.
  rdkitfp_diff    2048d  RDKit path-based (Daylight-style) fingerprint — linear
                         subgraph paths up to length 7, not circular.

Convention matches embed_erg.py exactly: per-unique-SMILES cache; reaction
vector = Sum(products) - Sum(reactants) over count/bit vectors; on any parse
failure the reaction row is zeroed and <name>_diff_mask[i] = False. Count
fingerprints are used where the generator supports them (fcfp/toptorsion/rdkit)
so the difference is a signed integer spectrum; Avalon is a folded bit vector.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator as rfg
from rdkit.Avalon import pyAvalonTools

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(HERE, "data"); EMB = os.path.join(HERE, "embeddings")
os.makedirs(EMB, exist_ok=True)
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, file=sys.stderr, flush=True)

FPSIZE = 2048
df = pd.read_parquet(os.path.join(DATA, "dataset.parquet"))
rxns = df.rxn_smiles.tolist(); N = len(rxns)

# generators (built once)
_fcfp = rfg.GetMorganGenerator(radius=2, fpSize=FPSIZE,
        atomInvariantsGenerator=rfg.GetMorganFeatureAtomInvGen())
_tt   = rfg.GetTopologicalTorsionGenerator(fpSize=FPSIZE)
_rdk  = rfg.GetRDKitFPGenerator(fpSize=FPSIZE, maxPath=7)

# name -> (fn(mol) -> np.float32 vector or None)
def _fcfp_fn(m): return _fcfp.GetCountFingerprintAsNumPy(m).astype(np.float32)
def _tt_fn(m):   return _tt.GetCountFingerprintAsNumPy(m).astype(np.float32)
def _rdk_fn(m):  return _rdk.GetCountFingerprintAsNumPy(m).astype(np.float32)
def _av_fn(m):
    from rdkit.DataStructs import ConvertToNumpyArray
    bv = pyAvalonTools.GetAvalonFP(m, nBits=FPSIZE)
    arr = np.zeros(FPSIZE, np.float32); ConvertToNumpyArray(bv, arr); return arr

SPECS = [("fcfp4_diff", _fcfp_fn), ("toptorsion_diff", _tt_fn),
         ("avalon_diff", _av_fn), ("rdkitfp_diff", _rdk_fn)]

# skip any already produced
SPECS = [(n, f) for (n, f) in SPECS
         if not os.path.exists(os.path.join(EMB, n + ".npy"))]
if not SPECS:
    log("all classical embeddings already present, nothing to do"); sys.exit(0)
log("producing:", ", ".join(n for n, _ in SPECS))

def unique_mols():
    s = set()
    for r in rxns:
        L, R = r.split(">>"); s.update(L.split(".")); s.update(R.split("."))
    return sorted(s)

umols = unique_mols(); log(f"{len(umols)} unique molecules")

# vecs[name][smi] -> vector or None
vecs = {n: {} for n, _ in SPECS}
t = time.time()
for i, s in enumerate(umols):
    m = Chem.MolFromSmiles(s)
    for n, fn in SPECS:
        if m is None:
            vecs[n][s] = None; continue
        try:
            vecs[n][s] = fn(m)
        except Exception:
            vecs[n][s] = None
    if i % 2000 == 0: log(f"  mols {i}/{len(umols)} ({time.time()-t:.0f}s)")

for n, _ in SPECS:
    v = vecs[n]
    n_fail = sum(x is None for x in v.values())
    log(f"{n}: {n_fail}/{len(umols)} mols failed")
    X = np.zeros((N, FPSIZE), np.float32); mask = np.ones(N, bool)
    for i, r in enumerate(rxns):
        L, R = r.split(">>"); lmols = L.split("."); rmols = R.split(".")
        if any(v[s] is None for s in lmols) or any(v[s] is None for s in rmols):
            mask[i] = False; continue
        X[i] = sum(v[s] for s in rmols) - sum(v[s] for s in lmols)
    np.save(os.path.join(EMB, n + ".npy"), X)
    np.save(os.path.join(EMB, n.replace("_diff", "_diff_mask") if n.endswith("_diff")
                         else n + "_mask") + ".npy", mask)
    log(f"saved {n}.npy {X.shape}, mask {mask.sum()}/{N} ({100*mask.mean():.1f}%) OK")
