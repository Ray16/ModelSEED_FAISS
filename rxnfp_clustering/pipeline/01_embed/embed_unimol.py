#!/usr/bin/env python
"""Uni-Mol per-molecule representation -> reaction difference (GPU).

The 3D-geometry axis: Uni-Mol (Zhou et al. 2023) is an SE(3)-invariant
transformer pretrained on ~209M molecular conformers with a masked-atom +
3D-position-denoising objective. Unlike every other embedding in this
benchmark (2D graph fingerprints, physicochemical descriptors, SMILES-string
language models), Uni-Mol's representation is conditioned on a *generated 3D
conformer* of each molecule -- it is the only view of true molecular shape
in the pool.

  unimol_diff   512-dim float (unimolv1 84M CLS token), Sum(products) - Sum(reactants)

pip install unimol_tools (already done). UniMolRepr generates ETKDG conformers
internally; conformer-generation failures are zero-filled by the library
itself (not treated as a hard failure here). A molecule only counts as a hard
failure if RDKit cannot parse its SMILES at all -- those reactions get a zero
row and unimol_diff_mask[i] = False.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from rdkit import Chem

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
DATA = os.path.join(HERE, "data"); RESULTS = os.path.join(HERE, "results"); CACHE = os.path.join(HERE, "cache")
EMB = os.path.join(HERE, "embeddings"); os.makedirs(EMB, exist_ok=True)
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, file=sys.stderr, flush=True)

df = pd.read_parquet(os.path.join(DATA, "dataset.parquet"))
rxns = df.rxn_smiles.tolist(); N = len(rxns)
DIM = 512

def unique_mols():
    s = set()
    for r in rxns:
        L, R = r.split(">>"); s.update(L.split(".")); s.update(R.split("."))
    return sorted(s)

if os.path.exists(os.path.join(EMB, "unimol_diff.npy")):
    log("unimol_diff.npy already exists, skipping"); sys.exit(0)

# ETKDG conformer generation cost blows up on very large/highly-branched
# molecules -- a handful of ModelSEED entries (polysaccharide/polyprenol
# chains, up to 899 heavy atoms) each took >10 minutes and stalled the run.
# Heavy-atom cap: 99th percentile of the unique-molecule size distribution is
# 130 atoms, so a 120-atom cap keeps ~99% of molecules while excluding the
# pathological tail. Molecules over the cap are treated as embedding
# failures (zero row + mask=False), same as an RDKit parse failure.
MAX_HEAVY_ATOMS = 120

umols = unique_mols(); log(f"{len(umols)} unique molecules")
mols_by_smi = {m: Chem.MolFromSmiles(m) for m in umols}
too_big = {m for m, mol in mols_by_smi.items()
          if mol is not None and mol.GetNumHeavyAtoms() > MAX_HEAVY_ATOMS}
valid = [m for m in umols if mols_by_smi[m] is not None and m not in too_big]
invalid = (set(umols) - set(valid)) - too_big
log(f"{len(valid)}/{len(umols)} parse with RDKit and are <= {MAX_HEAVY_ATOMS} heavy atoms "
    f"({len(invalid)} unparseable, {len(too_big)} too large -- skipped, not attempted)")

from unimol_tools import UniMolRepr
clf = UniMolRepr(data_type="molecule", remove_hs=False, model_name="unimolv1",
                 model_size="84m", batch_size=64, use_cuda=True)

t = time.time()
reprs = clf.get_repr(valid, return_atomic_reprs=False)
log(f"Uni-Mol repr for {len(valid)} molecules took {time.time()-t:.0f}s")

vecs = {}
for s, r in zip(valid, reprs):
    vecs[s] = np.asarray(r, dtype=np.float32)
for s in invalid | too_big:
    vecs[s] = None

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

np.save(os.path.join(EMB, "unimol_diff.npy"), X)
np.save(os.path.join(EMB, "unimol_diff_mask.npy"), mask)
log(f"saved unimol_diff.npy {X.shape}, mask: {mask.sum()}/{N} ({100*mask.mean():.1f}%) OK")
