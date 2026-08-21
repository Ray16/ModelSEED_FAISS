#!/usr/bin/env python
"""Condensed-Graph-of-Reaction (CGR) fingerprints — the atom-mapped, bond-change
representation. Pipeline: RXNMapper atom-maps each reaction (GPU) -> chython
composes the CGR (bonds carry a reactant->product dynamic label) -> fold the CGR
substructure hashes to fixed-length bit vectors.

  cgr_linear   linear (path) CGR fingerprint, 2048 bits
  cgr_morgan   morgan (circular) CGR fingerprint, 2048 bits

Unlike difference fingerprints (product FP - reactant FP), the CGR encodes the
*actual* broken/formed bonds via true atom mapping, so it can express the
transformation that distinguishes fine EC levels.

Also writes cgr_mask.npy (bool, which reactions mapped+composed OK) and
cgr_confidence.npy (RXNMapper confidence).
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
DATA = os.path.join(HERE, "data"); RESULTS = os.path.join(HERE, "results"); CACHE = os.path.join(HERE, "cache")
EMB = os.path.join(HERE, "embeddings"); os.makedirs(EMB, exist_ok=True)
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, file=sys.stderr, flush=True)

df = pd.read_parquet(os.path.join(DATA, "dataset.parquet"))
rxns = df.rxn_smiles.tolist(); N = len(rxns)
LEN = 2048

# ---- 1. atom-map with RXNMapper (batched, GPU) ----
map_cache = os.path.join(DATA, "cgr_mapped.npy")
if os.path.exists(map_cache):
    mapped = np.load(map_cache, allow_pickle=True)
    conf = np.load(os.path.join(DATA, "cgr_confidence.npy"))
    log(f"loaded cached atom maps ({(mapped!='').sum()} non-empty)")
else:
    from rxnmapper import BatchedMapper
    mapper = BatchedMapper(batch_size=32)
    mapped = np.empty(N, dtype=object); conf = np.zeros(N, np.float32)
    log("atom-mapping with RXNMapper ...")
    t = time.time()
    for i, info in enumerate(mapper.map_reactions_with_info(rxns)):
        mapped[i] = info.get("mapped_rxn", "") if isinstance(info, dict) else ""
        conf[i] = info.get("confidence", 0.0) if isinstance(info, dict) else 0.0
        if i % 1000 == 0: log(f"  {i}/{N} ({time.time()-t:.0f}s)")
    np.save(map_cache, mapped); np.save(os.path.join(DATA, "cgr_confidence.npy"), conf)
    log(f"atom-mapping done ({time.time()-t:.0f}s, {(mapped!='').sum()} mapped)")

# ---- 2. compose CGR with chython, 3. fold to bits ----
from chython import smiles as chy_smiles
Xlin = np.zeros((N, LEN), np.float32)
Xmor = np.zeros((N, LEN), np.float32)
mask = np.zeros(N, bool)
ok = 0; t = time.time()
for i, m in enumerate(mapped):
    if not m: continue
    try:
        r = chy_smiles(m)
        cgr = r.compose()
        for bit in cgr.linear_bit_set(min_radius=1, max_radius=4, length=LEN):
            Xlin[i, bit] = 1.0
        for bit in cgr.morgan_bit_set(min_radius=1, max_radius=4, length=LEN):
            Xmor[i, bit] = 1.0
        mask[i] = True; ok += 1
    except Exception:
        continue
    if i % 2000 == 0: log(f"  CGR {i}/{N} ok={ok} ({time.time()-t:.0f}s)")
log(f"CGR composed for {ok}/{N} reactions ({100*ok/N:.1f}%)")

np.save(os.path.join(EMB, "cgr_linear.npy"), Xlin)
np.save(os.path.join(EMB, "cgr_morgan.npy"), Xmor)
np.save(os.path.join(DATA, "cgr_mask.npy"), mask)
log("saved cgr_linear, cgr_morgan, cgr_mask")
