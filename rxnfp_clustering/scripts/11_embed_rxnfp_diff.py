#!/usr/bin/env python
"""RXNFP difference-style reaction embeddings (GPU) — testing whether the missing
product-reactant difference is what holds plain rxnfp back.

  rxnfp_moldiff   rxn-BERT used as a PER-MOLECULE encoder (mean-pool tokens per
                  compound), reaction = Sum(products) - Sum(reactants).  Direct
                  analog of chemberta_diff / molformer_diff.
  rxnfp_sidediff  rxn-BERT used as a reaction/set encoder on each side separately:
                  V(product-side SMILES) - V(reactant-side SMILES), 256-d.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
DATA = os.path.join(HERE, "data"); RESULTS = os.path.join(HERE, "results"); CACHE = os.path.join(HERE, "cache")
EMB = os.path.join(HERE, "embeddings"); os.makedirs(EMB, exist_ok=True)
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, file=sys.stderr, flush=True)
def want(n): return not os.path.exists(os.path.join(EMB, n + ".npy"))
dev = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_parquet(os.path.join(DATA, "dataset.parquet"))
rxns = df.rxn_smiles.tolist(); N = len(rxns)

from rxnfp.transformer_fingerprints import get_default_model_and_tokenizer
model, tok = get_default_model_and_tokenizer()   # bert_ft (same as plain rxnfp.npy)
model = model.to(dev).eval()
HID = model.config.hidden_size  # 256

def encode(strings, bs=64, pool="mean"):
    """Encode a list of SMILES strings -> (len, HID) array."""
    out = np.zeros((len(strings), HID), np.float32)
    with torch.no_grad():
        for i in range(0, len(strings), bs):
            b = strings[i:i+bs]
            enc = tok.batch_encode_plus(b, padding=True, truncation=True,
                                        max_length=512, return_tensors="pt")
            enc = {k: v.to(dev) for k, v in enc.items()}
            o = model(**enc)
            h = o.last_hidden_state if hasattr(o, "last_hidden_state") else o[0]
            if pool == "mean":
                m = enc["attention_mask"].unsqueeze(-1).float()
                e = (h * m).sum(1) / m.sum(1)
            else:  # cls
                e = h[:, 0, :]
            out[i:i+len(b)] = e.cpu().numpy()
            if i % (bs*20) == 0: log(f"  {i}/{len(strings)}")
    return out

def sides(r):
    l, p = r.split(">>"); return l.split("."), p.split(".")

# ---- rxnfp_moldiff : per-molecule mean-pooled, product-sum minus reactant-sum ----
if want("rxnfp_moldiff"):
    mols = sorted({m for r in rxns for side in sides(r) for m in side})
    log(f"rxnfp_moldiff: encoding {len(mols)} unique molecules")
    mv = dict(zip(mols, encode(mols, pool="mean")))
    X = np.zeros((N, HID), np.float32)
    for i, r in enumerate(rxns):
        L, P = sides(r)
        X[i] = sum(mv[m] for m in P) - sum(mv[m] for m in L)
    np.save(os.path.join(EMB, "rxnfp_moldiff.npy"), X)
    log(f"saved rxnfp_moldiff {X.shape}")

# ---- rxnfp_sidediff : encode each side string, subtract (CLS, as rxnfp does) ----
if want("rxnfp_sidediff"):
    react_str = [".".join(sides(r)[0]) for r in rxns]
    prod_str  = [".".join(sides(r)[1]) for r in rxns]
    log("rxnfp_sidediff: encoding reactant side")
    VR = encode(react_str, pool="cls")
    log("rxnfp_sidediff: encoding product side")
    VP = encode(prod_str, pool="cls")
    np.save(os.path.join(EMB, "rxnfp_sidediff.npy"), (VP - VR).astype(np.float32))
    log(f"saved rxnfp_sidediff {(VP-VR).shape}")

log("rxnfp diff embeddings done")
