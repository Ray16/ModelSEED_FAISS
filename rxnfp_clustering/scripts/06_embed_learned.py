#!/usr/bin/env python
"""Learned reaction embeddings (GPU).

  chemberta_diff   ChemBERTa per-molecule mean-pooled embedding (384-d),
                   reaction = Sum(products) - Sum(reactants)
  rxnfp_pretrained rxn-BERT *pretrained* variant (vs the fine-tuned default
                   already saved as rxnfp.npy)

Optional (set MOLFORMER=1): molformer_diff via ibm/MoLFormer-XL-both-10pct.
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

def unique_mols():
    """All unique compound SMILES across the dataset (embed each once)."""
    s = set()
    for r in rxns:
        L, R = r.split(">>"); s.update(L.split(".")); s.update(R.split("."))
    return sorted(s)

def embed_mols(smis, tok, mdl, bs=256, pool="mean", maxlen=256):
    vecs = {}
    order = list(smis)
    with torch.no_grad():
        for i in range(0, len(order), bs):
            b = order[i:i+bs]
            enc = tok(b, padding=True, truncation=True, max_length=maxlen, return_tensors="pt")
            enc = {k: v.to(dev) for k, v in enc.items()}
            o = mdl(**enc)
            h = o.last_hidden_state
            if pool == "mean":
                m = enc["attention_mask"].unsqueeze(-1).float()
                e = (h * m).sum(1) / m.sum(1)
            else:  # cls
                e = h[:, 0, :]
            for j, s in enumerate(b): vecs[s] = e[j].cpu().numpy()
            if i % (bs*10) == 0: log(f"  mols {i}/{len(order)}")
    return vecs

def diff_from_molvecs(vecs, dim):
    X = np.zeros((N, dim), np.float32)
    for i, r in enumerate(rxns):
        L, R = r.split(">>")
        vr = sum(vecs[s] for s in R.split("."))
        vl = sum(vecs[s] for s in L.split("."))
        X[i] = vr - vl
    return X

# ---- ChemBERTa ----
if want("chemberta_diff"):
    from transformers import AutoTokenizer, AutoModel
    name = "DeepChem/ChemBERTa-77M-MLM"
    log(f"loading {name}")
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModel.from_pretrained(name).to(dev).eval()
    dim = mdl.config.hidden_size
    umols = unique_mols(); log(f"{len(umols)} unique molecules")
    vecs = embed_mols(umols, tok, mdl, pool="mean")
    np.save(os.path.join(EMB, "chemberta_diff.npy"), diff_from_molvecs(vecs, dim).astype(np.float32))
    log(f"saved chemberta_diff (dim {dim})")

# ---- rxnfp pretrained variant ----
if want("rxnfp_pretrained"):
    try:
        from rxnfp.transformer_fingerprints import (RXNBERTFingerprintGenerator,
                                                    get_default_model_and_tokenizer)
        model, tokenizer = get_default_model_and_tokenizer("bert_pretrained")
        model = model.to(dev).eval()
        BS = 64; vecs = np.zeros((N, 256), np.float32)
        with torch.no_grad():
            for i in range(0, N, BS):
                b = rxns[i:i+BS]
                enc = tokenizer.batch_encode_plus(b, padding=True, truncation=True,
                                                  max_length=512, return_tensors="pt")
                enc = {k: v.to(dev) for k, v in enc.items()}
                o = model(**enc)
                cls = o.last_hidden_state[:, 0, :] if hasattr(o, "last_hidden_state") else o[0][:, 0, :]
                vecs[i:i+len(b)] = cls.cpu().numpy()
        np.save(os.path.join(EMB, "rxnfp_pretrained.npy"), vecs)
        log("saved rxnfp_pretrained")
    except Exception as ex:
        log(f"rxnfp_pretrained failed: {ex}")

# ---- optional MolFormer ----
if os.environ.get("MOLFORMER") == "1" and want("molformer_diff"):
    from transformers import AutoTokenizer, AutoModel
    name = "ibm/MoLFormer-XL-both-10pct"
    rev = "7b12d946c181"  # 2024 revision; later ones need transformers.masking_utils
    log(f"loading {name}@{rev}")
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True, revision=rev)
    mdl = AutoModel.from_pretrained(name, trust_remote_code=True, deterministic_eval=True,
                                    revision=rev).to(dev).eval()
    dim = mdl.config.hidden_size
    umols = unique_mols()
    vecs = embed_mols(umols, tok, mdl, pool="mean")
    np.save(os.path.join(EMB, "molformer_diff.npy"), diff_from_molvecs(vecs, dim).astype(np.float32))
    log(f"saved molformer_diff (dim {dim})")

log("learned embeddings done")
