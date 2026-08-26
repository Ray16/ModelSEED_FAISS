#!/usr/bin/env python
"""RXNFP pretrained rxn-BERT embedding (256-dim) on GPU, batched."""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
DATA = os.path.join(HERE, "data"); RESULTS = os.path.join(HERE, "results"); CACHE = os.path.join(HERE, "cache")
EMB = os.path.join(HERE, "embeddings"); os.makedirs(EMB, exist_ok=True)
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, file=sys.stderr, flush=True)

out = os.path.join(EMB, "rxnfp.npy")
if os.path.exists(out):
    log("rxnfp.npy exists, skip"); sys.exit(0)

from rxnfp.transformer_fingerprints import RXNBERTFingerprintGenerator, get_default_model_and_tokenizer
model, tokenizer = get_default_model_and_tokenizer()
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(dev).eval()
log(f"model on {dev}")

df = pd.read_parquet(os.path.join(DATA, "dataset.parquet"))
rxns = df.rxn_smiles.tolist()
N = len(rxns)

# batched CLS-embedding extraction (rxnfp uses CLS token, 256-dim)
BS = 64
vecs = np.zeros((N, 256), dtype=np.float32)
with torch.no_grad():
    for i in range(0, N, BS):
        batch = rxns[i:i+BS]
        enc = tokenizer.batch_encode_plus(batch, padding=True, truncation=True,
                                          max_length=512, return_tensors="pt")
        enc = {k: v.to(dev) for k, v in enc.items()}
        out_model = model(**enc)
        # BertModel returns (sequence_output, pooled_output) or ModelOutput
        if hasattr(out_model, "last_hidden_state"):
            cls = out_model.last_hidden_state[:, 0, :]
        else:
            cls = out_model[0][:, 0, :]
        vecs[i:i+len(batch)] = cls.cpu().numpy()
        if i % (BS*20) == 0: log(f"  {i}/{N}")
np.save(out, vecs)
log(f"saved rxnfp: {vecs.shape}")
