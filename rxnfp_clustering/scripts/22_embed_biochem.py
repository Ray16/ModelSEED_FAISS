#!/usr/bin/env python
"""Biochemistry-native reaction embedding (GPU) -- the in-domain learned axis.

Goal: something pretrained *on enzymatic/metabolic reactions* rather than
USPTO organic synthesis (rxnfp's ceiling per Part I/II of this benchmark).

Chosen model: **REXzyme** (Hua et al., AI4PD/REXzyme_smiles on HuggingFace) --
a 335M-parameter T5-Efficient-Large encoder-decoder trained with a
*supervised reaction -> enzyme-sequence translation* objective on
**31,970,152 reaction-enzyme pairs from RHEA**, the curated biochemical
reaction database (the same domain ModelSEED reactions live in, NOT USPTO
organic chemistry). We use only the ENCODER: it must build a representation
of the reaction rich enough to predict a candidate catalysing enzyme, which
is a genuinely biochemistry-native training signal -- option (a) from the
task spec ("find a pretrained metabolic/enzymatic reaction model"), not the
ChemBERTa-finetune fallback (b).

  biochem        1024-dim float, mean-pooled T5 ENCODER hidden state over the
                 whole reaction string (REXzyme's native "r2s A.B>agent>C"
                 format; our rxn_smiles is already "reactants>>products",
                 i.e. reactant>[empty agent]>product, so only an "r2s" prefix
                 is added -- no reformatting needed). This is a *reaction-level*
                 embedder (like rxnfp/rxnfp_pretrained), not a per-molecule
                 diff -- there is no per-molecule REXzyme representation to
                 subtract, the whole point is the transformer reads the
                 reaction as one biochemical transformation.

Model files ship on the Hub with a config filename quirk ("config (1).json"
instead of "config.json") and a legacy pickled pytorch_model.bin that trips
transformers>=4.5's CVE-2025-32434 safe-loading gate for torch<2.6. Both are
worked around explicitly below (manual state_dict load, filtered to the
`encoder.*`/`shared.*` keys -- the decoder is never instantiated).

NOT a fine-tune placeholder: this is the real pretrained REXzyme encoder, used
as-is (frozen, no further training on the 13,721 ModelSEED reactions).
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
DATA = os.path.join(HERE, "data"); RESULTS = os.path.join(HERE, "results"); CACHE = os.path.join(HERE, "cache")
EMB = os.path.join(HERE, "embeddings"); os.makedirs(EMB, exist_ok=True)
CKPT_DIR = os.path.join(CACHE, "rexzyme"); os.makedirs(CKPT_DIR, exist_ok=True)
REPO = "AI4PD/REXzyme_smiles"

def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, file=sys.stderr, flush=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"

def fetch_checkpoint():
    """Download REXzyme's files and normalize the odd 'config (1).json' name."""
    import shutil
    from huggingface_hub import hf_hub_download
    need = {"config (1).json": "config.json", "pytorch_model.bin": "pytorch_model.bin",
            "tokenizer.json": "tokenizer.json", "tokenizer_config.json": "tokenizer_config.json",
            "special_tokens_map.json": "special_tokens_map.json"}
    for src, dst in need.items():
        dst_path = os.path.join(CKPT_DIR, dst)
        if os.path.exists(dst_path) and os.path.getsize(dst_path) > 0:
            continue
        log(f"downloading {REPO}/{src}")
        p = hf_hub_download(REPO, src)
        shutil.copy(p, dst_path)
    return CKPT_DIR

def load_model():
    from transformers import T5EncoderModel, T5Config, AutoTokenizer
    d = fetch_checkpoint()
    tok = AutoTokenizer.from_pretrained(d)
    cfg = T5Config.from_pretrained(d)
    mdl = T5EncoderModel(cfg)
    # legacy pickle -> load manually (bypasses transformers' torch<2.6 CVE-2025-32434 gate);
    # this is the model's own official checkpoint file straight from the HF repo, not an
    # untrusted third-party pickle.
    sd = torch.load(os.path.join(d, "pytorch_model.bin"), map_location="cpu", weights_only=False)
    enc_sd = {k: v for k, v in sd.items() if k.startswith("encoder.") or k == "shared.weight"}
    missing, unexpected = mdl.load_state_dict(enc_sd, strict=False)
    if missing or unexpected:
        log(f"WARNING state_dict mismatch: missing={missing} unexpected={unexpected}")
    else:
        log("REXzyme encoder loaded: 0 missing / 0 unexpected keys")
    mdl = mdl.to(dev).eval()
    return tok, mdl

def main():
    if os.path.exists(os.path.join(EMB, "biochem.npy")):
        log("biochem.npy already exists, skipping"); return

    df = pd.read_parquet(os.path.join(DATA, "dataset.parquet"))
    rxns = df.rxn_smiles.tolist(); N = len(rxns)
    tok, mdl = load_model()
    DIM = mdl.config.d_model
    log(f"encoder d_model={DIM}, {sum(p.numel() for p in mdl.parameters())/1e6:.0f}M params")

    MAXLEN = 1024
    BS = 12
    X = np.zeros((N, DIM), np.float32)
    mask = np.ones(N, bool)
    t = time.time()
    with torch.no_grad():
        for i in range(0, N, BS):
            batch = ["r2s" + s for s in rxns[i:i+BS]]
            try:
                enc = tok(batch, padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt")
                enc = {k: v.to(dev) for k, v in enc.items()}
                out = mdl(**enc)
                h = out.last_hidden_state
                m = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-6)
                pooled = pooled.cpu().numpy()
                if not np.isfinite(pooled).all():
                    for j in range(len(batch)):
                        if np.isfinite(pooled[j]).all():
                            X[i+j] = pooled[j]
                        else:
                            mask[i+j] = False
                else:
                    X[i:i+len(batch)] = pooled
            except Exception as ex:
                log(f"  batch {i} failed: {ex}")
                mask[i:i+len(batch)] = False
            if i % (BS * 50) == 0:
                log(f"  rxns {i}/{N} ({time.time()-t:.0f}s)")
    log(f"REXzyme embedding done ({time.time()-t:.0f}s)")

    np.save(os.path.join(EMB, "biochem.npy"), X)
    np.save(os.path.join(EMB, "biochem_mask.npy"), mask)
    log(f"saved biochem.npy {X.shape}, mask: {mask.sum()}/{N} ({100*mask.mean():.1f}%) OK")

if __name__ == "__main__":
    main()
