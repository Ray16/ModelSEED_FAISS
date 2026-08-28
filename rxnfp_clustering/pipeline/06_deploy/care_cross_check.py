#!/usr/bin/env python
"""
Do CARE (reaction->EC) and the rxnfp_clustering FAISS retriever give SIMILAR EC predictions?

Two independent reaction->EC predictors, agreement = cross-source validation:
  rxnfp : ECFP4-diff||DRFP, k=15 majority, reference = 13,721 EC-labelled ModelSEED reactions
          (predictions already computed -> rxnfp_clustering/results/predicted_missing_ec.tsv)
  CARE  : CARE's Similarity baseline reimplemented with the SAME k=15 majority protocol,
          reference = CARE processed_data/reaction2EC.csv (61,766 curated reactions), DRFP only.

So the ONLY differences are the reference DB and the featurisation -> if they still agree on a
reaction's EC, that EC is corroborated by two independent sources. We run CARE on the SAME
ModelSEED query reactions rxnfp scored confidently, then tabulate EC1/EC2/EC3 agreement.

Env: palm (has drfp). Caches DRFP arrays so re-runs are cheap. Outputs to this folder.
"""
import os, sys, time
from multiprocessing import Pool
import numpy as np
import pandas as pd
from drfp import DrfpEncoder
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
CARE = "/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/tools/CARE"
RXNFP = "/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/rxnfp_clustering/results/predicted_missing_ec.tsv"
DIM = 2048
K = 15
NPROC = 32


def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def ec_levels(ec):
    p = str(ec).split(".")
    return ".".join(p[:1]), ".".join(p[:2]), ".".join(p[:3])


def _encode_chunk(rxns):
    # per-reaction so one bad SMILES can't drop/misalign a whole chunk
    out = np.zeros((len(rxns), DIM), dtype=np.float32)
    for i, r in enumerate(rxns):
        try:
            out[i] = np.asarray(DrfpEncoder.encode([r], n_folded_length=DIM)[0], dtype=np.float32)
        except Exception:
            pass  # leave as zeros (won't match anything under cosine)
    return out


def drfp_cached(rxns, path):
    if os.path.exists(path):
        log(f"load cached DRFP {path}")
        return np.load(path)
    rxns = list(rxns)
    log(f"DRFP encoding {len(rxns)} reactions on {NPROC} procs -> {path}")
    chunks = [rxns[i::NPROC] for i in range(NPROC)]        # round-robin split
    with Pool(NPROC) as p:
        parts = p.map(_encode_chunk, chunks)
    # reassemble in original order (round-robin inverse)
    X = np.zeros((len(rxns), DIM), dtype=np.float32)
    for ci, part in enumerate(parts):
        X[ci::NPROC] = part
    np.save(path, X)
    log(f"  done encoding {len(rxns)}")
    return X


def main():
    # ---- CARE reference ----
    ref = pd.read_csv(os.path.join(CARE, "processed_data", "reaction2EC.csv"))
    ref = ref.dropna(subset=["Reaction", "EC number"]).reset_index(drop=True)
    # keep full 4-level ECs only (comparable to rxnfp's full-EC reference)
    ref = ref[ref["EC number"].str.count(r"\.") == 3].reset_index(drop=True)
    log(f"CARE reference reactions (full EC): {len(ref)}")
    Rd = drfp_cached(ref["Reaction"].tolist(), os.path.join(HERE, "care_ref_drfp.npy"))
    R = normalize(Rd)
    ref_ec = ref["EC number"].to_numpy()

    # ---- query = rxnfp confident+in-domain predictions ----
    rx = pd.read_csv(RXNFP, sep="\t", dtype=str, keep_default_na=False)
    q = rx[(rx["confident"] == "True") & (rx["ood"] == "False")].reset_index(drop=True)
    log(f"query (rxnfp confident+in-domain): {len(q)}")
    Qd = drfp_cached(q["rxn_smiles"].tolist(), os.path.join(HERE, "care_query_drfp.npy"))
    Q = normalize(Qd)

    # ---- CARE k-NN majority (same protocol as rxnfp) ----
    log("CARE k-NN retrieval ...")
    nn = NearestNeighbors(n_neighbors=K, metric="cosine", algorithm="brute").fit(R)
    dist, idx = nn.kneighbors(Q)
    care_top1_sim = 1.0 - dist[:, 0]
    care_ec3, care_conf3 = [], []
    care_ec1, care_ec2 = [], []
    for row in idx:
        e1s = [ec_levels(ref_ec[j])[0] for j in row]
        e2s = [ec_levels(ref_ec[j])[1] for j in row]
        e3s = [ec_levels(ref_ec[j])[2] for j in row]
        t3, n3 = Counter(e3s).most_common(1)[0]
        care_ec1.append(Counter(e1s).most_common(1)[0][0])
        care_ec2.append(Counter(e2s).most_common(1)[0][0])
        care_ec3.append(t3); care_conf3.append(round(n3 / K, 3))

    out = q[["rxn_id", "pred_ec1", "pred_ec2", "pred_ec3", "conf_ec3", "top1_sim"]].copy()
    out.columns = ["rxn_id", "rxnfp_ec1", "rxnfp_ec2", "rxnfp_ec3", "rxnfp_conf3", "rxnfp_top1sim"]
    out["care_ec1"] = care_ec1; out["care_ec2"] = care_ec2
    out["care_ec3"] = care_ec3; out["care_conf3"] = care_conf3
    out["care_top1sim"] = np.round(care_top1_sim, 4)
    out["agree_ec1"] = out["rxnfp_ec1"] == out["care_ec1"]
    out["agree_ec2"] = out["rxnfp_ec2"] == out["care_ec2"]
    out["agree_ec3"] = out["rxnfp_ec3"] == out["care_ec3"]
    out.to_csv(os.path.join(HERE, "care_vs_rxnfp.tsv"), sep="\t", index=False)

    N = len(out)
    care_conf = out["care_conf3"].astype(float) >= 0.6
    both_conf = care_conf & (out["care_top1sim"] >= 0.3)
    log("=" * 64)
    log(f"reactions compared: {N}")
    for lvl in ("ec1", "ec2", "ec3"):
        a = out[f"agree_{lvl}"].mean() * 100
        ab = out.loc[both_conf, f"agree_{lvl}"].mean() * 100
        log(f"  agreement {lvl.upper()}: all={a:.1f}%   CARE-confident subset={ab:.1f}% "
            f"(n={int(both_conf.sum())})")
    log(f"  mean CARE top-1 similarity to its own reference: {out['care_top1sim'].mean():.3f}")
    log(f"  (CARE conf>=0.6 on {int(care_conf.sum())}/{N} = {care_conf.mean()*100:.0f}%)")
    log("wrote care_vs_rxnfp.tsv")


if __name__ == "__main__":
    main()
