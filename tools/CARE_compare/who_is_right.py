#!/usr/bin/env python
"""
When our model and CARE DISAGREE, which is more correct?

The missing-EC reactions have no ground truth, so this cannot be answered on them directly.
Instead we run BOTH predictors on the 13,721 ModelSEED reactions that DO carry a curated EC
(ground truth), held-out, find the disagreements, and check which prediction matches truth.

  our model (rxnfp): leave-one-out 15-NN over the SAME 13,721 (ECFP4-diff||DRFP), self excluded
                     -> proper held-out prediction.
  CARE             : 15-NN over CARE's 61,766-reaction reference (DRFP). Exact duplicates
                     (top-1 cosine sim >= 0.999) are dropped as leakage; note any residual
                     near-duplicate leakage would FAVOUR CARE, so a rxnfp win is conservative.

Restricted to reactions where BOTH are confident (>=0.6 neighbour agreement). Reports, among
the disagreements: our-model-correct / CARE-correct / both-wrong, at EC1/EC2/EC3.
Env: palm.
"""
import os, time
import numpy as np, pandas as pd
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
RXC = "/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/rxnfp_clustering"
CARE = "/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/tools/CARE"
K = 15


def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def lvl(ec, n):
    return ".".join(str(ec).split(".")[:n])


def knn_majority(idx_rows, labels):
    preds, confs = [], []
    for row in idx_rows:
        c = Counter(labels[row]); top, m = c.most_common(1)[0]
        preds.append(top); confs.append(m / len(row))
    return np.array(preds), np.array(confs)


def main():
    ds = pd.read_parquet(os.path.join(RXC, "data", "dataset.parquet"))
    truth1 = ds["ec1"].to_numpy(); truth2 = ds["ec2"].to_numpy(); truth3 = ds["ec3"].to_numpy()
    N = len(ds); log(f"labeled reactions (ground truth): {N}")

    # ---- our model: LOO 15-NN over ECFP4-diff||DRFP (self excluded) ----
    Re = np.load(os.path.join(RXC, "embeddings", "ecfp4_diff.npy")).astype(np.float32)
    Rd = np.load(os.path.join(RXC, "embeddings", "drfp.npy")).astype(np.float32)
    R = normalize(np.hstack([Re, Rd]))
    nn = NearestNeighbors(n_neighbors=K + 1, metric="cosine", algorithm="brute").fit(R)
    _, idx = nn.kneighbors(R)
    idx = idx[:, 1:]                                  # drop self -> leave-one-out
    us3, usc3 = knn_majority(idx, truth3)
    us1, _ = knn_majority(idx, truth1); us2, _ = knn_majority(idx, truth2)
    log("our-model LOO done")

    # ---- CARE: 15-NN over CARE reference (DRFP), drop exact-duplicate leakage ----
    ref = pd.read_csv(os.path.join(CARE, "processed_data", "reaction2EC.csv"))
    ref = ref[ref["EC number"].str.count(r"\.") == 3].reset_index(drop=True)
    cref = normalize(np.load(os.path.join(HERE, "care_ref_drfp.npy")).astype(np.float32))
    ref_ec3 = np.array([lvl(e, 3) for e in ref["EC number"]])
    ref_ec1 = np.array([lvl(e, 1) for e in ref["EC number"]])
    ref_ec2 = np.array([lvl(e, 2) for e in ref["EC number"]])
    Qd = normalize(Rd)                                # ModelSEED DRFP as query (same encoding)
    nnc = NearestNeighbors(n_neighbors=K + 1, metric="cosine", algorithm="brute").fit(cref)
    distc, idxc = nnc.kneighbors(Qd)
    care_top1sim = 1.0 - distc[:, 0]
    # drop the nearest neighbour when it is an exact duplicate (leakage)
    leak = care_top1sim >= 0.999
    idxc_use = np.where(leak[:, None], idxc[:, 1:], idxc[:, :K])
    ca3, cac3 = knn_majority(idxc_use, ref_ec3)
    ca1, _ = knn_majority(idxc_use, ref_ec1); ca2, _ = knn_majority(idxc_use, ref_ec2)
    log(f"CARE done ({int(leak.sum())} exact-duplicate leakages dropped)")

    df = pd.DataFrame({
        "rxn_id": ds["rxn_id"], "truth_ec3": truth3,
        "us_ec3": us3, "us_conf": usc3, "care_ec3": ca3, "care_conf": cac3,
        "care_top1sim": np.round(care_top1sim, 4),
        "us_ec1": us1, "care_ec1": ca1, "us_ec2": us2, "care_ec2": ca2,
        "truth_ec1": truth1, "truth_ec2": truth2,
    })
    df.to_csv(os.path.join(HERE, "who_is_right.tsv"), sep="\t", index=False)

    # ---- adjudicate on disagreements (both confident, CARE not exact-leaked) ----
    conf = (df.us_conf >= 0.6) & (df.care_conf >= 0.6) & (df.care_top1sim < 0.999)
    print(f"\nboth-confident, non-leaked labeled reactions: {int(conf.sum())}")
    for L in ("ec1", "ec2", "ec3"):
        d = df[conf & (df[f"us_{L}"] != df[f"care_{L}"])]
        n = len(d)
        us_right = int((d[f"us_{L}"] == d[f"truth_{L}"]).sum())
        care_right = int((d[f"care_{L}"] == d[f"truth_{L}"]).sum())
        both_wrong = n - us_right - care_right
        print(f"\n  === {L.upper()} disagreements: {n} ===")
        print(f"    our model correct : {us_right:4d}  ({us_right/n*100:.0f}%)")
        print(f"    CARE correct      : {care_right:4d}  ({care_right/n*100:.0f}%)")
        print(f"    both wrong        : {both_wrong:4d}  ({both_wrong/n*100:.0f}%)")
    # overall accuracy for context (all confident, not just disagreements)
    a = df[conf]
    print("\n  overall accuracy on confident labeled set (context):")
    for L in ("ec1", "ec2", "ec3"):
        print(f"    {L.upper()}: our {100*(a[f'us_{L}']==a[f'truth_{L}']).mean():.0f}%  "
              f"CARE {100*(a[f'care_{L}']==a[f'truth_{L}']).mean():.0f}%")
    print("\nwrote who_is_right.tsv")


if __name__ == "__main__":
    main()
