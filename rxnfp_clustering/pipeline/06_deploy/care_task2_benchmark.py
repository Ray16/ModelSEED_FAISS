#!/usr/bin/env python
"""
Fair external benchmark: our reaction->EC retriever vs CARE's task-2 baselines,
graded on CARE's OWN Rhea/SwissProt-derived ground truth (task2 easy/medium/hard
reaction-test splits). This removes the "home-DB advantage" that inflates the
in-house numbers in RESULTS.md Part VIII: here the ground truth is *not* ModelSEED
curation, and CARE's reference is closer to it than ours.

Two reference indexes for our retriever (same ECFP4-diff||DRFP featurisation, same
similarity-ranking protocol as CARE's Similarity baseline):

  (A) CARE-train  : retrieve from CARE's own per-split train reactions.
                    Isolates the *featurisation* (our ECFP4-diff||DRFP vs CARE's
                    DRFP-only) with the reference held equal  -> "is the method better?"
  (B) ModelSEED   : retrieve from our 13,721 EC-labelled ModelSEED reactions.
                    Tests our *actual deployed tool* off its home field  -> "is the
                    deliverable's quality real on external ground truth?"

Metric == CARE's: each method emits a ranked EC list per reaction; top-k accuracy
is whether the true EC (reduced to the scored level) appears in the first k. We
score full-EC / EC3 / EC2 / EC1 at k in {1,3,5,10}, exactly as applied to CARE's
precomputed baseline result frames.

Outputs:
  results/care_task2_benchmark.csv   (long: method x split x reference x level x k -> acc)
Run:  PY=/homes/rzhu/miniforge3/envs/palm/bin/python ; $PY pipeline/06_deploy/care_task2_benchmark.py
"""
import os, sys, time, warnings
# cap BLAS/OpenMP threads BEFORE numpy import — the un-capped SVD spawned ~140
# threads and thrashed on this shared node (5x slowdown). Must precede numpy.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")
import numpy as np, pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator as fpg
from rdkit import RDLogger
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))          # rxnfp_clustering/
CARE = os.path.abspath(os.path.join(ROOT, "..", "tools", "CARE"))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
SPLITS = os.path.join(CARE, "splits", "task2")
BASE = os.path.join(CARE, "task2_baselines", "results_summary")

SPLITNAMES = ["easy", "medium", "hard"]
BASELINES = ["Similarity", "CREEP", "CLIPZyme", "ChatGPT", "random"]
KS = [1, 3, 5, 10]
LEVELS = ["EC number", "EC3", "EC2", "EC1"]   # column names in the CARE frames
LEVEL_TAG = {"EC number": "full", "EC3": "ec3", "EC2": "ec2", "EC1": "ec1"}


def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)  # stdout, flushed


# ---------------- featurisation (mirrors pipeline/01_embed ecfp4_drfp) ----------
_morgan = fpg.GetMorganGenerator(radius=2, fpSize=2048)
_mol_cache = {}
def _mol(s):
    if s not in _mol_cache:
        _mol_cache[s] = Chem.MolFromSmiles(s)
    return _mol_cache[s]

def _count_vec(smis):
    v = np.zeros(2048, dtype=np.float32)
    for s in smis:
        m = _mol(s)
        if m is None: continue
        v += _morgan.GetCountFingerprintAsNumPy(m).astype(np.float32)
    return v

def ecfp4_diff(rxns):
    X = np.zeros((len(rxns), 2048), dtype=np.float32)
    for i, r in enumerate(rxns):
        L, R = r.split(">>")
        X[i] = _count_vec(R.split(".")) - _count_vec(L.split("."))
    return X

def drfp_mat(rxns):
    from drfp import DrfpEncoder
    return np.array(DrfpEncoder.encode(list(rxns), n_folded_length=2048), dtype=np.float32)

def _featurise_chunk(rxns):
    """Worker: ECFP4 count-diff (2048) || DRFP (2048) for one chunk. Module-level
    so it is picklable for ProcessPoolExecutor."""
    return np.hstack([ecfp4_diff(rxns), drfp_mat(rxns)]).astype(np.float32)

def featurise(rxns, workers=32, chunk=400):
    """ECFP4 count-diff || DRFP = 4096 (the RESULTS.md best embedding), parallelised
    across processes — DRFP is pure-Python single-threaded and the dominant cost on
    the ~60k-reaction CARE train references."""
    rxns = list(rxns)
    if len(rxns) <= chunk:
        return _featurise_chunk(rxns)
    from concurrent.futures import ProcessPoolExecutor
    chunks = [rxns[i:i + chunk] for i in range(0, len(rxns), chunk)]
    parts, done = [None] * len(chunks), 0
    with ProcessPoolExecutor(max_workers=min(workers, len(chunks))) as ex:
        futs = {ex.submit(_featurise_chunk, c): i for i, c in enumerate(chunks)}
        from concurrent.futures import as_completed
        for f in as_completed(futs):
            parts[futs[f]] = f.result(); done += 1
            if done % 20 == 0 or done == len(chunks):
                log(f"    featurise {done}/{len(chunks)} chunks ({done*chunk}/{len(rxns)} rxns)")
    return np.vstack(parts)


# ---------------- similarity ranking (our retriever == a Similarity baseline) ----
def fit_space(Xref, n_comp=256):
    """Faithful similarity baseline: L2-normalise the full 4096-d ECFP4-diff||DRFP
    feature vector (cosine geometry), NO SVD compression -- this matches CARE's
    Similarity (full-dim DRFP cosine) so the only difference vs CARE is the
    featurisation itself. (Our deployed FAISS pipeline additionally SVD-256s for
    speed; that is a deployment detail, not part of the method comparison.)"""
    def project(X):
        X = X.copy().astype(np.float32); X[~np.isfinite(X)] = 0.0
        return normalize(X)
    return project

def build_index(Xref, ref_ec, pool=500):
    """Fit the projection + cosine NN index on a reference ONCE, return a query
    closure. Building once (not per split) avoids refitting the 13,721x4096 SVD
    three times over. `query(Xq)` -> list-of-ranked-full-EC per query, ranking EC
    labels by summed cosine similarity of the retrieved neighbours (== CARE's
    Similarity protocol)."""
    proj = fit_space(Xref)
    Zref = proj(Xref)
    k = min(pool, len(Zref))
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", n_jobs=8).fit(Zref)
    ref_ec = np.asarray(ref_ec)
    def query(Xq):
        dist, idx = nn.kneighbors(proj(Xq))
        sim = 1.0 - dist
        out = []
        for qi in range(len(idx)):
            # rank EC labels by the MAX similarity of any neighbour of that class
            # (== CARE's Similarity protocol: nearest reaction per class). Summing
            # instead lets frequent classes dominate top-1 on a large reference.
            agg = {}
            for j, ni in enumerate(idx[qi]):
                ec = ref_ec[ni]; s = sim[qi, j]
                if s > agg.get(ec, -1.0):
                    agg[ec] = s
            out.append([ec for ec, _ in sorted(agg.items(), key=lambda t: -t[1])])
        return out
    return query


# ---------------- scoring (identical for our method and CARE frames) ------------
def reduce_level(ec, level):
    p = str(ec).split(".")
    if level == "EC number": return str(ec)
    if level == "EC3": return ".".join(p[:3])
    if level == "EC2": return ".".join(p[:2])
    if level == "EC1": return p[0]
    raise ValueError(level)

def topk_from_ranked(truth_full, ranked_full, level, k):
    """truth_full/ranked_full are full-EC strings; reduce both to `level`, dedupe
    the ranked list preserving order, check if truth is in the first k."""
    t = reduce_level(truth_full, level)
    seen, red = set(), []
    for e in ranked_full:
        r = reduce_level(e, level)
        if r not in seen:
            seen.add(r); red.append(r)
        if len(red) >= k: break
    return t in red[:k]

def score_ranked(truths_full, ranked_lists):
    rows = {}
    for level in LEVELS:
        for k in KS:
            hits = sum(topk_from_ranked(t, r, level, k)
                       for t, r in zip(truths_full, ranked_lists))
            rows[(LEVEL_TAG[level], k)] = hits / len(truths_full)
    return rows

def care_ranked_lists(df):
    """CARE result frames store the per-reaction ranked full-EC list in the
    integer-named columns 0,1,2,... (already full-EC strings)."""
    rc = [c for c in df.columns if str(c).isdigit()]
    rc = sorted(rc, key=lambda c: int(c))
    return [[str(v) for v in row if pd.notna(v)]
            for row in df[rc].to_numpy()]


# ---------------- data loaders --------------------------------------------------
def load_split(split, which):
    df = pd.read_csv(os.path.join(SPLITS, f"{split}_reaction_{which}.csv"))
    df = df[df["Reaction"].apply(lambda r: isinstance(r, str) and ">>" in r)].reset_index(drop=True)
    return df

def load_modelseed_ref():
    df = pd.read_parquet(os.path.join(DATA, "dataset.parquet"))
    return df["rxn_smiles"].tolist(), df["ec"].astype(str).to_numpy()


# ---------------- main ----------------------------------------------------------
def main():
    out = []
    # ModelSEED reference (shared across splits) -- featurise + index ONCE
    log("featurising ModelSEED reference (13,721) [DRFP is the slow step]...")
    ms_rxn, ms_ec = load_modelseed_ref()
    Xms = featurise(ms_rxn)
    log("building ModelSEED index (SVD+NN) once...")
    ms_query = build_index(Xms, ms_ec)
    log("ModelSEED index ready.")

    for split in SPLITNAMES:
        test = load_split(split, "test")
        train = load_split(split, "train")
        truths = test["EC number"].astype(str).tolist()
        log(f"[{split}] test={len(test)} train={len(train)}  featurising test+train...")

        Xte = featurise(test["Reaction"].tolist())

        # our method, reference (A) = CARE per-split train
        Xtr = featurise(train["Reaction"].tolist())
        rankedA = build_index(Xtr, train["EC number"].astype(str).to_numpy())(Xte)
        sA = score_ranked(truths, rankedA)
        for (tag, k), acc in sA.items():
            out.append(dict(method="ours", reference="CARE-train", split=split,
                            level=tag, k=k, acc=round(acc, 4), n=len(test)))

        # our method, reference (B) = ModelSEED 13,721 (index reused)
        rankedB = ms_query(Xte)
        sB = score_ranked(truths, rankedB)
        for (tag, k), acc in sB.items():
            out.append(dict(method="ours", reference="ModelSEED", split=split,
                            level=tag, k=k, acc=round(acc, 4), n=len(test)))
        log(f"[{split}] ours done (A ec3@1={sA[('ec3',1)]:.3f}, B ec3@1={sB[('ec3',1)]:.3f})")

        # CARE baselines, same metric
        for b in BASELINES:
            p = os.path.join(BASE, b, f"{split}_reaction_test_results_df.csv")
            if not os.path.exists(p):
                log(f"  (missing baseline {b} for {split})"); continue
            bdf = pd.read_csv(p)
            # align to same truths order/parseable subset
            bdf = bdf[bdf["Reaction"].isin(set(test["Reaction"]))].reset_index(drop=True)
            btruth = bdf["EC number"].astype(str).tolist()
            branked = care_ranked_lists(bdf)
            for (tag, k), acc in score_ranked(btruth, branked).items():
                out.append(dict(method=b, reference="CARE-own", split=split,
                                level=tag, k=k, acc=round(acc, 4), n=len(bdf)))
            log(f"  baseline {b}: ec3@1={score_ranked(btruth, branked)[('ec3',1)]:.3f}")

    res = pd.DataFrame(out)
    os.makedirs(RESULTS, exist_ok=True)
    fp = os.path.join(RESULTS, "care_task2_benchmark.csv")
    res.to_csv(fp, index=False)
    log(f"wrote {fp} ({len(res)} rows)")

    # headline pivot: EC3 top-1 accuracy, method x split
    piv = (res[(res.level == "ec3") & (res.k == 1)]
           .assign(m=lambda d: d.method + " [" + d.reference + "]")
           .pivot_table(index="m", columns="split", values="acc"))
    print("\n=== EC3 top-1 accuracy (fair external ground truth) ===")
    print(piv.reindex(columns=SPLITNAMES).round(3).to_string())


if __name__ == "__main__":
    main()
