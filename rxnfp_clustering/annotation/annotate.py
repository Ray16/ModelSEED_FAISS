#!/usr/bin/env python
"""EC-annotation transfer — beyond fixed-k majority vote.

Motivation: fixed-k equal-weight kNN uses neighbour RANK, not SIMILARITY. In a
sparse region a query still gets k neighbours, but the far ones (cosine sim ~0.2)
vote with the same weight as a near-identical reaction, so the majority is
polluted and we emit a confident-but-wrong EC. This module implements and
compares better rules, evaluated under a REFERENCE/TEST split (neighbours drawn
only from the reference pool) so that "sparse" queries — whose true EC has few or
zero reference examples — actually occur and are measured, unlike leave-one-out.

Methods compared (all share one precomputed k=50 neighbour list):
  M1 baseline      fixed k=15, equal-weight majority (reproduces scripts/08)
  M2 weighted      k=15, similarity-weighted vote
  M3 threshold     radius tau retrieval + weighted vote + ABSTAIN if none >= tau
  M4 backoff       report the finest EC level (EC3->EC2->EC1) with weighted
                   confidence >= theta; else abstain  (EC-tree-native)
  M5 conformal     split-conformal EC3 prediction SETS with 1-alpha coverage
  M6 ood           top-1 distance / LOF novelty score; AUROC for error, and the
                   accuracy gained by abstaining on the most-novel queries

Outputs (self-contained, does not touch the main benchmark):
  annotation/results/*.csv   annotation/figures/*.png   (all figures 18pt)

Env: palm  (/homes/rzhu/miniforge3/envs/palm/bin/python)
"""
import os, sys, time, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from collections import defaultdict
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
DATA = os.path.join(HERE, "data"); EMB = os.path.join(HERE, "embeddings")
ADIR = os.path.join(HERE, "annotation")
ARES = os.path.join(ADIR, "results"); AFIG = os.path.join(ADIR, "figures")
os.makedirs(ARES, exist_ok=True); os.makedirs(AFIG, exist_ok=True)
sys.path.insert(0, os.path.join(HERE, "scripts"))
import _style  # project rule: 18pt fonts, dpi 300
import matplotlib.pyplot as plt
FS = _style.FS

def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, file=sys.stderr, flush=True)

CONTINUOUS = {"rxnfp", "rxnfp_pretrained", "chemberta_diff", "descriptors_diff",
              "molformer_diff", "rxnfp_moldiff", "rxnfp_sidediff"}

def make_space(name, Xraw):
    """Identical geometry to the clustering/kNN benchmark: L2 -> SVD256 -> L2."""
    X = StandardScaler().fit_transform(Xraw) if name in CONTINUOUS else Xraw.astype(np.float32)
    X[~np.isfinite(X)] = 0.0
    X = normalize(X)
    d = min(256, X.shape[1] - 1)
    Z = TruncatedSVD(n_components=d, random_state=0).fit_transform(X)
    z = np.linalg.norm(Z, axis=1) < 1e-9
    if z.any(): Z[z] = 0.0; Z[z, 0] = 1.0
    return normalize(Z).astype(np.float32)

# ----------------------------- core vote helpers -----------------------------
def weighted_vote(labels_row, w_row):
    """Return (best_label, best_frac, total_weight) for one query."""
    agg = defaultdict(float)
    for lab, w in zip(labels_row, w_row):
        agg[lab] += w
    tot = sum(agg.values())
    if tot <= 0:
        return labels_row[0], 0.0, 0.0
    best = max(agg, key=agg.get)
    return best, agg[best] / tot, tot

def acc_coverage_curve(conf, correct, answered=None, n_points=101):
    """Accuracy as a function of coverage: rank by conf desc, answer the top c."""
    if answered is None:
        answered = np.ones(len(conf), bool)
    order = np.argsort(-conf)
    order = order[answered[order]]                # only rank answerable queries
    corr = correct[order].astype(float)
    n_total = len(conf)
    covs, accs = [], []
    for frac in np.linspace(1.0 / max(len(order), 1), 1.0, n_points):
        m = max(1, int(round(frac * len(order))))
        covs.append(m / n_total)                   # coverage over ALL queries
        accs.append(corr[:m].mean())
    return np.array(covs), np.array(accs)

def acc_at_coverage(conf, correct, answered, target_cov):
    covs, accs = acc_coverage_curve(conf, correct, answered)
    j = np.argmin(np.abs(covs - target_cov))
    return accs[j], covs[j]

def cov_at_accuracy(conf, correct, answered, target_acc):
    covs, accs = acc_coverage_curve(conf, correct, answered)
    ok = np.where(accs >= target_acc)[0]
    return covs[ok].max() if len(ok) else 0.0

# --------------------------------- main --------------------------------------
def run(emb, seed=0, test_frac=0.2, cal_frac=0.2, K=15, KMAX=50):
    df = pd.read_parquet(os.path.join(DATA, "dataset.parquet"))
    N = len(df)
    LEVELS = ("ec3", "ec2", "ec1")
    LAB = {lvl: df[lvl].to_numpy() for lvl in ("ec", "ec3", "ec2", "ec1")}
    Z = make_space(emb, np.load(os.path.join(EMB, emb + ".npy")))

    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    n_test = int(test_frac * N)
    test_idx = np.sort(perm[:n_test]); ref_idx = np.sort(perm[n_test:])
    log(f"{emb}: reference={len(ref_idx)}  test={len(test_idx)}")

    nn = NearestNeighbors(n_neighbors=KMAX, metric="cosine", n_jobs=8).fit(Z[ref_idx])
    dist, nbr = nn.kneighbors(Z[test_idx])         # (n_test, KMAX)
    sim = np.clip(1.0 - dist, 0.0, None)           # cosine similarity, >=0
    # map neighbour positions (into ref) -> global -> labels per level
    nbr_glob = ref_idx[nbr]
    LN = {lvl: LAB[lvl][nbr_glob] for lvl in LEVELS}   # (n_test, KMAX)
    yt = {lvl: LAB[lvl][test_idx] for lvl in LEVELS}
    nT = len(test_idx)

    summary = []      # rows for method_comparison.csv
    curves = {}       # method -> (cov, acc) at EC3 for the overlay figure

    # ---- M1 baseline: fixed k, equal weight, EC3 ----
    predm, confm, corrm = [], [], []
    for i in range(nT):
        b, f, _ = weighted_vote(LN["ec3"][i, :K], np.ones(K))
        predm.append(b); confm.append(f)
    predm = np.array(predm); confm = np.array(confm); corrm = predm == yt["ec3"]
    ans = np.ones(nT, bool)
    curves["M1 baseline (k=15)"] = acc_coverage_curve(confm, corrm, ans)
    summary.append(dict(method="M1 baseline equal-weight", level="EC3",
        acc_full=corrm.mean(),
        acc_at75=acc_at_coverage(confm, corrm, ans, 0.75)[0],
        cov_at90=cov_at_accuracy(confm, corrm, ans, 0.90), coverage=1.0))

    # ---- M2 similarity-weighted, fixed k, EC3 ----
    predw, confw = [], []
    for i in range(nT):
        b, f, _ = weighted_vote(LN["ec3"][i, :K], sim[i, :K])
        predw.append(b); confw.append(f)
    predw = np.array(predw); confw = np.array(confw); corrw = predw == yt["ec3"]
    curves["M2 sim-weighted (k=15)"] = acc_coverage_curve(confw, corrw, ans)
    summary.append(dict(method="M2 similarity-weighted", level="EC3",
        acc_full=corrw.mean(),
        acc_at75=acc_at_coverage(confw, corrw, ans, 0.75)[0],
        cov_at90=cov_at_accuracy(confw, corrw, ans, 0.90), coverage=1.0))

    # ---- M3 threshold + abstain (sweep tau), EC3 ----
    tau_rows = []
    for tau in np.round(np.arange(0.30, 0.96, 0.05), 2):
        preds = np.empty(nT, dtype=object); conf = np.zeros(nT); answ = np.zeros(nT, bool)
        for i in range(nT):
            keep = sim[i] >= tau
            if not keep.any():
                continue
            b, f, _ = weighted_vote(LN["ec3"][i][keep], sim[i][keep])
            preds[i] = b; conf[i] = f; answ[i] = True
        corr = np.array([answ[i] and preds[i] == yt["ec3"][i] for i in range(nT)])
        cov = answ.mean()
        acc_ans = corr[answ].mean() if answ.any() else 0.0
        tau_rows.append(dict(tau=tau, coverage=round(cov, 4), acc_answered=round(acc_ans, 4)))
    pd.DataFrame(tau_rows).to_csv(os.path.join(ARES, f"m3_threshold_{emb}.csv"), index=False)
    # representative operating point: highest tau with coverage >= 0.75
    ok = [r for r in tau_rows if r["coverage"] >= 0.75]
    rep = ok[-1] if ok else tau_rows[0]
    # build a conf/answered vector for the curve at a moderate tau (0.5)
    tau0 = 0.5; predsT = np.empty(nT, dtype=object); confT = np.full(nT, -1.0); answT = np.zeros(nT, bool)
    for i in range(nT):
        keep = sim[i] >= tau0
        if keep.any():
            b, f, _ = weighted_vote(LN["ec3"][i][keep], sim[i][keep])
            predsT[i] = b; confT[i] = f; answT[i] = True
    corrT = np.array([answT[i] and predsT[i] == yt["ec3"][i] for i in range(nT)])
    curves["M3 threshold+abstain (tau=0.5)"] = acc_coverage_curve(confT, corrT, answT)
    summary.append(dict(method=f"M3 threshold+abstain (tau={rep['tau']})", level="EC3",
        acc_full=np.nan, acc_at75=rep["acc_answered"], cov_at90=np.nan,
        coverage=rep["coverage"]))

    # ---- M4 hierarchical back-off ----
    THETA = 0.5
    finest = np.empty(nT, dtype=object)     # level called, or "abstain"
    called_lab = np.empty(nT, dtype=object)
    correct_bo = np.zeros(nT, bool)
    for i in range(nT):
        placed = False
        for lvl in LEVELS:                  # EC3 -> EC2 -> EC1
            b, f, _ = weighted_vote(LN[lvl][i, :K], sim[i, :K])
            if f >= THETA:
                finest[i] = lvl; called_lab[i] = b
                correct_bo[i] = (b == yt[lvl][i]); placed = True; break
        if not placed:
            finest[i] = "abstain"; correct_bo[i] = False
    lvl_counts = {lvl: int((finest == lvl).sum()) for lvl in LEVELS}
    lvl_counts["abstain"] = int((finest == "abstain").sum())
    lvl_acc = {}
    for lvl in LEVELS:
        m = finest == lvl
        lvl_acc[lvl] = float(correct_bo[m].mean()) if m.any() else 0.0
    answered_bo = finest != "abstain"
    pd.DataFrame([dict(level=k, n=v, frac=round(v/nT, 4),
                       acc=round(lvl_acc.get(k, np.nan), 4) if k in lvl_acc else np.nan)
                  for k, v in lvl_counts.items()]).to_csv(
        os.path.join(ARES, f"m4_backoff_{emb}.csv"), index=False)
    summary.append(dict(method=f"M4 hierarchical back-off (theta={THETA})", level="finest",
        acc_full=correct_bo.mean(),                    # correct-at-reported-level over ALL
        acc_at75=correct_bo[answered_bo].mean() if answered_bo.any() else 0.0,
        cov_at90=np.nan, coverage=answered_bo.mean()))

    # ---- M5 split-conformal EC3 prediction sets ----
    # carve a calibration slice out of the reference for nonconformity quantile
    rng2 = np.random.default_rng(seed + 1)
    rp = rng2.permutation(len(ref_idx)); n_cal = int(cal_frac * len(ref_idx))
    cal_local = np.sort(rp[:n_cal]); pool_local = np.sort(rp[n_cal:])
    pool_glob = ref_idx[pool_local]
    nn_c = NearestNeighbors(n_neighbors=K, metric="cosine", n_jobs=8).fit(Z[pool_glob])

    def softvote_ec3(Zq):
        d, nb = nn_c.kneighbors(Zq); s = np.clip(1 - d, 0, None)
        labs = LAB["ec3"][pool_glob[nb]]
        out = []
        for i in range(len(Zq)):
            agg = defaultdict(float)
            for lab, w in zip(labs[i], s[i]): agg[lab] += w
            tot = sum(agg.values()) or 1.0
            out.append({k: v/tot for k, v in agg.items()})
        return out

    cal_glob = ref_idx[cal_local]
    cal_probs = softvote_ec3(Z[cal_glob]); cal_y = LAB["ec3"][cal_glob]
    alpha = 0.1
    scores = np.array([1.0 - cal_probs[i].get(cal_y[i], 0.0) for i in range(len(cal_glob))])
    qhat = np.quantile(scores, min(1.0, (1 - alpha) * (1 + 1/len(scores))), method="higher")
    test_probs = softvote_ec3(Z[test_idx])
    set_sizes = np.zeros(nT, int); covered = np.zeros(nT, bool)
    for i in range(nT):
        pset = [c for c, p in test_probs[i].items() if (1 - p) <= qhat]
        set_sizes[i] = len(pset); covered[i] = yt["ec3"][i] in pset
    pd.DataFrame([dict(alpha=alpha, qhat=round(float(qhat), 4),
                       empirical_coverage=round(covered.mean(), 4),
                       mean_set_size=round(set_sizes.mean(), 3),
                       median_set_size=int(np.median(set_sizes)),
                       frac_singleton=round((set_sizes == 1).mean(), 4),
                       frac_empty=round((set_sizes == 0).mean(), 4))]).to_csv(
        os.path.join(ARES, f"m5_conformal_{emb}.csv"), index=False)
    summary.append(dict(method=f"M5 conformal set (alpha={alpha})", level="EC3-set",
        acc_full=covered.mean(), acc_at75=np.nan, cov_at90=np.nan,
        coverage=1.0 - (set_sizes == 0).mean()))

    # ---- M6 OOD / novelty ----
    top1 = sim[:, 0]                         # higher = closer = less novel
    ood_dist = 1.0 - top1                    # higher = more novel
    lof = LocalOutlierFactor(n_neighbors=20, metric="cosine", novelty=True).fit(Z[ref_idx])
    lof_score = -lof.score_samples(Z[test_idx])   # higher = more outlier/novel
    err = ~corrw                              # M2 error as the target
    auroc_dist = roc_auc_score(err, ood_dist) if err.any() and (~err).any() else np.nan
    auroc_lof = roc_auc_score(err, lof_score) if err.any() and (~err).any() else np.nan
    # accuracy if we abstain on the most-novel X% (by top-1 distance)
    order = np.argsort(ood_dist)             # least novel first
    keep_curve = []
    for frac in np.linspace(0.1, 1.0, 10):
        m = order[:int(frac * nT)]
        keep_curve.append(dict(coverage=round(frac, 2), acc=round(corrw[m].mean(), 4)))
    pd.DataFrame(keep_curve).to_csv(os.path.join(ARES, f"m6_ood_{emb}.csv"), index=False)
    summary.append(dict(method="M6 OOD-abstain (top-1 dist)", level="EC3",
        acc_full=corrw.mean(),
        acc_at75=next(r["acc"] for r in keep_curve if abs(r["coverage"]-0.7) < 0.06),
        cov_at90=np.nan, coverage=0.7,
        auroc_dist=round(auroc_dist, 4), auroc_lof=round(auroc_lof, 4)))

    sdf = pd.DataFrame(summary)
    sdf.insert(0, "embedding", emb)
    sdf.to_csv(os.path.join(ARES, f"method_comparison_{emb}.csv"), index=False)
    log("method comparison:\n" + sdf.to_string(index=False))

    _figures(emb, curves, tau_rows, lvl_counts, lvl_acc, nT, set_sizes, covered,
             keep_curve, auroc_dist, auroc_lof)
    return sdf

# --------------------------------- figures -----------------------------------
def _figures(emb, curves, tau_rows, lvl_counts, lvl_acc, nT, set_sizes, covered,
             keep_curve, auroc_dist, auroc_lof):
    # Fig A: accuracy vs coverage overlay (M1-M3)
    fig, ax = plt.subplots(figsize=(12, 8))
    for name, (cov, acc) in curves.items():
        ax.plot(cov, acc, lw=2.6, label=name)
    ax.set_xlabel("coverage (fraction of queries answered)", fontsize=FS)
    ax.set_ylabel("EC3 accuracy (answered)", fontsize=FS)
    ax.set_title(f"Accuracy vs coverage — annotation rules ({emb})", fontsize=FS)
    ax.grid(alpha=0.3); ax.tick_params(labelsize=FS)
    ax.legend(fontsize=FS, frameon=False, loc="lower left")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(AFIG, f"fig_accuracy_coverage_{emb}.png"),
                                    dpi=300, bbox_inches="tight"); plt.close(fig)

    # Fig B: M3 threshold sweep — coverage & accuracy vs tau
    td = pd.DataFrame(tau_rows)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(td.tau, td.acc_answered, "o-", lw=2.6, ms=8, color="#0072B2", label="accuracy (answered)")
    ax.plot(td.tau, td.coverage, "s-", lw=2.6, ms=8, color="#D55E00", label="coverage")
    ax.set_xlabel("similarity threshold tau", fontsize=FS); ax.set_ylabel("score", fontsize=FS)
    ax.set_title(f"Threshold retrieval + abstain — accuracy/coverage tradeoff ({emb})", fontsize=FS)
    ax.grid(alpha=0.3); ax.tick_params(labelsize=FS)
    ax.legend(fontsize=FS, frameon=False, loc="center left")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(AFIG, f"fig_threshold_sweep_{emb}.png"),
                                    dpi=300, bbox_inches="tight"); plt.close(fig)

    # Fig C: hierarchical back-off — where calls land + per-level accuracy
    order = ["ec3", "ec2", "ec1", "abstain"]
    counts = [lvl_counts[k] for k in order]
    fracs = [c / nT for c in counts]
    accs = [lvl_acc.get(k, 0.0) for k in order]
    x = np.arange(len(order)); w = 0.38
    fig, ax = plt.subplots(figsize=(12, 8))
    b1 = ax.bar(x - w/2, fracs, w, color="#0072B2", label="fraction of queries")
    b2 = ax.bar(x + w/2, accs, w, color="#009E73", label="accuracy at that level")
    for b, v in zip(b1, fracs): ax.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.2f}", ha="center", fontsize=FS)
    for b, v in zip(b2, accs): ax.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.2f}" if v else "-", ha="center", fontsize=FS)
    ax.set_xticks(x); ax.set_xticklabels(["EC3", "EC2", "EC1", "abstain"], fontsize=FS)
    ax.set_ylim(0, 1.05); ax.set_ylabel("score", fontsize=FS)
    ax.set_title(f"Hierarchical back-off — finest level called ({emb})", fontsize=FS)
    ax.grid(axis="y", alpha=0.3); ax.tick_params(labelsize=FS)
    ax.legend(fontsize=FS, frameon=False, loc="upper right")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(AFIG, f"fig_backoff_{emb}.png"),
                                    dpi=300, bbox_inches="tight"); plt.close(fig)

    # Fig D: conformal set-size histogram
    fig, ax = plt.subplots(figsize=(12, 8))
    maxs = int(np.percentile(set_sizes, 99)) + 1
    ax.hist(np.clip(set_sizes, 0, maxs), bins=np.arange(0, maxs + 2) - 0.5,
            color="#CC79A7", edgecolor="white")
    ax.set_xlabel("EC3 prediction-set size", fontsize=FS); ax.set_ylabel("# test reactions", fontsize=FS)
    ax.set_title(f"Conformal sets ({emb}): coverage={covered.mean():.3f}, "
                 f"mean size={set_sizes.mean():.2f}", fontsize=FS)
    ax.grid(axis="y", alpha=0.3); ax.tick_params(labelsize=FS)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(AFIG, f"fig_conformal_{emb}.png"),
                                    dpi=300, bbox_inches="tight"); plt.close(fig)

    # Fig E: OOD abstain curve
    kc = pd.DataFrame(keep_curve)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(kc.coverage, kc.acc, "o-", lw=2.6, ms=8, color="#0072B2")
    ax.set_xlabel("coverage (answer the least-novel fraction)", fontsize=FS)
    ax.set_ylabel("EC3 accuracy", fontsize=FS)
    ax.set_title(f"Abstain on novel queries by top-1 distance ({emb})\n"
                 f"AUROC(error): dist={auroc_dist:.3f}, LOF={auroc_lof:.3f}", fontsize=FS)
    ax.grid(alpha=0.3); ax.tick_params(labelsize=FS)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(AFIG, f"fig_ood_abstain_{emb}.png"),
                                    dpi=300, bbox_inches="tight"); plt.close(fig)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", nargs="*", default=["ecfp4_diff"],
                    help="embedding(s) to evaluate (must exist in embeddings/)")
    a = ap.parse_args()
    allsum = []
    for e in a.emb:
        if not os.path.exists(os.path.join(EMB, e + ".npy")):
            log(f"SKIP {e} (no embedding)"); continue
        allsum.append(run(e))
    if allsum:
        pd.concat(allsum, ignore_index=True).to_csv(
            os.path.join(ARES, "method_comparison_all.csv"), index=False)
        log(f"wrote method_comparison_all.csv")
