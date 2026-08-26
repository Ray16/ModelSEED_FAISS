#!/usr/bin/env python
"""Visualise the missing-EC predictions from annotate_missing.py.

Two figures:
  fig_pred_map.png          joint UMAP of the 13,721 labelled reactions (grey
                            background manifold) + the 22,871 missing-EC reactions,
                            confident predictions coloured by predicted EC1, OOD
                            (abstained) predictions in faint red. Shows WHERE the
                            predictions fall: confident calls tile the labelled
                            manifold; OOD calls sit in its sparse voids/periphery.
  fig_pred_distributions.png  (a) EC3 neighbour-agreement histogram + gate,
                              (b) top predicted EC3 subclasses (confident),
                              (c) nearest-labelled-analogue similarity by bucket.

Layout uses ecfp4_diff (fast to recompute; DRFP adds nothing to a 2-D layout);
the predictions/flags themselves come from the full ecfp4_drfp retriever (28).
"""
import os, sys, csv, glob, time
import numpy as np, pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator as fpg
from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
import umap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "_lib"))
import _style; FS = _style.FS
csv.field_size_limit(10**7)

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(HERE, "data"); RESULTS = os.path.join(HERE, "results")
EMB = os.path.join(HERE, "embeddings"); FIG = os.path.join(HERE, "figures")
CACHE = os.path.join(HERE, "cache"); os.makedirs(CACHE, exist_ok=True)
DB = os.path.join(HERE, "..", "ModelSEEDDatabase", "Biochemistry")
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, file=sys.stderr, flush=True)

EC1NAME = {"1":"oxidoreductase","2":"transferase","3":"hydrolase","4":"lyase",
           "5":"isomerase","6":"ligase","7":"translocase"}
OKABE = {"1":"#0072B2","2":"#E69F00","3":"#009E73","4":"#CC79A7",
         "5":"#56B4E9","6":"#D55E00","7":"#F0E442"}

# ---- query featuriser (== embed_fingerprints.py / 28) ----
_mc = {}
def mol(s):
    if s not in _mc: _mc[s] = Chem.MolFromSmiles(s)
    return _mc[s]
morgan = fpg.GetMorganGenerator(radius=2, fpSize=2048)
def cvec(smis):
    v = np.zeros(2048, dtype=np.float32)
    for s in smis:
        m = mol(s)
        if m is not None: v += morgan.GetCountFingerprintAsNumPy(m).astype(np.float32)
    return v
def ecfp4_diff(rxns):
    X = np.zeros((len(rxns), 2048), dtype=np.float32)
    for i, rxn in enumerate(rxns):
        l, r = rxn.split(">>")
        X[i] = cvec(r.split(".")) - cvec(l.split("."))
        if i % 4000 == 0: log(f"  ecfp4 {i}/{len(rxns)}")
    return X

def query_embedding(pred):
    cache = os.path.join(CACHE, "query_missing_ecfp4diff.npy")
    ids_c = os.path.join(CACHE, "query_missing_ids.npy")
    if os.path.exists(cache) and os.path.exists(ids_c):
        ids = np.load(ids_c, allow_pickle=True)
        if list(ids) == list(pred.rxn_id):
            log("loaded cached query ecfp4_diff"); return np.load(cache)
    # need rxn_smiles: rebuild from DB keyed by rxn_id (order = pred)
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location("m28", os.path.join(os.path.dirname(__file__), "annotate_missing.py"))
    m28 = module_from_spec(spec); spec.loader.exec_module(m28)
    dfm = m28.build_missing().set_index("rxn_id")
    rxns = [dfm.loc[i, "rxn_smiles"] for i in pred.rxn_id]
    X = ecfp4_diff(rxns)
    np.save(cache, X); np.save(ids_c, np.array(list(pred.rxn_id), dtype=object))
    return X

def main():
    pred = pd.read_csv(os.path.join(RESULTS, "predicted_missing_ec.tsv"), sep="\t")
    ref = pd.read_parquet(os.path.join(DATA, "dataset.parquet"))
    Rq = np.load(os.path.join(EMB, "ecfp4_diff.npy")).astype(np.float32)   # reference
    Qq = query_embedding(pred)                                             # query
    log(f"reference {Rq.shape}  query {Qq.shape}")

    # ---- joint 2-D layout (SVD-50 on union -> UMAP) ----
    cache2d = os.path.join(CACHE, "pred_umap_2d.npy")
    if os.path.exists(cache2d) and np.load(cache2d).shape[0] == len(Rq) + len(Qq):
        XY = np.load(cache2d); log("loaded cached UMAP")
    else:
        U = normalize(np.vstack([Rq, Qq]).astype(np.float32))
        Z = TruncatedSVD(n_components=50, random_state=0).fit_transform(U)
        log("UMAP on union (this takes a minute)...")
        XY = umap.UMAP(n_neighbors=30, min_dist=0.15, metric="cosine",
                       random_state=0).fit_transform(Z)
        np.save(cache2d, XY)
    XYr, XYq = XY[:len(Rq)], XY[len(Rq):]

    conf = pred.confident.to_numpy() & (~pred.ood.to_numpy())
    ood = pred.ood.to_numpy()
    p1 = pred.pred_ec1.astype(str).to_numpy()

    # ================================================= FIG 1: MAP
    fig, ax = plt.subplots(figsize=(13, 11))
    ax.scatter(XYr[:, 0], XYr[:, 1], s=3, c="#D9D9D9", alpha=0.5,
               linewidths=0, rasterized=True, label="_labelled")
    ax.scatter(XYq[ood, 0], XYq[ood, 1], s=5, c="#F4A6A6", alpha=0.45,
               linewidths=0, rasterized=True, label="_ood")
    for k in sorted(OKABE, key=lambda z: -(conf & (p1 == z)).sum()):
        m = conf & (p1 == k)
        if m.sum() == 0: continue
        ax.scatter(XYq[m, 0], XYq[m, 1], s=6, c=OKABE[k], alpha=0.75,
                   linewidths=0, rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    handles = [Line2D([0],[0], marker='o', ls='', ms=11, mfc="#D9D9D9", mec='none',
                      label="labelled reactions (reference)"),
               Line2D([0],[0], marker='o', ls='', ms=11, mfc="#F4A6A6", mec='none',
                      label="missing-EC: OOD / abstained")]
    for k in OKABE:
        n = int((conf & (p1 == k)).sum())
        if n == 0: continue
        handles.append(Line2D([0],[0], marker='o', ls='', ms=11, mfc=OKABE[k], mec='none',
                              label=f"pred EC{k} {EC1NAME[k]} (n={n:,})"))
    ax.legend(handles=handles, fontsize=FS, frameon=False, loc="upper left",
              bbox_to_anchor=(1.0, 1.0), borderaxespad=0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_pred_map.png"), dpi=300, bbox_inches="tight")
    plt.close(fig); log("wrote figures/fig_pred_map.png")

    # ================================================= FIG 2: DISTRIBUTIONS
    fig, ax = plt.subplots(1, 3, figsize=(22, 7))
    # (a) confidence histogram
    ax[0].hist(pred.conf_ec3, bins=np.linspace(0, 1, 21), color="#8593A3", edgecolor="white")
    ax[0].axvline(0.6, color="#D55E00", ls="--", lw=2.5)
    ax[0].text(0.61, ax[0].get_ylim()[1]*0.92, "gate = 0.6", color="#D55E00", fontsize=FS)
    ax[0].set_xlabel("EC3 neighbour agreement", fontsize=FS)
    ax[0].set_ylabel("missing-EC reactions", fontsize=FS)
    ax[0].set_title("(a) prediction confidence", fontsize=FS)
    # (b) top predicted EC3 subclasses (confident)
    cdf = pred[conf]
    top = cdf.pred_ec3.value_counts().head(15)[::-1]
    cols = [OKABE.get(str(e).split(".")[0], "#999999") for e in top.index]
    ax[1].barh(range(len(top)), top.values, color=cols)
    ax[1].set_yticks(range(len(top))); ax[1].set_yticklabels(top.index, fontsize=FS)
    for i, v in enumerate(top.values):
        ax[1].text(v + max(top.values)*0.01, i, str(v), va="center", fontsize=FS)
    ax[1].set_xlabel("confident predictions", fontsize=FS)
    ax[1].set_title("(b) top predicted EC3 subclasses", fontsize=FS)
    ax[1].set_xlim(0, max(top.values)*1.15)
    # (c) similarity to nearest labelled analogue, by bucket
    lowagree = (~pred.ood.to_numpy()) & (~pred.confident.to_numpy())
    bins = np.linspace(0, 1, 26)
    ax[2].hist(pred.top1_sim[conf], bins=bins, histtype="step", lw=3,
               color="#009E73", label=f"confident (n={conf.sum():,})")
    ax[2].hist(pred.top1_sim[lowagree], bins=bins, histtype="step", lw=3,
               color="#E69F00", label=f"low-agreement (n={lowagree.sum():,})")
    ax[2].hist(pred.top1_sim[ood], bins=bins, histtype="step", lw=3,
               color="#D55E00", label=f"OOD/abstain (n={ood.sum():,})")
    ax[2].axvline(0.511, color="#666666", ls=":", lw=2)
    ax[2].set_xlabel("cosine similarity to nearest labelled reaction", fontsize=FS)
    ax[2].set_ylabel("missing-EC reactions", fontsize=FS)
    ax[2].set_title("(c) distance to labelled manifold", fontsize=FS)
    ax[2].legend(fontsize=FS, frameon=False, loc="upper center")
    for a in ax:
        for sp in ("top", "right"): a.spines[sp].set_visible(False)
        a.grid(axis="y", alpha=0.25); a.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_pred_distributions.png"), dpi=300, bbox_inches="tight")
    plt.close(fig); log("wrote figures/fig_pred_distributions.png")

if __name__ == "__main__":
    main()
