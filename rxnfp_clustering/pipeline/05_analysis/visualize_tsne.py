#!/usr/bin/env python
"""2D reaction maps (openTSNE), each dot a reaction coloured by its EC.

  fig_tsne_ec_comparison.png : small-multiples t-SNE, one panel per embedding,
                               dots coloured by EC top class -> visual comparison
                               of how cleanly each embedding separates EC.
  fig_tsne_ec2_labeled.png   : best embedding, coloured by the most populous EC2
                               subclasses with the EC code printed at each centroid.

t-SNE coords are cached in tsne_coords/ so replotting is instant.
Palette: CVD-safe 6-hue set (validated) for EC classes 1-6; EC7 -> grey "Other".
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import TruncatedSVD
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "_lib"))
import _style  # project rule: all figure text >= 18pt, dpi 300
FS = _style.FS

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
DATA = os.path.join(HERE, "data"); RESULTS = os.path.join(HERE, "results"); CACHE = os.path.join(HERE, "cache")
EMB = os.path.join(HERE, "embeddings")
FIG = os.path.join(HERE, "figures", "benchmark"); os.makedirs(FIG, exist_ok=True)
TSNE = os.path.join(CACHE, "tsne_coords"); os.makedirs(TSNE, exist_ok=True)
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, file=sys.stderr, flush=True)

df = pd.read_parquet(os.path.join(DATA, "dataset.parquet"))
N = len(df)
ec1 = df.ec1.to_numpy()
CONTINUOUS = {"rxnfp", "rxnfp_pretrained", "chemberta_diff", "descriptors_diff", "molformer_diff"}

EC1_NAME = {"1": "Oxidoreductases", "2": "Transferases", "3": "Hydrolases",
            "4": "Lyases", "5": "Isomerases", "6": "Ligases", "7": "Translocases"}
EC1_COLOR = {"1": "#0072B2", "2": "#E69F00", "3": "#009E73", "4": "#CC79A7",
             "5": "#56B4E9", "6": "#D55E00", "7": "#9a9a9a"}  # 7 -> grey "Other"

def make_space(name, Xraw):
    X = StandardScaler().fit_transform(Xraw) if name in CONTINUOUS else Xraw.astype(np.float32)
    X[~np.isfinite(X)] = 0.0
    X = normalize(X)
    d = min(256, X.shape[1] - 1)
    Z = TruncatedSVD(n_components=d, random_state=0).fit_transform(X)
    z = np.linalg.norm(Z, axis=1) < 1e-9
    if z.any(): Z[z] = 0.0; Z[z, 0] = 1.0
    return normalize(Z).astype(np.float32)

def tsne_of(name):
    cache = os.path.join(TSNE, name + ".npy")
    if os.path.exists(cache):
        return np.load(cache)
    from openTSNE import TSNE as OpenTSNE
    Z = make_space(name, np.load(os.path.join(EMB, name + ".npy")))
    log(f"t-SNE {name} ...")
    Y = OpenTSNE(n_components=2, perplexity=30, metric="cosine", n_jobs=6,
                 random_state=0, verbose=False).fit(Z)
    Y = np.asarray(Y)
    np.save(cache, Y)
    log(f"  cached {name}")
    return Y

def clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_aspect("equal")

def knn_score(name):
    p = os.path.join(RESULTS, "knn_results.csv")
    if not os.path.exists(p): return None
    d = pd.read_csv(p)
    r = d[(d.embedding == name) & (d.k == 1)]
    return float(r.ec3_acc.iloc[0]) if len(r) else None

def scatter_ec1(ax, Y, order):
    # draw grey/Other first, then classes so coloured points sit on top
    for cls in ["7"] + [c for c in order if c != "7"]:
        m = ec1 == cls
        if not m.any(): continue
        ax.scatter(Y[m, 0], Y[m, 1], s=3, c=EC1_COLOR.get(cls, "#9a9a9a"),
                   alpha=0.55, linewidths=0, rasterized=True)

def fig_comparison(names):
    order = [c for c in "1234567"]
    ncol = 3; nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.0*ncol, 6.2*nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, name in zip(axes, names):
        Y = tsne_of(name)
        scatter_ec1(ax, Y, order)
        s = knn_score(name)
        ttl = name if s is None else f"{name}   (1-NN EC3 acc {s:.2f})"
        ax.set_title(ttl, fontsize=FS, pad=6)
        clean(ax)
    for ax in axes[len(names):]: ax.set_visible(False)
    # shared legend — two rows so 7 large entries never crowd
    handles = [Line2D([0],[0], marker='o', linestyle='', markersize=12,
                      markerfacecolor=EC1_COLOR[c], markeredgecolor='none',
                      label=f"{c}  {EC1_NAME[c]}") for c in order if (ec1==c).any()]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               frameon=False, fontsize=FS, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Reaction embeddings — t-SNE map, each dot a reaction coloured by EC class",
                 fontsize=FS, y=1.0)
    fig.tight_layout(rect=(0, 0.06, 1, 0.98))
    out = os.path.join(FIG, "fig_tsne_ec_comparison.png")
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    log(f"wrote {out}")

def fig_ec2_labeled(name, topn=12):
    Y = tsne_of(name)
    ec2 = df.ec2.to_numpy()
    top = pd.Series(ec2).value_counts().head(topn).index.tolist()
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(14, 11))
    # background: everything grey
    ax.scatter(Y[:, 0], Y[:, 1], s=3, c="#dddddd", alpha=0.5, linewidths=0, rasterized=True)
    for i, e in enumerate(top):
        m = ec2 == e
        ax.scatter(Y[m, 0], Y[m, 1], s=8, color=cmap(i % 20), alpha=0.8,
                   linewidths=0, rasterized=True, label=f"{e}  (n={m.sum()})")
        cx, cy = Y[m, 0].mean(), Y[m, 1].mean()
        ax.text(cx, cy, e, fontsize=FS, weight="bold", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    clean(ax)
    ax.set_title(f"{name}: t-SNE coloured by the {topn} most common EC2 subclasses",
                 fontsize=FS)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=FS, frameon=False, markerscale=2.5)
    fig.tight_layout()
    out = os.path.join(FIG, "fig_tsne_ec2_labeled.png")
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    log(f"wrote {out}")

if __name__ == "__main__":
    compare = ["ecfp4_diff", "ecfp4_drfp", "drfp",
               "rxnfp", "chemberta_diff", "molformer_diff"]
    fig_comparison(compare)
    fig_ec2_labeled("ecfp4_diff")
    log("visualizations done")
