#!/usr/bin/env python
"""Apply the best retriever to the reactions ModelSEED is MISSING an EC for.

This is the deployment step the whole benchmark was building toward: instead of
hiding known labels and measuring accuracy (annotation/), we run real inference on
the ~32k non-obsolete reactions that carry NO full 4-level EC, using the 13,721
EC-labelled reactions as the reference index.

Model = the validated best retriever: `ecfp4_drfp` (ECFP4-diff || DRFP), cosine,
confidence-gated 15-NN majority vote (RESULTS.md: EC3 0.90 acc @ ~75% coverage).
For each missing-EC reaction we emit a predicted EC1/EC2/EC3 with a per-level
confidence (fraction of the 15 neighbours agreeing), the nearest labelled analogue,
an abstain flag (EC3 agreement < 0.6) and an OOD flag (top-1 similarity below the
5th percentile of the reference set's own 1-NN similarity). Metric-learned second
opinion is added best-effort if torch is available.

Query featurisation is byte-identical to scripts/02_embed.py so query and reference
live in the same geometry.

Outputs:
  results/predicted_missing_ec.tsv     one row per scoreable missing-EC reaction
  results/missing_ec_landscape.csv     coverage/agreement summary
  figures/fig_missing_landscape.png    the landscape at a glance
"""
import os, sys, csv, glob, time
import numpy as np, pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator as fpg
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _style; FS = _style.FS
csv.field_size_limit(10**7)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data"); RESULTS = os.path.join(HERE, "results")
EMB = os.path.join(HERE, "embeddings"); FIG = os.path.join(HERE, "figures")
DB = os.path.join(HERE, "..", "ModelSEEDDatabase", "Biochemistry")
def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, file=sys.stderr, flush=True)

GATE = 0.6   # EC3 neighbour-agreement threshold for a confident (non-abstain) call

# ---------------------------------------------------------------- build query set
def first_full_ec(e):
    if not e or e == 'null': return None
    for tok in e.replace('|', ';').split(';'):
        p = tok.strip().split('.')
        if len(p) == 4 and all(x.isdigit() for x in p): return tok.strip()
    return None

def parse_stoich(s):
    out = []
    for part in s.split(';'):
        if not part.strip(): continue
        f = part.split(':')
        if len(f) >= 2 and f[1]:
            try: coef = float(f[0])
            except ValueError: continue
            out.append((coef, f[1]))
    return out

def build_rxn_smiles(stoich, smi):
    react, prod = [], []
    for coef, cid in stoich:
        if cid not in smi: return None
        mult = min(4, max(1, int(round(abs(coef)))))
        (react if coef < 0 else prod).extend([smi[cid]] * mult)
    if not react or not prod: return None
    return ".".join(react) + ">>" + ".".join(prod)

def load_smiles():
    smi = {}
    for f in sorted(glob.glob(os.path.join(DB, "compound_*.tsv"))):
        for row in csv.DictReader(open(f), delimiter='\t'):
            s = row.get('smiles', '')
            if s and s != 'null':
                m = Chem.MolFromSmiles(s)
                if m is not None: smi[row['id']] = Chem.MolToSmiles(m)
    return smi

def build_missing():
    smi = load_smiles(); log(f"parseable compounds: {len(smi)}")
    rows = []
    for f in sorted(glob.glob(os.path.join(DB, "reaction_*.tsv"))):
        for row in csv.DictReader(open(f), delimiter='\t'):
            if row.get('is_obsolete') == '1': continue
            if first_full_ec(row.get('ec_numbers', '')): continue      # keep only NO-EC
            stoich = parse_stoich(row.get('stoichiometry', ''))
            if not stoich: continue
            rs = build_rxn_smiles(stoich, smi)
            if rs is None: continue
            rows.append(dict(rxn_id=row['id'], name=row.get('name', ''),
                             n_cpd=len(stoich), rxn_smiles=rs))
    df = pd.DataFrame(rows).drop_duplicates('rxn_id').reset_index(drop=True)
    log(f"scoreable missing-EC reactions: {len(df)}")
    return df

# ---------------------------------------------------------------- featurise (== 02_embed.py)
_mol_cache = {}
def mol(s):
    if s not in _mol_cache: _mol_cache[s] = Chem.MolFromSmiles(s)
    return _mol_cache[s]
def sides(rxn):
    l, r = rxn.split(">>"); return l.split("."), r.split(".")
morgan = fpg.GetMorganGenerator(radius=2, fpSize=2048)
def count_vec(smis, gen, dim):
    v = np.zeros(dim, dtype=np.float32)
    for s in smis:
        m = mol(s)
        if m is not None: v += gen.GetCountFingerprintAsNumPy(m).astype(np.float32)
    return v
def ecfp4_diff(rxns):
    X = np.zeros((len(rxns), 2048), dtype=np.float32)
    for i, rxn in enumerate(rxns):
        L, R = sides(rxn)
        X[i] = count_vec(R, morgan, 2048) - count_vec(L, morgan, 2048)
        if i % 2000 == 0: log(f"  ecfp4 {i}/{len(rxns)}")
    return X
def drfp_encode(rxns):
    from drfp import DrfpEncoder
    log("  DRFP encode (query)...")
    return np.array(DrfpEncoder.encode(rxns, n_folded_length=2048), dtype=np.float32)

# ---------------------------------------------------------------- main
def main():
    df = build_missing()
    qrx = df.rxn_smiles.tolist()

    log("featurising query set (ecfp4_diff || drfp)...")
    Qe = ecfp4_diff(qrx); Qd = drfp_encode(qrx)
    Q = normalize(np.hstack([Qe, Qd]).astype(np.float32))

    ref = pd.read_parquet(os.path.join(DATA, "dataset.parquet"))
    Re = np.load(os.path.join(EMB, "ecfp4_diff.npy")).astype(np.float32)
    Rd = np.load(os.path.join(EMB, "drfp.npy")).astype(np.float32)
    R = normalize(np.hstack([Re, Rd]).astype(np.float32))
    log(f"reference {R.shape}  query {Q.shape}")

    ec = {lvl: ref[lvl].to_numpy() for lvl in ("ec1", "ec2", "ec3", "ec")}

    # reference self 1-NN similarity (LOO) -> OOD threshold (5th pct)
    nnR = NearestNeighbors(n_neighbors=2, metric="cosine", algorithm="brute").fit(R)
    dR, _ = nnR.kneighbors(R)
    ref_top1_sim = 1.0 - dR[:, 1]
    ood_thr = float(np.percentile(ref_top1_sim, 5))
    log(f"OOD threshold (5th pct of ref 1-NN sim) = {ood_thr:.3f}")

    # query -> reference 15-NN
    nn = NearestNeighbors(n_neighbors=15, metric="cosine", algorithm="brute").fit(R)
    dist, idx = nn.kneighbors(Q)
    top1_sim = 1.0 - dist[:, 0]

    from collections import Counter
    out = df.copy()
    out["top1_sim"] = np.round(top1_sim, 4)
    out["top1_ref_id"] = ref["rxn_id"].to_numpy()[idx[:, 0]]
    out["top1_ref_ec"] = ec["ec"][idx[:, 0]]
    for lvl in ("ec1", "ec2", "ec3"):
        lab = ec[lvl]; preds, confs = [], []
        for row in idx:
            c = Counter(lab[row]); top, n = c.most_common(1)[0]
            preds.append(top); confs.append(n / len(row))
        out[f"pred_{lvl}"] = preds
        out[f"conf_{lvl}"] = np.round(confs, 3)
    out["confident"] = out["conf_ec3"] >= GATE
    out["ood"] = out["top1_sim"] < ood_thr

    # ---- metric-learned second opinion (best-effort) ----
    try:
        import torch, torch.nn as nn_
        from sklearn.decomposition import TruncatedSVD
        log("metric-learned second opinion (train on all labelled)...")
        DEV = "cuda" if torch.cuda.is_available() else "cpu"
        rng = np.random.default_rng(0); torch.manual_seed(0)
        sv = TruncatedSVD(n_components=256, random_state=0).fit(R)
        Zr = normalize(sv.transform(R)).astype(np.float32)
        Zq = normalize(sv.transform(Q)).astype(np.float32)
        class Proj(nn_.Module):
            def __init__(s, din):
                super().__init__()
                s.net = nn_.Sequential(nn_.Linear(din, 512), nn_.BatchNorm1d(512),
                                       nn_.ReLU(), nn_.Linear(512, 128))
            def forward(s, x): return nn_.functional.normalize(s.net(x), dim=1)
        def supcon(z, y, tau=0.1):
            y = y.view(-1, 1); pos = (y == y.t()).float()
            eye = torch.eye(z.size(0), device=z.device); pos = pos - eye
            lg = z @ z.t() / tau; lg = lg - lg.max(1, keepdim=True)[0].detach()
            exp = torch.exp(lg) * (1 - eye)
            logp = lg - torch.log(exp.sum(1, keepdim=True) + 1e-9)
            npos = pos.sum(1)
            return (-(pos * logp).sum(1) / npos.clamp(min=1))[npos > 0].mean()
        y3 = ec["ec3"]; labs, inv = np.unique(y3, return_inverse=True)
        by = [np.where(inv == c)[0] for c in range(len(labs))]
        ok = [c for c in range(len(labs)) if len(by[c]) >= 2]
        Xt = torch.tensor(Zr, device=DEV); yt = torch.tensor(inv, device=DEV)
        m = Proj(256).to(DEV); opt = torch.optim.Adam(m.parameters(), 1e-3, weight_decay=1e-4)
        P, K = 48, 8; steps = max(1, len(Zr) // (P * K)); m.train()
        for ep in range(80):
            for _ in range(steps):
                cls = rng.choice(ok, size=min(P, len(ok)), replace=False); ii = []
                for c in cls:
                    pool = by[c]; ii.extend(rng.choice(pool, size=min(K, len(pool)),
                                                       replace=len(pool) < K))
                ii = np.array(ii); loss = supcon(m(Xt[ii]), yt[ii])
                opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            Lr = m(torch.tensor(Zr, device=DEV)).cpu().numpy()
            Lq = m(torch.tensor(Zq, device=DEV)).cpu().numpy()
        nnm = NearestNeighbors(n_neighbors=15, metric="cosine", algorithm="brute").fit(Lr)
        _, im = nnm.kneighbors(Lq)
        preds = [Counter(ec["ec3"][row]).most_common(1)[0][0] for row in im]
        out["pred_ec3_metric"] = preds
        out["metric_agrees"] = out["pred_ec3_metric"] == out["pred_ec3"]
        log(f"  metric agrees with retrieval on {out['metric_agrees'].mean()*100:.1f}% of calls")
    except Exception as e:
        log(f"  metric-learned second opinion skipped: {e}")

    out.to_csv(os.path.join(RESULTS, "predicted_missing_ec.tsv"), sep="\t", index=False)
    log(f"wrote results/predicted_missing_ec.tsv  ({len(out)} rows)")

    # ---------------------------------------------------------------- landscape
    N = len(out)
    conf = out[out.confident & ~out.ood]
    summ = []
    for thr in [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        sel = out[(out.conf_ec3 >= thr) & ~out.ood]
        summ.append(dict(ec3_conf_threshold=thr, n_covered=len(sel),
                         coverage=round(len(sel)/N, 3),
                         mean_top1_sim=round(sel.top1_sim.mean(), 3) if len(sel) else np.nan))
    pd.DataFrame(summ).to_csv(os.path.join(RESULTS, "missing_ec_landscape.csv"), index=False)

    log("=" * 64)
    log(f"TOTAL scoreable missing-EC reactions: {N}")
    log(f"OOD (far from any labelled reaction):  {int(out.ood.sum())} ({out.ood.mean()*100:.1f}%)")
    log(f"CONFIDENT EC3 call (agree>= {GATE}, in-domain): {len(conf)} ({len(conf)/N*100:.1f}%)")
    log(f"  of which top1_sim==1.0 (exact labelled twin): {int((conf.top1_sim>=0.999).sum())}")
    if "metric_agrees" in out:
        log(f"  metric-learned agrees on confident calls: "
            f"{conf['metric_agrees'].mean()*100:.1f}%")
    log("Top predicted EC1 among confident calls:")
    ec1n = {"1":"oxidoreductase","2":"transferase","3":"hydrolase","4":"lyase",
            "5":"isomerase","6":"ligase","7":"translocase"}
    for k, v in conf.pred_ec1.value_counts().items():
        log(f"    EC{k} {ec1n.get(str(k),''):16s} {v:5d} ({v/len(conf)*100:.1f}%)")

    # ---------------------------------------------------------------- figure
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    # (a) coverage vs threshold
    s = pd.DataFrame(summ)
    ax[0].plot(s.ec3_conf_threshold, s.coverage, "o-", color="#0072B2", lw=2, ms=9)
    for _, r in s.iterrows():
        ax[0].annotate(f"{int(r.n_covered)}", (r.ec3_conf_threshold, r.coverage),
                       textcoords="offset points", xytext=(0, 10), ha="center", fontsize=FS)
    ax[0].axvline(GATE, color="#D55E00", ls="--", lw=2)
    ax[0].set_xlabel("EC3 neighbour-agreement threshold", fontsize=FS)
    ax[0].set_ylabel("fraction of missing-EC reactions covered", fontsize=FS)
    ax[0].set_ylim(0, 1); ax[0].grid(alpha=0.25); ax[0].set_axisbelow(True)
    for sp in ("top", "right"): ax[0].spines[sp].set_visible(False)
    # (b) predicted EC1 breakdown among confident
    vc = conf.pred_ec1.value_counts().sort_index()
    colors = ["#0072B2","#E69F00","#009E73","#CC79A7","#56B4E9","#D55E00","#F0E442"]
    labels = [f"EC{k}\n{ec1n.get(str(k),'')}" for k in vc.index]
    ax[1].bar(range(len(vc)), vc.values, color=[colors[int(k)-1] for k in vc.index])
    for i, v in enumerate(vc.values):
        ax[1].text(i, v + max(vc.values)*0.01, str(v), ha="center", va="bottom", fontsize=FS)
    ax[1].set_xticks(range(len(vc))); ax[1].set_xticklabels(labels, fontsize=FS)
    ax[1].set_ylabel("confident predictions", fontsize=FS)
    for sp in ("top", "right"): ax[1].spines[sp].set_visible(False)
    ax[1].grid(axis="y", alpha=0.25); ax[1].set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_missing_landscape.png"), dpi=300, bbox_inches="tight")
    log("wrote figures/fig_missing_landscape.png")

if __name__ == "__main__":
    main()
