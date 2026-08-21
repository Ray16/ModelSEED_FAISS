#!/usr/bin/env python
"""Pretrained GIN reaction-difference embedding (GPU) -- the learned-*graph*
axis (Hu et al. 2020, "Strategies for Pre-training Graph Neural Networks",
snap-stanford/pretrain-gnns). Every other "learned" embedding in this
benchmark (ChemBERTa, MolFormer, RXNFP) is a SMILES-*string* language model;
GIN instead pretrains directly on the 2D molecular *graph* (5-layer GIN,
context-prediction objective over 2M ZINC15 + PPI-supervised molecules) --
a genuinely different inductive bias from both the string transformers and
the hand-engineered circular fingerprints.

Checkpoint: official `supervised_contextpred.pth` from
https://raw.githubusercontent.com/snap-stanford/pretrain-gnns/master/chem/model_gin/supervised_contextpred.pth
(cached under cache/gin_checkpoint/). It is an encoder-only state dict
(GNN, not GNN_graphpred) -- loaded directly into the bare `GNN` module below.

Featurization matches chem/loader.py::mol_to_graph_data_obj_simple EXACTLY
(atomic-num index 0..117 + chirality tag 0..2, bond type 0..3 + bond dir
0..2, +1 self-loop bond-type slot at index 4, +1 reserved mask-token slot
at index 5) -- verified the checkpoint's embedding table shapes
(120, 300)/(3, 300)/(6, 300)/(3, 300) match this scheme before trusting it.
GINConv is reimplemented for modern torch_geometric (aggr passed to
MessagePassing.__init__, not to propagate()).

  gin_diff   300-dim float, mean-pooled node embeddings,
             Sum(products) - Sum(reactants)
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data, Batch
from rdkit import Chem

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
DATA = os.path.join(HERE, "data"); RESULTS = os.path.join(HERE, "results"); CACHE = os.path.join(HERE, "cache")
EMB = os.path.join(HERE, "embeddings"); os.makedirs(EMB, exist_ok=True)
CKPT_DIR = os.path.join(CACHE, "gin_checkpoint"); os.makedirs(CKPT_DIR, exist_ok=True)
CKPT = os.path.join(CKPT_DIR, "supervised_contextpred.pth")
CKPT_URL = "https://raw.githubusercontent.com/snap-stanford/pretrain-gnns/master/chem/model_gin/supervised_contextpred.pth"

def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, file=sys.stderr, flush=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------- checkpoint
if not os.path.exists(CKPT):
    import urllib.request
    log(f"downloading {CKPT_URL}")
    urllib.request.urlretrieve(CKPT_URL, CKPT)
sz = os.path.getsize(CKPT)
if sz < 1_000_000:
    log(f"FATAL: checkpoint at {CKPT} is only {sz} bytes -- likely an LFS pointer / HTML error page, not the real weights")
    sys.exit(1)
log(f"checkpoint OK: {CKPT} ({sz} bytes)")

# ---------------------------------------------------------------- featurization (chem/loader.py, exact)
POSSIBLE_ATOMIC_NUM = list(range(1, 119))          # index 0..117 -> atomic num 1..118
POSSIBLE_CHIRALITY = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]
POSSIBLE_BONDS = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
POSSIBLE_BOND_DIRS = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
]
NUM_CHIRALITY_TAG = 3  # x_embedding2 table only has 3 rows; CHI_OTHER (idx 3) must clamp

def mol_to_data(mol):
    """chem/loader.py::mol_to_graph_data_obj_simple, ported."""
    atom_feats = []
    for atom in mol.GetAtoms():
        an_idx = POSSIBLE_ATOMIC_NUM.index(atom.GetAtomicNum())
        ch_idx = POSSIBLE_CHIRALITY.index(atom.GetChiralTag())
        ch_idx = min(ch_idx, NUM_CHIRALITY_TAG - 1)  # guard CHI_OTHER out-of-range
        atom_feats.append([an_idx, ch_idx])
    x = torch.tensor(atom_feats, dtype=torch.long)

    if mol.GetNumBonds() > 0:
        edges, eattrs = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bt = POSSIBLE_BONDS.index(bond.GetBondType())
            bd = POSSIBLE_BOND_DIRS.index(bond.GetBondDir())
            edges += [(i, j), (j, i)]
            eattrs += [[bt, bd], [bt, bd]]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(eattrs, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 2), dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# ---------------------------------------------------------------- model (chem/model.py, ported to modern PyG)
NUM_ATOM_TYPE = 120
NUM_CHIRALITY = 3
NUM_BOND_TYPE = 6      # 4 real + 1 self-loop + 1 mask-token
NUM_BOND_DIRECTION = 3

class GINConv(MessagePassing):
    def __init__(self, emb_dim, aggr="add"):
        super().__init__(aggr=aggr)
        self.mlp = nn.Sequential(nn.Linear(emb_dim, 2 * emb_dim), nn.ReLU(),
                                 nn.Linear(2 * emb_dim, emb_dim))
        self.edge_embedding1 = nn.Embedding(NUM_BOND_TYPE, emb_dim)
        self.edge_embedding2 = nn.Embedding(NUM_BOND_DIRECTION, emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loop_attr = torch.zeros(x.size(0), 2, dtype=edge_attr.dtype, device=edge_attr.device)
        self_loop_attr[:, 0] = 4  # self-loop bond type
        edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)
        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)

class GNN(nn.Module):
    def __init__(self, num_layer=5, emb_dim=300, JK="last", drop_ratio=0.0):
        super().__init__()
        self.num_layer = num_layer; self.JK = JK; self.drop_ratio = drop_ratio
        self.x_embedding1 = nn.Embedding(NUM_ATOM_TYPE, emb_dim)
        self.x_embedding2 = nn.Embedding(NUM_CHIRALITY, emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        self.gnns = nn.ModuleList([GINConv(emb_dim, aggr="add") for _ in range(num_layer)])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(emb_dim) for _ in range(num_layer)])

    def forward(self, x, edge_index, edge_attr):
        h = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])
        h_list = [h]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)
        return h_list[-1]  # JK="last"

def unique_mols(rxns):
    s = set()
    for r in rxns:
        L, R = r.split(">>"); s.update(L.split(".")); s.update(R.split("."))
    return sorted(s)

def main():
    if os.path.exists(os.path.join(EMB, "gin_diff.npy")):
        log("gin_diff.npy already exists, skipping"); return

    df = pd.read_parquet(os.path.join(DATA, "dataset.parquet"))
    rxns = df.rxn_smiles.tolist(); N = len(rxns)

    model = GNN(num_layer=5, emb_dim=300, JK="last", drop_ratio=0.0)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(sd, strict=True)
    log(f"loaded checkpoint into GNN encoder strict=True (missing={missing}, unexpected={unexpected})")
    model = model.to(dev).eval()

    umols = unique_mols(rxns); log(f"{len(umols)} unique molecules")
    mols, valid_smis = [], []
    for s in umols:
        m = Chem.MolFromSmiles(s)
        if m is not None:
            mols.append(m); valid_smis.append(s)
    log(f"{len(valid_smis)}/{len(umols)} parse with RDKit")

    datas = []
    ok_smis = []
    for s, m in zip(valid_smis, mols):
        try:
            datas.append(mol_to_data(m)); ok_smis.append(s)
        except Exception as ex:
            log(f"  featurization failed for {s}: {ex}")
    log(f"{len(ok_smis)}/{len(valid_smis)} featurized OK")

    BS = 256
    vecs = {}
    t = time.time()
    with torch.no_grad():
        for i in range(0, len(datas), BS):
            batch = Batch.from_data_list(datas[i:i+BS]).to(dev)
            node_rep = model(batch.x, batch.edge_index, batch.edge_attr)
            graph_rep = global_mean_pool(node_rep, batch.batch)
            gr = graph_rep.cpu().numpy()
            for j, s in enumerate(ok_smis[i:i+BS]):
                vecs[s] = gr[j]
            if i % (BS * 10) == 0: log(f"  mols {i}/{len(datas)} ({time.time()-t:.0f}s)")
    log(f"GIN embedding done ({time.time()-t:.0f}s)")

    DIM = 300
    X = np.zeros((N, DIM), np.float32)
    mask = np.ones(N, bool)
    for i, r in enumerate(rxns):
        L, R = r.split(">>")
        lmols = L.split("."); rmols = R.split(".")
        if any(s not in vecs for s in lmols) or any(s not in vecs for s in rmols):
            mask[i] = False
            continue
        vr = sum(vecs[s] for s in rmols)
        vl = sum(vecs[s] for s in lmols)
        X[i] = vr - vl

    np.save(os.path.join(EMB, "gin_diff.npy"), X)
    np.save(os.path.join(EMB, "gin_diff_mask.npy"), mask)
    log(f"saved gin_diff.npy {X.shape}, mask: {mask.sum()}/{N} ({100*mask.mean():.1f}%) OK")

if __name__ == "__main__":
    main()
