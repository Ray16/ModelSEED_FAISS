# Reaction fingerprint × clustering benchmark (ModelSEED)

**Goal.** Given ModelSEED reactions, find the (reaction-embedding, clustering-algorithm)
combination that groups reactions so that each cluster is as clean as possible with
respect to **EC number** — i.e. reactions catalyzed by the same enzyme class fall in
the same cluster and different classes fall in different clusters.

> **Repo layout.** Scripts are organised by stage under `pipeline/` (see
> `pipeline/README.md` for the stage→results map); shared code is in `_lib/`.
> Below, script names refer to their new locations.

## Dataset (`pipeline/00_data/build_dataset.py` → `dataset.parquet`)

From `ModelSEEDDatabase/Biochemistry`:
- start from all reactions, drop obsolete ones
- keep reactions with ≥1 **full 4-level EC** number (e.g. `1.1.1.1`)
- keep only reactions where **every** participating compound has a parseable RDKit SMILES
- build a reaction SMILES `reactants >> products`, each species repeated by its
  (rounded, abs, capped-at-4) stoichiometric coefficient, RDKit-canonicalized

Result: **13,721 reactions**, **5,622 distinct full EC** numbers
(**261** EC3 subclasses, **68** EC2, **7** EC1).
The label used for scoring is the first full EC per reaction; EC1/EC2/EC3 are its prefixes.

> Because 3,290 of the 5,622 full ECs are singletons, full-EC clustering is partly
> ill-posed. We therefore report the primary ranking at **EC3 (subclass)** granularity
> (261 well-populated classes) and also report full-EC and EC2.

## Embeddings (`pipeline/01_embed/*.py` → `embeddings/*.npy`)

| method          | dim   | description |
|-----------------|-------|-------------|
| `ecfp4_diff`    | 2048  | Morgan r=2 count fingerprint, Σ(products) − Σ(reactants) |
| `ecfp4_concat`  | 4096  | Morgan r=2 binary, [reactant-side ‖ product-side] |
| `ecfp4_drfp`    | 4096  | **ECFP4 diff ‖ DRFP** (the concatenation combo) |
| `drfp`          | 2048  | DRFP reaction difference fingerprint (Probst et al. 2022) |
| `atompair_diff` | 2048  | AtomPair folded count difference |
| `rdk_diff`      | 2048  | RDKit topological folded count difference |
| `maccs_diff`    | 167   | MACCS-key count difference |
| `rxnfp`         | 256   | pretrained rxn-BERT transformer fingerprint (Schwaller et al.) |

Difference fingerprints naturally cancel cofactors that appear identically on both
sides. RXNFP CLS-embeddings were verified to match the official `convert()` to 1e-6.

## Clustering (`pipeline/02_cluster/cluster_eval.py` → `results.csv`)

Every embedding is L2-normalized (cosine geometry); RXNFP is standardized first.

- **MiniBatchKMeans** at k ∈ {50,100,200,400,800,1600}
- **Agglomerative** average-linkage (cosine) and Ward (euclidean) at k ∈ {100,400,800}
- **Birch** at k ∈ {200,800}
- **HDBSCAN** at min_cluster_size ∈ {5,10,20} (discovers k)
- **Butina** (Taylor-Butina, Tanimoto on binarized bits) at distance cutoff ∈ {0.3,0.5,0.7}
  — for the 2048-bit binary fingerprints only

## Metrics

Scored at EC granularity {full, EC3, EC2}. Primary = **EC3 AMI**.

- **homogeneity** — "each cluster contains a single EC class" (the clean-split target);
  rises trivially with #clusters, so compared only at matched k
- **completeness / V-measure**
- **AMI** (adjusted mutual information) — chance- and cardinality-corrected, the fair
  single number for ranking methods that produce different numbers of clusters
- **ARI**
- **purity** — weighted mean over clusters of the majority-EC fraction

## Reproduce

```bash
PY=/homes/rzhu/miniforge3/envs/palm/bin/python
$PY pipeline/00_data/build_dataset.py
$PY pipeline/01_embed/embed_fingerprints.py  # DRFP + ECFP + combos (CPU)
$PY pipeline/01_embed/embed_rxnfp.py         # RXNFP (GPU)
$PY pipeline/01_embed/embed_rdkit.py         # RDKit fps in parallel to DRFP (optional speedup)
$PY pipeline/02_cluster/cluster_eval.py      # -> results.csv
$PY pipeline/02_cluster/consolidate.py       # -> summary_*.csv, figures/
```

See `pipeline/README.md` for the full stage list (embeddings, fusion, analysis,
deployment) and `RESULTS.md` for findings.

Results and findings: see `RESULTS.md` (written after the benchmark completes).
