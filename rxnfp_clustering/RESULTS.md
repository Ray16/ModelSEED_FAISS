# Results — reaction embeddings × similarity/clustering for clean EC splits

**Scope (expanded run).** 13,721 ModelSEED reactions · the full sweep is **27
embeddings × 12 clustering algorithms** · plus a **kNN EC-annotation-transfer**
evaluation and **t-SNE maps**. Primary metric: **EC3 (subclass) AMI** — mutual
information vs. the 261 EC3 subclasses, corrected for chance and cluster count.
Also scored at EC1/EC2/full.

**Reported set (curated to 22 × 9).** The figures and ranked tables below report a
curated **22 embeddings × 9 algorithms**; the five weakest/redundant embeddings
(`rxnfp_sidediff`, `rxnfp_moldiff`, `ecfp4_concat`, `cgr_morgan`, `molformer_diff`)
and three algorithms (`agglo_single_cos` — chaining; `dbscan` — duplicate of
`hdbscan`; `spectral` — no unique win) are dropped from the analysis because each
answers no question a kept candidate already answers. Their rows remain in
`results.csv` / `knn_results.csv` for the record; the curation is applied only in
`pipeline/02_cluster/consolidate.py` (`EXCLUDE_EMB` / `EXCLUDE_ALGO`).

## Bottom line

> **Best clustering:** a **structure-based difference fingerprint + agglomerative
> average/complete-linkage (cosine) at k≈400–800** → EC3 AMI ≈ 0.63.
> The top embeddings are a tight pack — the choice of *fingerprint within the
> structural family barely matters*; the *clustering algorithm and the difference
> construction matter more*.
>
> **Best for annotation:** don't cluster — **retrieve**. 15-NN majority vote assigns
> EC3 at **0.90 accuracy over ~76% of reactions** (confidence-gated). This is the
> operating mode a reaction FAISS index would use.

## Embedding ranking (best config each, by EC3 AMI)

| # | embedding | family | EC3 AMI | EC3 homog | EC3 purity | full-EC AMI |
|--:|-----------|--------|--------:|----------:|-----------:|------------:|
| 1 | **ecfp4_drfp** (ECFP4+DRFP) | structural | 0.630 | 0.747 | 0.680 | 0.369 |
| 2 | **secfp_diff** (MHFP) | structural | 0.629 | 0.729 | 0.659 | 0.368 |
| 3 | ecfp6_diff | structural | 0.628 | 0.670 | 0.612 | 0.333 |
| 4 | **descriptors_diff** | physicochem | 0.624 | 0.787 | 0.717 | 0.401 |
| 5 | ecfp4_diff | structural | 0.623 | 0.736 | 0.673 | 0.354 |
| 6 | ecfp2_diff | structural | 0.620 | 0.815 | 0.744 | 0.401 |
| 7 | atompair_diff | structural | 0.618 | 0.727 | 0.665 | 0.379 |
| 8 | maccs_diff | structural | 0.609 | 0.711 | 0.640 | 0.387 |
| 9 | drfp | structural | 0.588 | 0.686 | 0.605 | 0.353 |
| 10 | rdk_diff | structural | 0.588 | 0.697 | 0.621 | 0.358 |
| 11 | **chemberta_diff** | learned | 0.580 | 0.767 | 0.691 | 0.387 |
| 12 | torsion_diff | structural | 0.579 | 0.624 | 0.555 | 0.334 |
| 13 | avalon_diff | structural | 0.575 | 0.682 | 0.610 | 0.363 |
| 14 | **molformer_diff** | learned | 0.569 | 0.755 | 0.681 | 0.387 |
| 15 | ecfp4_concat | structural | 0.567 | 0.669 | 0.594 | 0.373 |
| 16 | **rxnfp** | learned | 0.542 | 0.802 | 0.714 | 0.424 |
| 17 | **rxnfp_pretrained** | learned | 0.490 | 0.685 | 0.567 | 0.395 |

## Algorithm ranking (best EC3 AMI over all embeddings)

| algorithm | best EC3 AMI | paradigm | note |
|-----------|-------------:|----------|------|
| **agglo average (cosine)** | 0.630 | hierarchical | best overall |
| **agglo complete (cosine)** | 0.623 | hierarchical | ~tied, higher homogeneity |
| **Birch** | 0.619 | tree/centroid | fast, nearly as good |
| agglo Ward | 0.603 | hierarchical | |
| **Leiden** | 0.602 | graph community | coarser (≈45–90 clusters); good but not at matched k |
| MiniBatch KMeans | 0.598 | centroid | cheap; peaks k≈100–200 |
| Butina | 0.591 | Tanimoto threshold | classic; binary FPs only |
| Louvain | 0.577 | graph community | coarser still |
| Spectral | 0.574 | graph/eigen | mid, and slow |
| DBSCAN | 0.472 | density | high noise |
| HDBSCAN | 0.468 | density | high noise |
| agglo **single** (cosine) | 0.257 | hierarchical | **chaining collapse — avoid** |

## Findings

1. **The structural family is a plateau, not a peak.** ECFP (r=2/4/6), MHFP/SECFP,
   AtomPair, MACCS, and even a plain **physicochemical descriptor difference** all
   land at EC3 AMI 0.61–0.63. Radius and exact fingerprint hardly move the needle —
   what matters is (a) taking the *product−reactant difference* and (b) the
   clustering algorithm. `descriptors_diff` is the surprise: 217 RDKit 2D descriptors
   match the circular fingerprints, with the **highest homogeneity/purity** of the pack.

2. **Learned embeddings underperform structural ones — consistently.** ChemBERTa (0.58)
   and MolFormer (0.57), pooled per-molecule into reaction differences, sit below the
   cheap structural fingerprints. Reaction-level **RXNFP (0.54)** and especially
   **RXNFP-pretrained (0.49, worst of all 17)** trail further — the USPTO-organic
   pretraining transfers poorly to enzymatic biochemistry. Note RXNFP posts high
   *homogeneity/purity* at large k but low *AMI*: it makes pure-but-fragmented clusters.

3. **Average/complete-linkage agglomerative wins; single-linkage fails.** Hierarchical
   average/complete, Birch, and KMeans form the strong tier. **Single-linkage chains
   into one giant cluster (AMI 0.26).** Graph-community methods (**Leiden/Louvain**) are
   respectable but settle on far fewer, coarser clusters, so they lose at EC3 granularity.
   Density methods (DBSCAN/HDBSCAN) are the weakest — reaction density is too uneven,
   30–50% of points become noise.

4. **Retrieval ≫ clustering for annotation.** 15-NN majority vote (best embeddings):
   **EC1 0.91 · EC2 0.83 · EC3 0.78**; with a confidence gate (≥60% of neighbours agree)
   **EC3 0.90 accuracy at 76% coverage**. 1-NN alone gets EC3 0.87. Full 4-level EC stays
   hard everywhere (≈0.32–0.51) — the structure-blind leaf distinctions (stereo, chain
   length, substrate identity) that difference fingerprints can't see.

5. **The t-SNE maps make it visible** (`fig_tsne_ec_comparison.png`): every embedding
   separates the seven EC top-classes into recognizable territories; tight EC2 islands
   (2.1 methyltransferases, 2.4 glycosyltransferases, 1.14 oxygenases) vs. diffuse ones
   (1.1, 1.2) explain exactly where clustering purity comes from and where it breaks.

## Recommendation

- **Clustering:** `ecfp4_diff` (robust default; or `descriptors_diff` for max purity)
  → L2-normalize → **agglomerative average or complete linkage (cosine)**, k≈400–800.
  Cheap, no GPU. Birch if you need speed at scale.
- **Annotation transfer / FAISS index:** index `ecfp4_diff` (256-d SVD, cosine) and use
  **confidence-gated 15-NN**. Trust EC1–EC3 calls; treat full-EC as a suggestion.
- **Skip** for this goal: single-linkage, density clustering, and the learned/transformer
  embeddings (RXNFP/ChemBERTa/MolFormer) — they cost more and score less.

## Figures (`figures/`)
- `fig_heatmap_embed_algo.png` — EC3 AMI, embedding × algorithm (the master comparison)
- `fig_tsne_ec_comparison.png` — t-SNE maps, dots coloured by EC class, 6 embeddings
- `fig_tsne_ec2_labeled.png` — t-SNE coloured/labelled by top EC2 subclasses
- `fig_knn_transfer.png` — kNN EC1/EC2/EC3 annotation-transfer accuracy per embedding
- `fig_best_per_embedding.png`, `fig_best_per_algo.png`, `fig_kmeans_ami_vs_k.png`,
  `fig_homogeneity_completeness.png`

Data: `results.csv` (550 configs), `knn_results.csv`, `summary_*.csv`.

---

# Part II — CGR, RXNFP-difference, and the fusion method

Expansion (21 embeddings total): added atom-mapped **CGR** fingerprints
(RXNMapper → chython), two **RXNFP-difference** variants, and a **multi-view fusion**.

## Does giving RXNFP a difference operation help?
- `rxnfp_moldiff` (per-molecule difference, the ChemBERTa recipe): EC3 AMI 0.556 vs
  plain rxnfp 0.542, 1-NN EC3 0.839 vs 0.834 — a **hair better**.
- `rxnfp_sidediff` (encode each side, subtract): **worse** (1-NN EC3 0.776) — one-sided
  strings are off-distribution for a reaction transformer.
- **Verdict:** the missing difference is *not* what holds RXNFP back — its ceiling is
  the training distribution/taxonomy (USPTO organic synthesis), not the encoding.

## CGR (atom-mapped bond changes) — fair subset comparison
CGR only composes for **74.8%** of reactions (RXNMapper confidence 0.43 mean; large
cofactor-heavy reactions exceed the 512-token limit). Scored on the CGR-defined
subset (n=10,260) against incumbents on the *same* subset:

| embedding | EC3 AMI | **full-EC 1-NN** |
|-----------|--------:|-----------------:|
| ECFP4-diff | 0.628 | 0.490 |
| Descriptors | **0.667** | 0.484 |
| **CGR (morgan)** | 0.613 | **0.543** |

**CGR wins the full-EC retrieval it was designed for (+5 pts)** but does *not* dominate
EC3, and is coverage-limited. Hypothesis: *partially* confirmed — the mechanism is real
but CGR alone is not the SOTA-beater.

## The fusion method (rung 1 — unsupervised, no fitting)
Different views win different metrics ⇒ fuse them (per-view normalize → concat → SVD-256
→ agglo-average). On the fair subset:

| config | EC3 AMI | EC3 15-NN | full-EC 1-NN |
|--------|--------:|----------:|-------------:|
| best single view | 0.667 | 0.798 | 0.543 |
| **FUSE ECFP+Desc+CGR** | **0.696** | **0.827** | **0.569** |
| **greedy-optimal (6 views)** | **0.718** | 0.834 | 0.542 |

Greedy forward-selection: `descriptors → +drfp → +chemberta → +rxnfp → +secfp → +cgr_morgan`.
**Descriptors are the anchor** (0.667 alone); +DRFP → 0.698; the rest add <0.007 each
(saturating, need held-out confirmation). Notably the ensemble *recruits* the "weak"
learned embeddings (chemberta, rxnfp) for complementary signal.

**Bottom line:** unsupervised multi-view fusion **beats every individual method** (EC3 AMI
0.70–0.72 vs best single 0.667), before any supervision. Next lever = hierarchical-EC
metric learning (rungs 4–5), evaluated zero-shot on held-out EC subclasses.

Figures: `fig_fusion.png`, `fig_compare_ecfp4drfp_vs_rxnfp.png`. Slide deck: `slides/main.pdf`.

# Part III — Four novel representational axes, and why they don't help

The Part I/II pool was 2D substructure fingerprints, physicochemical
descriptors, atom-mapped bond-change (CGR), and SMILES-string language models.
To test whether *genuinely orthogonal* representations carry EC signal the
incumbents miss, we added four axes that each view the reaction differently:

| axis | embedding | model | dim | coverage |
|------|-----------|-------|----:|---------:|
| pharmacophore reduced-graph | `erg_diff` | RDKit ErG (Stiefl 2006) | 315 | 100% |
| pretrained 2D-graph GNN | `gin_diff` | snap-stanford `supervised_contextpred` (5-layer GIN, ZINC15+PPI) | 300 | 76.2% |
| 3D molecular geometry | `unimol_diff` | Uni-Mol `unimolv1` 84M (SE(3), ~209M conformers) | 512 | 98.9% |
| biochemistry-native reaction | `biochem` | REXzyme T5 encoder 335M (32M RHEA reaction→enzyme pairs) | 1024 | 100% |

`erg`/`unimol`/`biochem` are reaction differences (Σproducts − Σreactants) of a
per-molecule representation; `biochem` reads the whole reaction string (like
rxnfp). GIN featurizes only 76% of molecules (its graph featurizer rejects the
rest); Uni-Mol zero-fills ~700 molecules whose ETKDG conformer failed.

## Each novel axis individually underperforms the 2D-structural plateau
Full-set numbers (best config each; retrieval is leave-one-out 1-NN EC3):

| embedding | EC3 AMI (cluster) | 1-NN EC3 (retrieval) |
|-----------|------------------:|---------------------:|
| *plateau top (ecfp6 / ecfp4-drfp)* | *0.628–0.630* | *0.874* |
| **unimol_diff** | **0.593** | 0.752 |
| **erg_diff** | 0.563 | **0.822** |
| **biochem (REXzyme)** | 0.468 | 0.783 |
| **gin_diff** | 0.443 | 0.656 |

- **Uni-Mol** is the strongest novel axis on *clustering* (0.593, 9th of 25,
  above rdk/drfp/chemberta/molformer) — its 3D-shape clusters are unusually
  *homogeneous* (homogeneity 0.785) — but it is mediocre on retrieval (0.752).
- **ErG** is the strongest novel axis on *retrieval* (0.822) — pharmacophore
  topology transfers EC surprisingly well — but mid-pack on clustering.
- **REXzyme is the headline negative.** A 335M encoder pretrained on 32M curated
  RHEA reaction→enzyme pairs — the *exact* biochemical domain ModelSEED lives
  in — annotates **worse than USPTO-trained rxnfp** (0.783 vs 0.834) and far
  below a plain circular fingerprint (0.874). Biochemistry-native pretraining
  did not buy EC-discriminative structure; its objective (predict a catalysing
  enzyme sequence) is not the objective that separates EC subclasses.
- **GIN** is the weakest, and coverage-limited (76%).

## Cross-axis fusion does not beat same-family fusion — and adding novel axes *hurts*
Re-scored on the fair common subset (intersection of all masks, n=10,240 = 74.6%):

| fusion | EC3 AMI | EC3 15-NN | full-EC 1-NN |
|--------|--------:|----------:|-------------:|
| **Same-family: PhysChem+CGR+ChemBERTa** (incumbent) | **0.710** | **0.825** | **0.565** |
| Cross-axis: PhysChem+CGR+GIN+UniMol+ErG | 0.698 | 0.822 | 0.559 |
| Incumbent **+ Uni-Mol** (best novel axis) | 0.705 | 0.824 | 0.545 |
| Incumbent **+ all 4 novel** | 0.700 | 0.824 | 0.559 |

Fair-subset single views: PhysChem 0.666 > **Uni-Mol 0.624** > CGR 0.614 >
ErG 0.571 > GIN 0.529.

- The all-novel cross-axis fusion **loses** to the same-family incumbent
  (−0.012).
- Crucially, **augmenting the incumbent winner with the best novel axis hurts
  (−0.004), and with all four hurts more (−0.010).** Unlike Part II — where the
  fusion *recruited* the "weak" ChemBERTa/rxnfp for complementary signal — these
  orthogonal axes add no EC signal the descriptor+CGR+ChemBERTa trio lacks; they
  only add variance that TruncatedSVD then spends dimensions on.

## Verdict: representational diversity ≠ task-relevant diversity
The four axes are genuinely different *representations* (pharmacophore topology,
learned graph, 3D shape, enzyme-supervised reaction language) yet none carries
EC-discriminative information the 2D structural/bond-change views miss. EC
subclass is, operationally, a **substructure/bond-change** label; once the
descriptor + CGR + SMILES-transformer trio captures that, orthogonal views of
shape, pharmacophore, or catalysis are redundant-at-best. The ~0.63 clustering /
~0.87 retrieval plateau is therefore a property of the **EC taxonomy vs. the
reaction set**, not a ceiling of any one encoder family — a broader and more
exotic representation pool does not move it.

**Deployment implication:** the FAISS EC-annotation index should stay on the
cheap 2D structural/physchem fusion; the 3D-conformer (Uni-Mol) and 335M-encoder
(REXzyme) costs buy nothing here. The lever remains the *annotation rule*
(threshold+abstain, Part on annotation), not a richer embedding.

Figures: `fig_crossaxis_fusion.png` (9-bar single/fusion comparison),
`fig_best_per_embedding.png` + `fig_knn_transfer.png` (now include all 25
embeddings). Scripts: `pipeline/01_embed/embed_erg.py`, `pipeline/01_embed/embed_gin.py`, `pipeline/01_embed/embed_unimol.py`,
`pipeline/01_embed/embed_biochem.py`, `pipeline/04_fusion/fusion_crossaxis.py`.

*Method note:* the new embeddings' clustering rows were computed in the
`modelseed` env (which has `leidenalg`/`hdbscan`); the full algorithm sweep
(incl. leiden/louvain/hdbscan) is present for all four. `butina` is intentionally
absent (it runs only on the three raw-bit fingerprints, by design).

## Part IV — two last off-the-shelf shots at the plateau (both miss)
After the four novel *axes* failed (Part III), we tested the two remaining
off-the-shelf candidates with the strongest *a-priori* reason to separate EC
better than the incumbents. Neither does; both land in or below the plateau,
so the sweep now covers **27 embeddings**.

| new embedding | motivation | EC3 AMI | rank | kNN gated EC3 |
|---------------|------------|--------:|-----:|--------------:|
| **FCFP4** (functional-class Morgan, `useFeatures`) | atoms → pharmacophore roles; EC subclasses *are* functional-transformation labels, so a functional-class circular FP is the untried substructure axis best aligned with the label | 0.618 | 8 / 27 | 0.899 @ 76% |
| **Mordred** (1613 2D descriptors) | 7× the RDKit-217 `descriptors_diff` that already ties for best (0.624) — more descriptors in the winning family | 0.585 | 13 / 27 | 0.885 @ 72% |

- **FCFP4 ties exact-atom ECFP, no gain.** Abstracting atoms to pharmacophore
  roles does not separate EC any better than exact atom+degree — it is another
  point in the 0.61–0.63 band, and its gated kNN (0.899 @ 76%) is
  indistinguishable from the incumbent (0.90 @ 76%). The last untried
  *substructure* stone confirms the plateau even for the label-aligned axis.
- **Mordred lands *below* RDKit-217 descriptors (0.585 < 0.624).** 7× more
  descriptors *hurt* — the same failure mode as Part III's cross-axis fusion:
  the extra ~1400 features add variance, not EC signal, and TruncatedSVD then
  spends dimensions on the noise. Feature count is not the lever.

**Reinforced verdict.** The plateau is not "we haven't found the right
off-the-shelf embedding yet." A functional-class abstraction and a 7×-richer
descriptor basis — the two most label-aligned untried candidates — both fail to
move ~0.63. The remaining unexplored lever is *supervision* (a hierarchical-EC
metric learned with held-out EC eval), which is the only thing that can
distinguish a representation ceiling from an information ceiling; every
*unsupervised, off-the-shelf* stone now points to the latter.

Scripts: `pipeline/01_embed/embed_classical.py` (FCFP4 — plus torsion/rdk/avalon, which turned
out already benchmarked as `torsion_diff`/`rdk_diff`/`avalon_diff`),
`pipeline/01_embed/embed_mordred.py` (needs `mordredcommunity`, pip-installed into `palm`).

## Part V — Irreducible-error probe: the EC3 plateau is *sparsity*, not an information ceiling
`pipeline/05_analysis/irreducible_error.py` measures the Bayes floor — where two reactions
share a structure but carry different EC, no structural model can be right on both.

**(A) Exact-structure collisions (representation-free).** Of 13,721 reactions, 904
share their exact `rxn_smiles` with another. Among those, label conflicts are
**rare at EC3 and coarser**:

| level | reactions with an exact twin of *different* label | floor accuracy |
|-------|--------------------------------------------------:|---------------:|
| EC1   | 11 (0.08%) | 0.9992 |
| EC2   | 17 (0.12%) | 0.9988 |
| EC3   | 59 (0.43%) | 0.9957 |
| full EC | 337 (2.46%) | 0.9754 |

Literally-identical reactions almost always share EC3 (99.6%). So exact-collision
label noise does **not** explain the 0.87 retrieval / 0.63 clustering plateau.

**(B) Conflict vs structural similarity.** For every reaction we take its 1-NN in
the `ecfp4_diff` geometry and record (cosine similarity, EC match). The conflict
probability is a **steep function of similarity** (`fig_irreducible_error.png`):

| 1-NN cosine | EC3 conflict | EC2 | EC1 |
|-------------|-------------:|----:|----:|
| ≥ 0.99 (near-identical, 65% of reactions) | **0.036** | 0.021 | 0.011 |
| ~0.6 (mid) | ~0.65 | ~0.56 | ~0.37 |
| ~0.4 (far / sparse) | ~0.86 | ~0.82 | ~0.50 |

**Interpretation — this reverses the "information ceiling" reading.** Where a close
structural neighbour exists, structure determines EC3 with **96–99.6%** fidelity.
The 0.87 EC3 retrieval ceiling is therefore driven by reactions in **sparse
regions** — those whose nearest structural analogue is far — not by irreducible
label noise. **There is headroom:** the lever is either (i) *densifying* the space
so more reactions have a close same-EC neighbour — exactly what **supervised
hierarchical-EC metric learning** does — or (ii) more reference data. No off-the-
shelf embedding moved the plateau (Parts I–IV) because they all inherit the same
sparsity; a supervised metric attacks the sparsity directly.

**The one genuine information ceiling is full 4-level EC.** Even among near-identical
structures (cos ≥ 0.99) the full-EC conflict is **35%** (2.5% at exact identity):
the 4th digit (substrate/serial specificity) is often simply **not encoded in the
reaction structure** — which is why full-EC accuracy caps at ~0.49 everywhere and
no method fixes it. Report EC3 as the structural-prediction target; treat the 4th
digit as enzyme-side information.

Outputs: `results/irreducible_error.csv`, `figures/fig_irreducible_error.png`,
`pipeline/05_analysis/irreducible_error.py`.

---

# Part VI — Deployment: predicting EC for the reactions ModelSEED is missing one

Parts I–V benchmark the method by *hiding* known labels. Part VI runs the real
inference: the validated best retriever (`ecfp4_drfp`, cosine, confidence-gated
15-NN) applied to the **22,871 non-obsolete ModelSEED reactions that carry no full
EC**, using the 13,721 EC-labelled reactions as the reference index.

| bucket | count | share |
|--------|------:|------:|
| **confident EC3 call** (agreement ≥ 0.6, in-domain) | 11,513 | 50.3% |
| in-domain but low-agreement | 4,891 | 21.4% |
| **OOD / abstained** (far from any labelled reaction) | 6,467 | 28.3% |

The 28% OOD + 21% low-agreement split is the real finding: much of unannotated
ModelSEED sits *outside* the enzyme-labelled manifold (transport, spontaneous,
unusual chemistry), and the tool correctly refuses those rather than forcing a guess.
Confident calls span 85 EC3 subclasses (transferases 34%, oxidoreductases 32%,
hydrolases 25% lead). Scripts `pipeline/06_deploy/annotate_missing.py` (compute) +
`viz_predictions.py` (figures). Outputs `results/predicted_missing_ec.tsv`,
`figures/fig_missing_landscape.png`, `fig_pred_map.png`, `fig_pred_distributions.png`.

## Validating the predictions (don't just assume the benchmark number)
`pipeline/06_deploy/validate_predictions.py` checks the confident calls two ways
instead of assuming the 0.90 held-out accuracy carries over:

- **Exact-twins vs genuine.** 1,843 of the 11,513 confident calls (16%) have an
  *identical* labelled reaction (cosine = 1.0) — trivial label propagation, not
  prediction. The remaining **9,670 (84%) are genuine novel predictions.**
- **Distribution-matched accuracy.** Reweighting labelled-set gated accuracy by the
  *missing set's own* neighbour-similarity distribution gives an expected EC3
  accuracy of **0.78 (all confident) / 0.74 (genuine non-twin)** — **not** the 0.90
  benchmark average. The missing predictions skew toward lower-similarity
  neighbourhoods, so their true accuracy is meaningfully lower. Report ~0.74–0.78,
  not 0.90, as the honest expected accuracy on this population.

Output `results/prediction_validation.csv`, `figures/fig_prediction_validation.png`.

---

# Part VII — How robust are these numbers? (variance, multi-label, seeds)

Parts I–VI report point estimates; Part VII puts error bars on them.
`pipeline/05_analysis/variance_analysis.py`, `pipeline/03_retrieval/multilabel_eval.py`.

**Retrieval CIs (bootstrap of per-reaction LOO correctness, B=1000).** The top
structural pack is a *statistical tie*: `ecfp6_diff` 0.876 [0.871–0.881],
`ecfp4_drfp` 0.874 [0.869–0.880], `ecfp4_diff` 0.874 [0.869–0.880] — CIs fully
overlap. `rxnfp` 0.834 [0.828–0.840] and `chemberta_diff` 0.853 are clearly
*below* (non-overlapping). Gated EC3 accuracy 0.905 [0.899–0.911] — the 0.90
headline is real and tight.

**Clustering CIs (subsample n=6000, B=30).** The plateau has *real structure*, not
uniform noise: `ecfp4_drfp` 0.613 [0.600–0.621] ≈ `ecfp6_diff` 0.606 [0.596–0.614]
sit significantly **above** `descriptors_diff` 0.571 and `ecfp2_diff` 0.568
(non-overlapping). So "choice of fingerprint barely matters" holds only *within* the
top ECFP/MHFP tier — the wider gaps are genuine.

**Multi-label EC.** 6.5% of reactions carry >1 full EC; scoring only the first EC
understates accuracy by **+0.3–0.5 pp (EC3)** and **+0.7–1.0 pp (full-EC)** under
any-match scoring — real but minor label noise. `results/multilabel_eval.csv`,
`fig_multilabel.png`.

**Metric learning, 5 seeds (reframed).** The honest question is transfer to *unseen*
classes, and the answer is null: held-out **classes** EC3-AMI Δ (metric − baseline)
= **−0.000 ± 0.052 → within noise**. The large held-out **reactions** gain
(**+0.254 ± 0.012**) is in-distribution densification of classes the metric was
trained on — expected by construction, not transfer. Lead with the null result;
the 0.87 is not evidence the metric generalises. `results/metric_learning.csv`
(mean/std over seeds), `fig_metric_learning.png` (bars ± std).

**Environment.** Pinned in `environment.txt`; note `drfp` requires `numpy≥2.2` but
the sklearn 1.2.2 ABI needs `numpy 1.26.4` — keep numpy 1.26.4 (drfp runs fine on it).

# Part VIII — Independent cross-check against CARE (reaction→EC)

Does an *independent, differently-built* reaction→EC predictor agree with this retriever?
CARE ([Yang et al., NeurIPS 2024](https://arxiv.org/abs/2406.15669)) provides a curated
reaction→EC reference (`reaction2EC.csv`, **61,766** pairs, SwissProt/Rhea-derived). We ran
CARE's **Similarity baseline** (DRFP, k=15 majority) — the *same retrieval protocol* as ours —
so the only differences are the **reference DB** (CARE-curated 61,766 vs our ModelSEED 13,721)
and the **featurisation** (DRFP-only vs ECFP4-diff‖DRFP). Agreement then measures whether the
retriever's calls survive changing the reference and the fingerprint.
Script: `pipeline/06_deploy/care_cross_check.py` (needs CARE cloned at `../tools/CARE`);
per-reaction output `results/care_vs_rxnfp_agreement.tsv`.

Run on the **11,513** missing-EC ModelSEED reactions this retriever scored confidently:

| level | agreement (all 11,513) | agreement (both predictors confident, n=7,533) |
|---|---|---|
| EC1 (main class) | 78.0% | **87.9%** |
| EC2 | 65.5% | 78.3% |
| EC3 (subclass) | 61.5% | **75.9%** |

EC3 has 261 populated classes → chance agreement <1%, so **76% EC3 / 88% EC1 is strong
cross-source corroboration**: the retriever's calls are not an artefact of the ModelSEED
reference — an independently-curated predictor lands on the same enzyme class most of the time,
and nearly always on the main class.

**Agreement calibrates, and the yield is real.** Gating both predictors by their neighbour-
agreement confidence, EC3 agreement climbs monotonically **62% → 76% → 93% → 95%** at gates
0/0.6/0.9/1.0 — so the confidence score is meaningful and the two methods converge where both
are sure. Cascading the gate as an annotation funnel: **22,871** missing-EC reactions → **11,513**
our model confident+in-domain → **7,533** where CARE is also confident → **5,721 where both
predictors agree at EC3** — a *double-validated* EC-subclass annotation set (mean neighbour
confidence 0.94), a concrete deliverable stronger than either predictor alone. Figure
`figures/care_cross_check.png` (script `pipeline/06_deploy/care_headtohead.py`).

**Caveat (honest):** both methods use DRFP, so they are not *fully* independent — CARE's
**CREEP** (learned reaction↔protein contrastive model, no shared featurisation) would be the
truly independent test, and is the worthwhile follow-up. The reference DBs do differ
substantially, so this is still a meaningful check.

**Where it bites — and why the *reaction* beats the *fingerprint*.** On a downstream use
(13 predicted-EC mass/charge-balance corrections in `mass_charge_balance/`), the two agree
**13/13 at EC1** but only **4/13 at EC3**: CARE calls several O₂-consuming oxidoreductases
**1.6.3** (NAD(P)H oxidase) or **1.1.1** (dehydrogenase) where this retriever says **1.14.13/14**
(monooxygenase). But the disagreement is **resolvable from the balanced reaction**, and it
resolves *against CARE*: every one of the 9 disputed reactions consumes **O₂ + NAD(P)H**,
produces **H₂O and never H₂O₂**, and **transforms the organic substrate** — the definitional
monooxygenase signature (one O of O₂ into the substrate chemistry, the other to water). A 1.6.3
oxidase would leave the substrate untouched and make H₂O₂; a 1.1.1 uses no O₂. So **rxnfp's 1.14
is correct on all 9** (incl. CYP79 amino-acid→aldoxime and CYP51 sterol-demethylase cases where
CO₂/formate loss fools a naive substrate-O count). CARE mis-calls them because **DRFP is
dominated by the large NAD(P)H↔NADP cofactor substructure**, so its nearest neighbours are
cofactor-matched but chemistry-mismatched (NAD(P)H oxidases share that exact moiety).
Lesson: **cross-predictor agreement is a good confidence gate at EC1–EC2, but at EC3 the
fingerprint predictors share a cofactor-substructure bias — reading the O₂ fate in the balanced
reaction adjudicates the disagreements the fingerprints can't.**

**When they disagree, which is right? (ground-truth test.)** The missing-EC set has no ground
truth, so we ran BOTH predictors on the **13,721 labelled** reactions (ModelSEED's curated EC as
truth): our model leave-one-out 15-NN over the same 13,721 (ECFP4-diff‖DRFP, self excluded);
CARE 15-NN over its 61,766 reference (exact-duplicate leakage, top-1 sim ≥ 0.999, dropped —
so if anything the test favours CARE). On the 5,105 both-confident reactions, among the
disagreements: **our model is correct ~80% vs CARE ~8–15%** (EC1 80/15, EC2 84/8, EC3 81/8;
remainder both-wrong). Overall confident accuracy: our model 96/95/93 vs CARE 89/79/76
(EC1/2/3). **Two caveats, one measured.** (1) *Near-duplicate leakage:* our model is a
parameter-free retriever whose reference IS the labelled set; LOO removes the reaction itself but
**not near-duplicates**, and **33–36% of the 13,721 have a twin at cosine ≥ 0.99** (ModelSEED is
redundant). Stripping reactions where our model has such a twin, the EC3 disagreement win deflates
**81%→73%** and overall EC3 acc **93%→90%** (CARE ~11%/77%, unchanged) — real but robust. (2)
*Home-DB advantage (unremovable here):* ground truth is ModelSEED's *own* curation and our model
retrieves from ModelSEED neighbours, so even the leakage-free 90% is ModelSEED-EC-from-ModelSEED;
some "CARE wrong" cases are defensible SwissProt/Rhea ECs. So: for annotating *ModelSEED*
reactions our model is the one to trust (home alignment is the goal), but this is **not** a fair
general "rxnfp > CARE" benchmark. Scripts `pipeline/06_deploy/care_who_is_right.py`
(+ `tools/CARE_compare/who_is_right.py` for the leakage split), per-reaction
`results/care_who_is_right.tsv`. Refines the "no uniform winner" reading: the disagreement
*structure* is symmetric, but on ModelSEED ground truth our model resolves them correctly far
more often (leakage-corrected 73% vs 11%).
