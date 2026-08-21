# EC annotation — beyond fixed-k majority vote

**Question (from the sparse-space concern).** Fixed-k kNN with equal-weight
majority voting uses neighbour *rank*, not *similarity*. In a sparse region a
query still gets k neighbours, but the far ones vote with full weight and pollute
the majority — so we emit a confident-but-wrong EC. Can we do better, and can the
method know when it *shouldn't* answer?

**Protocol.** A **reference/test split** (80/20, seed 0), neighbours drawn only
from the reference pool — so "sparse" queries (true EC rare or absent in the
reference) genuinely occur, unlike leave-one-out. Same geometry as the benchmark
(L2 → SVD-256 → L2, cosine). Reference = 10,977, test = 2,744. Primary level EC3.
Everything is self-contained in `annotation/` (does not touch the main benchmark).
`palm` env. Run: `python annotation/annotate.py --emb ecfp4_diff descriptors_diff`.

## Results (ecfp4_diff, EC3, held-out split)

| method | EC3 accuracy | coverage | note |
|--------|-------------:|---------:|------|
| **M1** baseline equal-weight (k=15) | 0.762 | 100% | current rule (scripts/08) |
| **M2** similarity-weighted (k=15) | **0.768** | 100% | strictly ≥ M1 everywhere — *free* |
| **M3** threshold τ + abstain | **0.903** | 81% | τ=0.9: answer only where a similar reaction exists |
| **M4** hierarchical back-off (EC3→EC2→EC1) | **0.823** | 96% | correct-at-reported-level; falls back when EC3 is unsafe |
| **M5** conformal set (α=0.1) | 0.898 coverage | 100% | prediction *sets*, calibrated to 90% |
| **M6** OOD-abstain (top-1 dist) | 0.860 | 70% | novelty score; AUROC(error)=0.72 |

Supporting operating points: at the coverage where accuracy first reaches 0.90,
equal-weight covers **71%**, sim-weighted **73%**; the explicit threshold gate does
better (**0.90 at 81%**) because a similarity floor is a smarter gate than
confidence-ranking a fixed-k vote. `descriptors_diff` behaves the same (M3: 0.926
at 77%).

## Verdict — yes, fixed-k struggles in sparse regions, and the fixes work

1. **Similarity-weighting is a free win.** Weighting each neighbour by cosine
   similarity dominates equal-weight at every coverage (the far 15th neighbour
   stops mattering). No reason to keep equal-weight. *(fig_accuracy_coverage)*
2. **Threshold + abstain is the direct fix for sparsity.** Requiring at least one
   neighbour within τ, and abstaining otherwise, lifts EC3 accuracy from 0.76 (at
   100%) to **0.90 at 81% coverage** — exactly the sparse-query failures removed.
   *(fig_threshold_sweep)*
3. **Hierarchical back-off never leaves you empty-handed.** When EC3 is unreliable
   it reports EC2 or EC1 (denser, easier), giving **0.82 correct-at-reported-level
   over 96%** of queries. EC-tree-native. *(fig_backoff)*
4. **Conformal sets are calibrated.** At α=0.1 the sets contain the true EC3
   **89.8%** of the time (≈ the 90% target) — a principled uncertainty layer whose
   set grows in sparse regions. *(fig_conformal)*
5. **Novelty is detectable.** Top-1 distance predicts error with **AUROC 0.72**
   (LOF weaker, 0.68); abstaining on the most-novel 30% raises accuracy to 0.86.
   *(fig_ood_abstain)*

## Recommended production annotator

Retrieve within a **similarity threshold τ** → **similarity-weighted** vote →
**abstain** if no neighbour clears τ → **hierarchical back-off** to the finest EC
level the local evidence supports, with **conformal sets** as the uncertainty
layer. Select τ/θ on the **accuracy-vs-coverage curve**, not a single accuracy.
This replaces the fixed-k=15 equal-weight rule everywhere.

*Not pursued:* a trained per-class classifier — 3,290 of 5,622 full ECs are
singletons, so full-EC classification is ill-posed and a softmax head can't help
the open-set / novel-reaction case that retrieval + back-off handle directly.

## Files
`annotate.py` (all methods + figures) · `results/method_comparison_all.csv` and
per-method CSVs (`m3_threshold_*`, `m4_backoff_*`, `m5_conformal_*`, `m6_ood_*`) ·
`figures/fig_{accuracy_coverage,threshold_sweep,backoff,conformal,ood_abstain}_*.png`
(all 18pt). Evaluated on `ecfp4_diff` and `descriptors_diff`; add any embedding via
`--emb`.
