# Session handoff — rxnfp_clustering (2026-08-21)

Branch: **`rxnfp-clustering-benchmark`** (off main). Benchmark env: **palm**
(`/homes/rzhu/miniforge3/envs/palm/bin/python`). New-embedding env: **modelseed**.
Figure rule (enforced everywhere): **all text ≥ 18pt** via `scripts/_style.py`
(`import _style`); also in repo-root `CLAUDE.md`.

## What this project is
Benchmark of reaction embeddings × clustering for clean EC-number groups on 13,721
ModelSEED reactions (261 EC3 subclasses). Primary clustering metric = EC3 AMI;
deployment target = a FAISS retrieval index for EC annotation transfer. Full
results in `RESULTS.md` (21 embeddings, best single ≈ EC3 AMI 0.63; multi-view
fusion ≈ 0.70–0.72; retrieval ≫ clustering for annotation).

## Done this session (committed)
- `e739c7d` **All 11 figures → 18pt** (shared `scripts/_style.py`, overlaps fixed).
  Repo-root `CLAUDE.md` records the 18pt rule.
- `7b02ae8` **Deck recompiled** with 18pt figures (`slides/main.pdf`, 13 frames).
- `5224535` **EC-annotation module** (`annotation/`) — 6 rules vs fixed-k under a
  reference/test split; landscape `fig_triple_combos` + full-width slide 11.
- `9d6751b` **Fusion-annotation test** (`annotation/annotate_fusion.py`).
- `31c838e` **Paired significance** (`annotation/significance.py`).

## Key findings this session
**EC annotation (the "sparse space" question).** Fixed-k=15 equal-weight vote uses
neighbour *rank* not *similarity* → fails in sparse regions. Fixes, EC3, ref/test
split, on `ecfp4_diff`:
- similarity-weighted: strictly ≥ equal-weight (free).
- **threshold τ + abstain: 0.76 → 0.90 EC3 accuracy at 81% coverage** (the big win).
- hierarchical back-off (EC3→EC2→EC1): 0.82 correct-at-reported-level over 96%.
- conformal sets (α=0.1): calibrated to 90%. OOD flag: top-1 dist AUROC 0.72.
Writeup: `annotation/RESULTS.md`.

**Does the fusion annotate better than a single fingerprint? Yes, and it's real.**
(`annotate_fusion.py` + `significance.py`, EC3, paired ref/test split)
- full set, CGR-free fusion (PhysChem+ChemBERTa+ECFP): weighted 0.790 vs 0.766;
  threshold+abstain **0.94 @ 80%** vs 0.90.
- CGR subset, slide-11 fusion (PhysChem+CGR+ChemBERTa): threshold+abstain
  **0.96 @ 81%** vs 0.92 — but only over the ~75% of reactions CGR maps.
- **Significance:** paired McNemar p=4e-5 (full) / 0.012 (subset); bootstrap 95% CI
  excludes 0. So the gain is statistically real.
- **Honest reading:** at full-coverage weighted vote the gain is modest
  (Δacc ~2pp, ~9–10% relative error reduction); at the high-accuracy abstain
  operating points it is a **~35–47% relative error reduction** (e.g. 9.7%→6.3%
  errors). BUT the **rule change (abstain) is the bigger lever** (0.76→0.90) than
  the fusion (0.90→0.94). Prioritise the rule; fusion is a real second-order gain.

**Slides 10 vs 11 (three-view fusion).** Slide 10 = concept build-up featuring one
principled triple ECFP+PhysChem+CGR (0.696). Slide 11 = exhaustive search over all
35 triples; best PhysChem+CGR+ChemBERTa (0.710). They are consistent (slide 10's
triple = slide 11's 0.697 bar). Report **slide 11**; slide 10 is concept-only.
"score" axis on both = generic label over heterogeneous [0,1] metrics (blue=EC3 AMI
clustering; orange=retrieval accuracy; slide 10 green=full-EC 1-NN). Consider
relabelling "AMI / accuracy". **Open TODO:** reconcile slide 10 vs 11 (user leaned
toward "report only slide 11"); not yet edited.

## In progress — 4 novel embeddings (background agent, INCOMPLETE)
Goal: add GIN (learned-graph), ErG (pharmacophore-topology), Uni-Mol (3D),
biochem-native reaction model — genuinely novel axes — then a cross-axis fusion
{descriptors, CGR, GIN, Uni-Mol, ErG}. Scripts `19_embed_erg.py`,
`20_embed_gin.py`, `21_embed_unimol.py`, `22_embed_biochem.py`,
`23_fusion_crossaxis.py`; `09_consolidate.py` already extended with the new
embeddings/families. Status:
- **Done (embeddings produced):** `erg_diff.npy`, `gin_diff.npy`, `biochem.npy`
  (+ masks). **NOT yet committed.**
- **Running:** `scripts/21_embed_unimol.py` in `modelseed` (conformer gen, slow).
  `unimol_diff.npy` not written yet.
- **NOT done:** clustering + kNN integration into `results.csv`/`knn_results.csv`
  for the new embeddings; running `23_fusion_crossaxis.py`; verifying the biochem
  model is a real fine-tune (check `22_embed_biochem.py` — may be placeholder);
  regenerating `09` figures; committing; updating `RESULTS.md` Part III.

## Next steps for the new session
1. Wait for `unimol_diff.npy`, then run clustering/kNN for the 4 new embeddings and
   `09_consolidate.py`; check each novel axis vs the 0.61–0.63 structural plateau.
2. Run `23_fusion_crossaxis.py`; then run `annotation/annotate_fusion.py`-style test
   on the cross-axis fusion to see if it annotates better than the CGR fusion.
3. Verify `22_embed_biochem.py` isn't an un-fine-tuned placeholder.
4. Commit new embeddings' scripts/results/figures (NOT the `.npy` — gitignored) and
   add `RESULTS.md` Part III. Reconcile slides 10/11.
5. `git` index lock is shared with background agents — retry commits.
