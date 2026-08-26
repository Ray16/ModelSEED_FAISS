# Pipeline

Scripts are grouped by stage; each stage folder is numbered to give run order.
Individual scripts are named for what they do (no numeric prefixes). Every script
is standalone (`python pipeline/<stage>/<script>.py`) and resolves the project
root as three directories up, so it must stay at `pipeline/<stage>/<file>.py`
depth. Shared code lives in `../_lib/` (`_style.py` = the 18pt figure rule).

Benchmark env: **palm** (`/homes/rzhu/miniforge3/envs/palm/bin/python`).
New-embedding scripts may need the **modelseed** env (leidenalg/hdbscan/etc.).

| stage | scripts | what it produces | RESULTS.md |
|-------|---------|------------------|-----------|
| `00_data/` | `build_dataset.py` | `data/dataset.parquet` (13,721 EC-labelled reactions) | intro |
| `01_embed/` | `embed_fingerprints.py`, `embed_rdkit.py`, `embed_rxnfp.py`, `embed_extra.py`, `embed_learned.py`, `embed_rxnfp_diff.py`, `embed_cgr.py`, `embed_erg.py`, `embed_gin.py`, `embed_unimol.py`, `embed_biochem.py`, `embed_classical.py`, `embed_mordred.py` | `embeddings/*.npy` (27 embeddings; gitignored, regenerable) | Parts I–IV |
| `02_cluster/` | `cluster_eval.py`, `consolidate.py` | `results.csv`, `summary_*.csv`, cluster figures | Part I |
| `03_retrieval/` | `knn_annotation.py` | `knn_results.csv` (kNN EC-transfer) | Part I |
| `04_fusion/` | `fusion_eval.py`, `fusion_search.py`, `cgr_subset_eval.py`, `fusion_figure.py`, `triple_combos.py`, `triple_figure.py`, `fusion_crossaxis.py` | fusion figures/tables | Parts II–III |
| `05_analysis/` | `visualize_tsne.py`, `compare_two.py`, `irreducible_error.py`, `metric_learning.py` | t-SNE maps, Bayes-floor probe, supervised metric | Part V |
| `06_deploy/` | `annotate_missing.py`, `viz_predictions.py` | `predicted_missing_ec.tsv`, prediction figures | Part VI (deployment) |

(An early `cluster_eval` and `report` were superseded by `02_cluster/cluster_eval.py`
and `consolidate.py` and removed; recover them from git history if ever needed.)

The EC-annotation-rule study (threshold+abstain, back-off, conformal, OOD) lives
in its own self-contained module: `../annotation/` (see `annotation/RESULTS.md`).

## Typical order

```bash
PY=/homes/rzhu/miniforge3/envs/palm/bin/python
$PY pipeline/00_data/build_dataset.py
$PY pipeline/01_embed/embed_fingerprints.py   # + the other embed_*.py as needed
$PY pipeline/02_cluster/cluster_eval.py
$PY pipeline/02_cluster/consolidate.py        # -> results.csv, summaries, figures
$PY pipeline/03_retrieval/knn_annotation.py   # -> knn_results.csv
# analysis / deployment as needed:
$PY pipeline/05_analysis/metric_learning.py
$PY pipeline/06_deploy/annotate_missing.py    # predict EC for unannotated reactions
$PY pipeline/06_deploy/viz_predictions.py
```
