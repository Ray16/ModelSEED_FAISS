# Where everything lives — artifact & structure locations

All paths are absolute. Cofold model PDBs are gitignored (regenerable, GBs); the
manifests/state/tables/figures/MSA indices are tracked.

## Cofolded STRUCTURES — now in exactly TWO folders

Every candidate's cofold lives under one of two consolidated folders, each
self-contained (`outputs/` + `inputs/` + `state/` + `manifest.json`). Each has 124
ensembles × 50 models = 6200 PDBs. Per-model path:
`<folder>/outputs/<ACC>_<N>mg/by_seed/<ACC>_<N>mg/s42/<ACC>_<N>mg_model_{0..49}.pdb`
(+ sibling `confidence_*.json`).

- **STEP 1** (protein + ATP + Mg + adipate; activation) — all 124:
  `/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/rxnfp_clustering/query_amide_synthesis_reaction/results/cofold_step1/`
- **STEP 2** (acyl intermediate + neutral-amine HMDA + Mg [+ADP]; accommodation) — all 124:
  `/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/rxnfp_clustering/query_amide_synthesis_reaction/results/cofold_step2/`

Note on the 10 reused panel step-1 folds: in `cofold_step1/outputs/` these are
**symlinks** into the external reference store
`/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/rxnfp_clustering/query_amide_synthesis_reaction/results/boltz2_activation_20261016/outputs/`.
That store is kept because work under `/homes/rzhu/Enzyme_agent/suggested_experimental_panel/`
depends on it; it is NOT part of the two candidate-screen folders (it also holds 7
extra ensembles unrelated to this pool).

## MSAs (gitignored; indices tracked) — unchanged homes
- Expressible 86: `/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/rxnfp_clustering/query_amide_synthesis_reaction/results/top10_cofold_expanded/msa/` (index.tsv: id→csv)
- Hard-host 25: `/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/rxnfp_clustering/query_amide_synthesis_reaction/results/top10_cofold_expanded/msa_hard/` (SEPARATE dir — see pitfall in STATUS.md)
- Panel/reused: `/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/rxnfp_clustering/query_amide_synthesis_reaction/results/top10_cofold/msa/`,
  `/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/rxnfp_clustering/query_amide_synthesis_reaction/results/boltz2_activation_20261016/msa/`
  (manifests reference MSA CSVs by absolute path; the MSA dirs were not moved.)

## Manifests / run state (per folder)
- `<folder>/manifest.json` (124 inputs; `merged_from` records the historical per-increment
  runs it was consolidated from), `<folder>/state/*.done.json` (per-ensemble geometry+poses),
  `<folder>/inputs/*.yaml` (cofold configs). `cofold_step1/geometry_ranking.tsv` = the
  original panel-13 step-1 geometry table.

## Derived tables & figures (tracked, on GitHub) — analysis home `top10_cofold_expanded/`
- Pool definition: `/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/rxnfp_clustering/query_amide_synthesis_reaction/results/top10_cofold_expanded/pool.tsv` (124 candidates, status column)
- Step-1 ranking: `/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/rxnfp_clustering/query_amide_synthesis_reaction/results/top10_cofold_expanded/step1_pool_ranking.{tsv,png}` (within-family)
- Combined step-1+step-2 (all 124): `/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/rxnfp_clustering/query_amide_synthesis_reaction/results/top10_cofold_expanded/pool_combined.tsv`,
  `pool_combined_within_family.png` (conservative), `pool_combined_cross_family.png` (hypothesis)
- Step-2 geometry (pool-wide): `/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/rxnfp_clustering/query_amide_synthesis_reaction/results/cofold_step2/step2_near_attack.{tsv,json}`,
  `step2_pose_clouds.png`
- Mapping evidence: `/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/rxnfp_clustering/query_amide_synthesis_reaction/results/top10_mapping/panel_evidence_consolidation.{tsv,json,png}`
- Shortlist: `/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/rxnfp_clustering/query_amide_synthesis_reaction/results/top10_mapping/expansion_shortlist.tsv`

## Code
- Builders: `/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/rxnfp_clustering/query_amide_synthesis_reaction/amide_synthesis/structure_modeling/{campaign,step2_build,step2_build_expanded}.py`
  (step-1 source + step-2 output now both point at `cofold_step1`/`cofold_step2`)
- Runners (lambda13, gated): `/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/rxnfp_clustering/query_amide_synthesis_reaction/amide_synthesis/structure_modeling/{fleet,step2_fleet}.py`
- Analysis: `/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/rxnfp_clustering/query_amide_synthesis_reaction/amide_synthesis/evaluation/{expanded_step1_ranking,step2_geometry,combined_geometry,pool_combined,evidence_consolidation}.py`
  (run with the `modelseed` conda env python, which has gemmi+numpy+matplotlib:
  `/nfs/lambda_stor_01/homes/rzhu/miniforge3/envs/modelseed/bin/python -m amide_synthesis.evaluation.pool_combined`)

See STATUS.md for pipeline-stage status and caveats.
