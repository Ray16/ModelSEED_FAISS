# Where everything lives — artifact & structure locations

All paths are absolute, under `/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/rxnfp_clustering/query_amide_synthesis_reaction/`. Cofold model PDBs are
gitignored (regenerable, GBs); the tables/figures/manifests/MSAs indices are tracked.
Each model PDB is at `<run>/outputs/<ACC>_<N>mg/by_seed/<ACC>_<N>mg/s42/<ACC>_<N>mg_model_{0..49}.pdb`
(+ sibling `confidence_*.json`), where `<run>` is one of the absolute run dirs below.

## Cofolded STRUCTURES (PDBs, on disk — gitignored)

STEP 1 (protein + ATP + Mg + adipate; activation):
- Expanded expressible (86 enzymes, 4300 pdb): `/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/rxnfp_clustering/query_amide_synthesis_reaction/results/top10_cofold_expanded/campaign/outputs/`
- Expanded hard-host (25 enzymes, 1250 pdb): `/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/rxnfp_clustering/query_amide_synthesis_reaction/results/top10_cofold_expanded/campaign_hard/outputs/`
- Original 13-panel: `/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/rxnfp_clustering/query_amide_synthesis_reaction/results/top10_cofold/campaign/outputs/` — 3 real dirs (I3R584, Q9KZC7/GlnA3,
  P39641/BacD) + 10 SYMLINKS resolving to `/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/rxnfp_clustering/query_amide_synthesis_reaction/results/boltz2_activation_20261016/outputs/`
- Reused reference folds canonical store: `/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/rxnfp_clustering/query_amide_synthesis_reaction/results/boltz2_activation_20261016/outputs/`

STEP 2 (acyl intermediate + neutral-amine HMDA + Mg [+ADP]; accommodation):
- Original 13-panel (13 enzymes, 650 pdb): `/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/rxnfp_clustering/query_amide_synthesis_reaction/results/top10_cofold_step2/outputs/`
- Expanded, all new candidates (111, folding): `/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/rxnfp_clustering/query_amide_synthesis_reaction/results/top10_cofold_expanded/step2/outputs/`

## MSAs (gitignored; indices tracked)
- Expressible 86: `/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/rxnfp_clustering/query_amide_synthesis_reaction/results/top10_cofold_expanded/msa/` (index.tsv: id→csv)
- Hard-host 25: `/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/rxnfp_clustering/query_amide_synthesis_reaction/results/top10_cofold_expanded/msa_hard/` (SEPARATE dir — see pitfall in STATUS.md)
- Original panel: `/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/rxnfp_clustering/query_amide_synthesis_reaction/results/top10_cofold/msa/`, `/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/rxnfp_clustering/query_amide_synthesis_reaction/results/boltz2_activation_20261016/msa/`

## Manifests / run state (per run dir)
- `…/manifest.json` (inputs, sha256, settings), `…/state/*.done.json` (per-ensemble geometry+poses),
  `…/inputs/*.yaml` (cofold configs), `…/logs/`.

## Derived tables & figures (tracked, on GitHub)
- Pool definition: `/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/rxnfp_clustering/query_amide_synthesis_reaction/results/top10_cofold_expanded/pool.tsv` (124 candidates, status column)
- Step-1 ranking: `/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/rxnfp_clustering/query_amide_synthesis_reaction/results/top10_cofold_expanded/step1_pool_ranking.{tsv,png}` (within-family)
- Step-2 panel (original 13): `/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/rxnfp_clustering/query_amide_synthesis_reaction/results/top10_cofold_step2/step2_near_attack.{tsv,json}`,
  `step2_pose_clouds.png`, `step1_step2_combined.{tsv,png}`
- Mapping evidence: `/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/rxnfp_clustering/query_amide_synthesis_reaction/results/top10_mapping/panel_evidence_consolidation.{tsv,json,png}`
- Shortlist: `/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/rxnfp_clustering/query_amide_synthesis_reaction/results/top10_mapping/expansion_shortlist.tsv`

## Code
- Builders: `/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/rxnfp_clustering/query_amide_synthesis_reaction/amide_synthesis/structure_modeling/{campaign,step2_build,step2_build_expanded}.py`
- Runners (lambda13, gated): `/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/rxnfp_clustering/query_amide_synthesis_reaction/amide_synthesis/structure_modeling/{fleet,step2_fleet}.py`
- Analysis: `/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/rxnfp_clustering/query_amide_synthesis_reaction/amide_synthesis/evaluation/{expanded_step1_ranking,step2_geometry,combined_geometry,evidence_consolidation}.py`
- Candidate selection: `/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/rxnfp_clustering/query_amide_synthesis_reaction/amide_synthesis/candidate_selection/expansion_shortlist.py`

See STATUS.md for pipeline-stage status and caveats.
