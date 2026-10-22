# Where everything lives — artifact & structure locations

All paths relative to `query_amide_synthesis_reaction/`. Cofold model PDBs are
gitignored (regenerable, GBs); the tables/figures/manifests/MSAs indices are tracked.
Pose files follow `…/outputs/<ACC>_<N>mg/by_seed/<ACC>_<N>mg/s42/<ACC>_<N>mg_model_{0..49}.pdb`
(+ sibling `confidence_*.json`).

## Cofolded STRUCTURES (PDBs, on disk — gitignored)

STEP 1 (protein + ATP + Mg + adipate; activation):
- Expanded expressible (86 enzymes, 4300 pdb): `results/top10_cofold_expanded/campaign/outputs/`
- Expanded hard-host (25 enzymes, 1250 pdb): `results/top10_cofold_expanded/campaign_hard/outputs/`
- Original 13-panel: `results/top10_cofold/campaign/outputs/` — 3 real dirs (I3R584, Q9KZC7/GlnA3,
  P39641/BacD) + 10 SYMLINKS resolving to `results/boltz2_activation_20261016/outputs/`
- Reused reference folds canonical store: `results/boltz2_activation_20261016/outputs/`

STEP 2 (acyl intermediate + neutral-amine HMDA + Mg [+ADP]; accommodation):
- Original 13-panel (13 enzymes, 650 pdb): `results/top10_cofold_step2/outputs/`
- Expanded, all new candidates (111, folding): `results/top10_cofold_expanded/step2/outputs/`

## MSAs (gitignored; indices tracked)
- Expressible 86: `results/top10_cofold_expanded/msa/` (index.tsv: id→csv)
- Hard-host 25: `results/top10_cofold_expanded/msa_hard/` (SEPARATE dir — see pitfall in STATUS.md)
- Original panel: `results/top10_cofold/msa/`, `results/boltz2_activation_20261016/msa/`

## Manifests / run state (per run dir)
- `…/manifest.json` (inputs, sha256, settings), `…/state/*.done.json` (per-ensemble geometry+poses),
  `…/inputs/*.yaml` (cofold configs), `…/logs/`.

## Derived tables & figures (tracked, on GitHub)
- Pool definition: `results/top10_cofold_expanded/pool.tsv` (124 candidates, status column)
- Step-1 ranking: `results/top10_cofold_expanded/step1_pool_ranking.{tsv,png}` (within-family)
- Step-2 panel (original 13): `results/top10_cofold_step2/step2_near_attack.{tsv,json}`,
  `step2_pose_clouds.png`, `step1_step2_combined.{tsv,png}`
- Mapping evidence: `results/top10_mapping/panel_evidence_consolidation.{tsv,json,png}`
- Shortlist: `results/top10_mapping/expansion_shortlist.tsv`

## Code
- Builders: `amide_synthesis/structure_modeling/{campaign,step2_build,step2_build_expanded}.py`
- Runners (lambda13, gated): `amide_synthesis/structure_modeling/{fleet,step2_fleet}.py`
- Analysis: `amide_synthesis/evaluation/{expanded_step1_ranking,step2_geometry,combined_geometry,evidence_consolidation}.py`
- Candidate selection: `amide_synthesis/candidate_selection/expansion_shortlist.py`

See STATUS.md for pipeline-stage status and caveats.
