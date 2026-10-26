# Archive index

Maintained code is in [`../amide_synthesis/`](../amide_synthesis/); commands are
documented in [CODE_LAYOUT.md](../docs/CODE_LAYOUT.md). Archived implementations
are provenance records and may require their original directory context.

| Directory | Contents |
|---|---|
| `refactor_20261026/` | Loose-script relocation manifest, obsolete five-reaction plot and sampled legacy audit, cleanup verification |
| `refactor_20261019/` | Pre-phase-layout implementation, numbered wrappers, redundant campaign generator, original tests/reports |
| `mapping_refactor_20261020/` | Before-change mappings/code, superseded reports, slides and pre-gate candidate outputs |
| `reproducibility_20261020/` | Frozen reaction selections and replay inputs |
| `retrieval_decision_20261020/` | Earlier retrieval comparisons |
| `selection_rules_before_user_correction_20261020/` | Earlier chemical policies and validated reference tables |
| `top10_verified_source_20261020/` | Verified model-panel source snapshots |
| `modelseed_drfp_selection_before_chemical_review_20261020T170107/`, `shortlist_before_native_diagrams_20261020T171516/` | Earlier retrieval and reaction depictions |
| `enzyme_names_before_20261022T185716Z/`, `candidate_color_before_20261022T201425Z/` | Before-change annotation and plotting outputs |
| `top_candidate_display_before_20261022T200509Z/`, `top_candidate_evaluation_before_20261022T195346Z/`, `top_candidate_evaluation_nine_20261022T200145Z/` | Earlier candidate evaluation displays |

Some large local snapshots and runtime evidence are not versioned. This cleanup
commits moved source, historical files removed from the tracked active tree, and
compact verification records. It does not upload model ensembles or caches.

Reproduction locks actively depend on selected frozen tables here. Keep those
paths stable. Historical runs in `results/` that supply mapping evidence or
reused model ensembles also remain dependencies of the current pipeline.
