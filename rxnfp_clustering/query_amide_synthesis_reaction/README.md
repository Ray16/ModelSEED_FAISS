# Enzyme candidates for adipate–HMDA amide formation

The current panel contains **ten chemically selected ModelSEED reactions**. DRFP–Tanimoto retrieves the top 100 eligible reactions; versioned soft chemical criteria select the ten. All five previously selected reactions remain included without pinning their IDs.

- [Point-by-point response](docs/COLLABORATOR_RESPONSE.md)
- [Chemical selection, component plot and native structures](results/modelseed_nylon_top10/report.html)
- [Reaction-to-enzyme evidence, counts and names](results/top10_mapping/report.html)
- [Cofold geometry plot](results/top10_cofold/activation_geometry.png)
- [Geometry values](results/top10_cofold/activation_geometry.tsv)
- [Verified structure coverage](results/top10_cofold/final_verification.json)

The ten pools contain **1,064 candidate identifiers**, **1,045 exact sequences** and **834 representatives at 95% identity**. Matching reaction annotations cover 12 reference proteins across nine reactions. Glycylglycine formation has no established exact assignment; BacD/YwfE is an explicitly inferred related-reaction candidate. MurE’s symbolic Rhea participant and missing stereochemistry in citrate-derived entries remain qualified.

The reference structure panel has **13 proteins × 50 poses**. Ten ensembles were verified and reused; three new ensembles complete the panel. The complexes contain protein, ATP, Mg and adipate, without HMDA or activated intermediate. Sequence/composition checks and geometric descriptors do not demonstrate amidation or certify biological assembly.

## Reproduce and update

Use the existing scientific Python environment:

```bash
/homes/rzhu/miniforge3/envs/palm/bin/python -m amide_synthesis reproduce
```

This performs an isolated, strictly offline replay from pinned inputs, recomputing the full reaction screen, ranking, chemical selection, evidence-scoped mapping, sequence alignments and clustering. It does not submit predictions. See [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for inputs and environment.

Recheck all existing structures and regenerate the current plots/Markdown:

```bash
python -m amide_synthesis model inspect --run-dir results/top10_cofold/campaign
python -m amide_synthesis evaluate current-panel
```

[PIPELINE.md](PIPELINE.md) explains target-to-reaction retrieval, reaction-to-enzyme mapping and candidate narrowing. [MAPPING_REVIEW.md](MAPPING_REVIEW.md) documents the original mapping defects and evidence qualifications.

## Code and provenance

| Phase | Maintained implementation |
|---|---|
| Reaction encoding and retrieval | `amide_synthesis/reaction_search/`, `rank_modelseed_nylon.py` |
| Soft chemical selection | `reaction_search/chemical_selection.py`, `config/chemical_selection_policy.json` |
| Reaction/protein evidence and snapshots | `amide_synthesis/enzyme_mapping/`, `evaluation/top5_mapping_audit.py` (supports arbitrary declared panel size) |
| Sequence redundancy | `candidate_selection/sequence_redundancy.py` |
| Reference/model preparation and gated execution | `structure_modeling/reference_panel.py`, `campaign.py`, `fleet.py` |
| Reference consistency, reports and model checks | `amide_synthesis/evaluation/`, `structure_modeling/poses.py` |

Paths without a package prefix in this table are relative to `amide_synthesis/`. Configuration, raw responses, sequences, cluster membership, commands and hashes remain available. Inference is never promoted to an exact native-reaction assignment.

Historical five-reaction selections, the separate 50-protein campaign and superseded scripts/reports are retained with provenance. Their outputs must not be presented as results for the current ten-reaction panel. Existing structural campaigns remain active dependencies where their verified models were reused.
