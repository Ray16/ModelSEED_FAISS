# Enzyme candidates for adipate–HMDA amide formation

The current panel contains **ten chemically selected ModelSEED reactions**. DRFP–Tanimoto retrieves the top 100 eligible reactions; versioned soft chemical criteria select the ten. All five previously selected reactions remain included without pinning their IDs.

- [Point-by-point response](docs/COLLABORATOR_RESPONSE.md)
- [Chemical selection, component plot and native structures](results/modelseed_nylon_top10/report.html)
- [Reaction-to-enzyme evidence, counts and names](results/top10_mapping/report.html)
- [Cofold geometry plot](results/top10_cofold/activation_geometry.png)
- [Geometry values](results/top10_cofold/activation_geometry.tsv)
- [Verified structure coverage](results/top10_cofold/final_verification.json)
- [Per-mapping evidence consolidation (reliability-weighted)](results/top10_mapping/panel_evidence_consolidation.tsv)
- [Step-2 HMDA near-attack accommodation](results/top10_cofold_step2/step2_near_attack.tsv)

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

## Evidence and limitations

State these plainly wherever the panel or its ranking is presented.

**Mapping is multi-source where possible, and flagged where not.** Every exact
mapping rests on a UniProt reaction annotation and is corroborated by independent
axes — primary ECO:0000269 literature, independent curated reaction databases, and
EC-class agreement. Sources are not equally reliable: Rhea (expert-curated,
ChEBI-structured) ranks above MetaCyc/KEGG/EcoCyc, above genome-scale models, above
the MetaNetX aggregator, which is the least reliable and never counts as
corroboration. All twelve exact mappings are multi-source; `rxn05123` is strongest
(four curated databases). BacD/YwfE for glycylglycine (`rxn09897`) is
`inference_only`: it is a well-characterised dipeptide ligase for its *own* reaction
but has no exact native assignment to the target, and must not be presented as one.
See the [evidence consolidation](results/top10_mapping/panel_evidence_consolidation.tsv).

**Step 1 models acid activation only.** The activation-geometry ranking describes how
well adipate is positioned for adenylylation/phosphorylation, and is the geometry that
discriminated active amide synthetases on the experimental panel. It says nothing on
its own about amide-bond formation.

**Step 2 tests conditional amine accommodation, not activity.** The step-2 cofolds place
the activated acyl intermediate (acyl-adenylate + 1 Mg, or acyl-phosphate + ADP + 2 Mg,
per the enzyme's mechanism) with HMDA carrying one neutral nucleophilic amine, to ask
whether the C6 diamine can reach the activated carbonyl — relevant because native
amine-role substrates here span ammonia-sized to large polyamines. The primary readout
is reachability (fraction of poses with an HMDA amine within 3.5 Å of the acyl carbon);
the Bürgi–Dunitz angle gate [90,120]° is reported as a stricter, more fragile secondary
metric. GlnA3 (Q9KZC7) is the only enzyme strong on both half-reactions.

Three caveats condition this readout. (i) The amine is held by a **forced pocket
restraint** (`force: true`, the amine within 10 Å of ~20 residues derived from that
enzyme's own step-1 adipate pose); without it a small flexible diamine strands on the
surface, but it also means accommodation is measured *given* HMDA is localized to the
acid pocket, and the per-enzyme restraint shells differ. (ii) The output has no
hydrogens, so the closest amine nitrogen is used — the neutral vs protonated amine
cannot be told apart geometrically. (iii) Step-2 intermediate-state geometry was
non-predictive of measured activity on the experimental panel, so this is a feasibility
screen, not a ranking; step-1 adenylation geometry carried the activity signal.

**Boltz-2 gives geometry, not affinity, on a non-native substrate.** A favourable pose
is not evidence that adipate/HMDA binds or turns over. Composition and geometric
descriptors do not establish native assignment, substrate promiscuity, completed
amidation, biological assembly or yield; those require experimental testing.
