# Amide synthetase candidates: current ten-reaction panel

Updated 21 October 2026, 06:05 UTC. Target: adipate + hexamethylenediamine (HMDA) → nylon-6,6 amide diad.

## 1. How are reactions retrieved and chemically selected?

**DRFP–Tanimoto retrieves the top 100** from 480 balanced amide-forming reactions in the 33,246-reaction searchable corpus. Cofactors remain included; ECFP4-difference + DRFP cosine breaks retrieval ties.

**Chemical selection then chooses ten**, rather than taking the first ten DRFP hits. The code checks free-acid/primary-amine amide formation and softly compares functional architecture, retained acid/amine ends, carbon spacers, backbone and substrate size. Monoacids, monoamines, ω-amino acids, AB self-condensation, rings and aromatic substrates remain eligible.

The declared weights are 35% architecture, 10% retained ends, 20% spacers, 15% backbone and 20% size. They are transparent chemical preferences, **not activity probabilities**. Every component and explanation is exported; original reaction IDs never control selection.

**Validation against the previous five:** 5/5 remain in the ten without being forced in. Across 20 tested weight perturbations, retention ranged from 5/5 to 5/5 (each component varied from 0.5 to 1.5 times its default, then renormalized). This checks consistency and sensitivity, not experimental accuracy.

[Chemical selection and structures](../results/modelseed_nylon_top10/report.html) · [Component visualization](../results/modelseed_nylon_top10/chemical_selection_components.png) · [Validation details](../results/modelseed_nylon_top10/original_five_validation.json).

## 2. How are reactions mapped to enzymes, and how many remain?

Map ModelSEED → Rhea → matching UniProt reaction/direction annotations, preserving their own evidence. Equivalent-equation bridges and complete Rhea/ChEBI structures are checked separately. EC/name candidates also require ≥30% sequence identity and ≥80% coverage of both proteins against an annotated reference.

Remove identical sequences, then cluster within each reaction at 99%/95% identity with ≥90% coverage of both proteins. Preserve every accession and membership; genome labels alone never merge proteins.

| ModelSEED | Selected reaction | Annotated IDs | Expanded IDs | Exact sequences | 99% reps | 95% reps |
|---|---|---:|---:|---:|---:|---:|
| rxn05123 | Glutamate–putrescine ligation | 4 | 978 | 961 | 903 | 763 |
| rxn46045 | Fumarate–diaminopropionate ligation (DdaG) | 1 | 6 | 6 | 6 | 6 |
| rxn42510 | Citrate–D-ornithine ligation | 1 | 2 | 2 | 1 | 1 |
| rxn47618 | Citrate–diaminopropionate ligation | 1 | 11 | 11 | 11 | 11 |
| rxn21038 | Citrate–spermidine ligation | 1 | 33 | 32 | 30 | 26 |
| rxn22343 | UDP-peptide–D-lysine ligation (MurE) | 1 | 5 | 5 | 5 | 5 |
| rxn42395 | Staphyloferrin B formation (SbnC) | 1 | 2 | 2 | 2 | 2 |
| rxn09897 | Glycylglycine formation | 0 | 16 | 16 | 16 | 10 |
| rxn45007 | Citryl-intermediate–diaminopropionate ligation (SbnF) | 1 | 5 | 5 | 5 | 5 |
| rxn21036 | Petrobactin-intermediate ligation (AsbB) | 1 | 6 | 5 | 5 | 5 |
| **Unique total** | | **12** | **1,064** | **1,045** | **984** | **834** |

**Qualifications:** glycylglycine formation has no established exact reaction assignment; its 16 candidates come from related L-amino-acid ligases. BacD/YwfE normally joins alanine and anticapsin; its Gly+Gly assignment remains unverified. MurE’s Rhea entry contains a symbolic participant, limiting full structural verification. Missing stereochemistry remains flagged for citrate-derived entries. Annotation supports native activity, not adipate/HMDA activity.

10 fragment records remain explicitly flagged in the expanded pools; they are not selected reference models.

[Evidence, named enzymes and mapping visualization](../results/top10_mapping/report.html) · [All candidate records](../results/top10_mapping/20261020_chemical_panel/reaction_candidates.tsv).

[95% representative FASTA](../results/top10_mapping/representatives95.fasta) · [Identifier/evidence table](../results/top10_mapping/representatives95.tsv).

## 3. Which enzymes are modeled?

The reference panel contains **13 proteins**: 12 with matching reaction annotations and one related-reaction BacD/YwfE candidate. It includes PuuA; GlnA2 from Haloferax mediterranei; GlnA2 and GlnA3 from Streptomyces coelicolor; DdaG; SfnaD/SfaD; SbnE; AsbA; MurE; SbnC; SbnF; AsbB; and BacD/YwfE. The two GlnA2 proteins are distinct sequences.

The two GlnA2 proteins have 32.5% global sequence identity. Adipate–ATP distances are below 6 Å in 49/50 Haloferax poses and 1/50 Streptomyces poses. Neither ensemble has a pose below 4 Å. This is a model-placement difference, not evidence of different measured activity. The ensembles are monomeric and differ in reuse/implementation provenance. [Pose comparison](../results/top10_cofold/glnA2_comparison/pose_comparison.png).

**Current checked coverage: 13/13 proteins, 650/650 poses.** Ten existing ensembles were sequence/input-checked and reused. The three new ensembles are Haloferax GlnA2, Streptomyces GlnA3 and BacD/YwfE. Each uses exact Boltz-2, an exact-sequence MSA and 50 diffusion samples; execution uses the shared GPU reservation gate.

Complexes contain **protein + ATP + Mg + adipate**, without HMDA or activated intermediate. These are comparative acid-activation models, not proof of amidation, polymerization or correct biological assembly. The BacD model does not establish glycylglycine activity. This reference panel is separate from a larger sequence-diversity panel.

[Model inputs/provenance](../results/cofold_step1/manifest.json) · [Current coverage](../results/top10_mapping/current_panel_status.json) · [Model output folders](../results/cofold_step1/outputs).

[Updated cofold geometry plot](../results/top10_cofold/activation_geometry.png) · [Geometry table](../results/top10_cofold/activation_geometry.tsv). Rows follow the reaction panel, not predicted activity. Angle and distance remain structural descriptors.

Harmonic distance summarizes the 50 pose distances as 50 / Σ(1/d), emphasizing closer poses. The unverified-candidate label concerns enzyme mapping, independently of this distance metric.

## 4. What were Josh’s five literature enzymes?

| Enzyme | Original UniProt | Native transformation | ModelSEED |
|---|---|---|---|
| DdaG | E2JA32 | Fumarate + L-diaminopropionate → fumaroyl-DAP. | rxn46045 |
| SfaB | A0A2H4T912 | 3-Isocyanobutanoate → acyl-AMP. | rxn60314 |
| CphA1 | P73833 | Cyanophycin backbone Asp addition and Arg side-chain ligation. | rxn48822, rxn49577 |
| McbA | R4R1U5 | β-Carboline carboxylate + β-phenethylamine → marinacarboline amide. | No exact native entry located. |
| PylC | Q46E79 | Methyl-D-ornithine + L-lysine → N⁶-linked isopeptide. | rxn23366 |

All five UniProt sequences match the corresponding GenBank records in the saved audit. CphA strains/constructs sharing the SuCphA1 name are not interchangeable. DdaG and SfaB have reported adipate/HMDA activity in the supplied literature.

[Native structures and nylon comparison](../results/native_nylon_similarity/similarity_heatmap.png) · [Sequence audit](../results/native_reaction_audit/20261020T204408703497Z/manifest.json).

## 5. Does another structural statistic help predict yield?

Adding log harmonic distance to angle improved producer-classification AUROC from **0.659 to 0.703** in the 49-enzyme panel. The improvement interval spans zero; continuous-titre prediction did not improve, and unseen-enzyme transfer failed in the small literature dataset. Distance remains exploratory. Poses are not independent experiments; paralog selectivity needs experimental testing.

[Yield visualization](../results/colleague_review_20261017/figures/yield_validation.png) · [Validation audit](../results/colleague_review_20261017/validation/yield_analysis_audit.json).

## 6. Reproducibility and notes for the mapping colleague

Queries, chemical selection, mapping, sequence filtering, reference selection, model reuse and structural checks are code. Configuration and raw responses retain provenance. See [reproduction instructions](REPRODUCIBILITY.md) and the [chemical policy](../config/chemical_selection_policy.json).

For the colleague: TSV formatting is consistent, but some links contain RHEA-COMP participant/reaction-number collisions. Preserve reaction-specific evidence and catalytic roles; EC/GO/name matches alone are weaker. Unresolved links are not automatically errors. Missing database aliases and our separate rebuilt-parser bug must not be conflated with defects in the colleague’s code.

[Detailed mapping review](../MAPPING_REVIEW.md).
