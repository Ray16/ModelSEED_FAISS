# Reproducing the current ten-reaction panel

## Run an isolated offline replay

From `query_amide_synthesis_reaction/`, use the existing scientific environment:

```bash
/homes/rzhu/miniforge3/envs/palm/bin/python -m amide_synthesis reproduce
```

The default [lock](../config/top10_reproduction.lock.json) selects the current ten-reaction case. Output goes to a new timestamped directory in `results/reproducibility/`; choose an empty directory with `--output PATH`. `--check-only` verifies inputs and package/tool versions without computing.

[Verified current replay](../results/reproducibility/current_top10_20261020/report.html).

## What is recomputed?

| Step | Source | Decision |
|---|---|---|
| Target queries | `reaction_search/encode_and_search.py` | Balanced adipate/HMDA framings and retained cofactors. |
| Full mechanistic screen | `reaction_search/amide_enzymes.py` | Reconstruct free-amine acylation and apply substrate/balance rules. |
| Top-100 retrieval | `rank_modelseed_nylon.py`, `reaction_search/retrieval_policy.py` | DRFP–Tanimoto primary; hybrid cosine and reaction ID for ties. |
| Chemical selection | `nylon_chemical_review.py`, `reaction_search/chemical_selection.py` | Soft chemical preferences from versioned configuration; no original-ID allowlist. |
| Mapping and alignments | `evaluation/top5_mapping_audit.py` | Generic panel engine; replay exact reaction evidence and keep related-reaction candidates separate. |
| Redundancy | `candidate_selection/sequence_redundancy.py` | Exact duplicates and CD-HIT 99%/95% groups with both-sequence coverage checks. |
| Exports and checks | `reproduce.py` | Tables/FASTA, figures, commands, environment, hashes and scientific regression comparisons. |

Paths without a root-script name are relative to `amide_synthesis/`. Reaction display explanations and drawing labels never affect selection. The previous five are checked only after selection. Weight sensitivity is a consistency check, not experimental validation.

## Inputs, environment and failure behavior

The lock pins database tables, the normalized corpus/embeddings, reference outputs, mapping profiles, chemical policy, model weights and tool binaries. Scientific Python/package versions are recorded; full inventories are in [reproduction requirements](../config/reproduction-requirements.lock.txt) and [RXNFP worker requirements](../config/rxnfp-requirements.lock.txt). The installed CD-HIT 4.8.1 and separate RXNFP worker are required.

The corpus and fingerprints are code-built prerequisite intermediates. To rebuild them from the pinned ModelSEED checkout in a separate work directory, use `AMIDE_SYNTHESIS_WORKDIR` with `python -m amide_synthesis.reaction_search.rebuild_corpus`. The default replay verifies those intermediates instead of rebuilding all fingerprints.

The complete [ten-reaction raw snapshot](../results/top10_mapping/20261020_chemical_panel/manifest.json) retains UniProt, Rhea and ChEBI responses. Missing responses, changed URLs/bytes, pagination mismatches or mixed releases stop execution; **there is no network fallback**. The [mapping profiles](../config/top10_mapping_profiles.json) explicitly distinguish target Rhea IDs from related-reference IDs. Symbolic structures and missing stereochemistry retain qualifications.

Regression checks compare the eligible reaction set, 480 reaction rankings, top 100, chemically selected ten, candidate/rejected protein records, full-equation audits and cluster memberships. Changed protein identifiers or sequences fail even if counts stay constant. Display-name improvements do not alter scientific counts. Timestamps, paths and figure/PDF metadata need not be byte-identical.

The historical five-reaction replay remains available explicitly with `--lock config/nylon_reproduction.lock.json`; it is not the current production selection.

## Models and plots

The offline replay ends with filtered protein pools and submits no predictions. The separate [reference-selection record](../results/top10_cofold/reference_selection.json) records all native-annotation references, the inferred BacD/YwfE choice, exact sequence/MSA checks, frozen source copies and model-reuse decisions.

Maintained model preparation and submission commands are in `structure_modeling/reference_panel.py`: `stage`, `freeze-sources`, `prepare` and explicit `launch`. Batched exact-sequence MSA retrieval precedes preparation. `launch` runs on preflighted lambda13 through the shared GPU gate with explicit CPU/thread allocation, bounded retries and output checks. Existing predictions remain tied to their original campaigns.

Recheck actual structures and regenerate plots and the response:

```bash
python -m amide_synthesis model inspect --run-dir results/top10_cofold/campaign
python -m amide_synthesis evaluate current-panel
```

The [final verification](../results/top10_cofold/final_verification.json) checks every requested model. The [updated plot](../results/top10_cofold/activation_geometry.png) and [geometry table](../results/top10_cofold/activation_geometry.tsv) distinguish the unverified exact-reaction candidate. Models describe acid activation, not HMDA attack, completed amidation or experimentally validated activity.
