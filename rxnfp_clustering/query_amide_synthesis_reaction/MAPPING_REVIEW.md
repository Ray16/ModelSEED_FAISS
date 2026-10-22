# Reaction-to-enzyme mapping: robustness review

## 20 October extension: current top-five reaction audit

The [live audit and count visualization](results/top5_mapping_audit/review.html)
retrieves all pages from UniProt release 2026_03, checks returned reaction JSON
fields independently, and compares each full ModelSEED equation with Rhea
participants and current ChEBI structures. Cofactors and stoichiometric copies
are retained; protonation normalization is explicit. `rxn42510` and `rxn47618`
have compatible but underspecified product stereochemistry, not an exact
stereochemical certification. All eight supported protein records carry
ECO:0000269 on their matching native-reaction annotation.

The native reaction supported for SbnE (`Q2G1M9`) was missed under `rxn47618`
because the local entry lacks Rhea aliases. Rhea independently links its
MetaCyc RXN-19820 to RHEA:59124. The local `rxn54616` counterpart has the same
full saved reaction structure and supplies those aliases. The approved bridge
is in [reaction_mapping_bridges.json](data/reaction_mapping_bridges.json).
The production offline join and live resolver share fail-closed checks of
equation identity, source sequence/evidence, registry/audit hashes and the
corpus snapshot. Bridge provenance never overwrites an existing direct link.
The underlying 233,299-protein direct mapping remains its recorded source
snapshot; the bridge is an explicitly qualified additional resolution route.

Exact sequences and 99%/95% identity representative groups are reported
separately. Group membership requires 90% alignment coverage of each sequence.
Gene names alone are rejected: `asbA` and `sfaD` also name unrelated proteins.
The 1,030 expanded candidates include inferred EC/name/homology assignments;
they must not be relabeled as eight-to-1,030 experimentally established mappings.
Ten expanded AsbA records are fragments and remain explicitly flagged.

The implementation is separated into snapshot retrieval, reaction chemistry,
approved equation bridges, sequence redundancy and report rendering modules.
Earlier reports, slide code and obsolete logs have been moved to
[the archive](archive/mapping_refactor_20261020/README.md).

Reviewed and corrected 19 October 2026. The reviewed-UniProt source snapshot was
retrieved on 17 October; this correction reuses that snapshot. It is not a new
retrieval of every protein record.

## Assessment

The rebuilt mapping is suitable for finding proteins with traceable native-reaction
annotations. It is **not a universally validated reaction-to-enzyme assignment**.
Identifier and sequence consistency now pass the full reconstruction audit, but
ModelSEED alias chemistry, annotation inference, protein-complex requirements and
activity on adipate/HMDA remain separate questions.

The original merged mapping is unsuitable as an unquestioned source of exact
reaction assignments. Keep it for discovery and historical comparisons, with its
links explicitly treated as unverified until independently supported.

## What was checked

The repeatable [audit script](amide_synthesis/evaluation/audit_mapping_robustness.py) uses its own
field-scoped parser rather than importing the production parser. It checks every
mapping pair against the saved source annotations, sequence hashes, nonobsolete
ModelSEED aliases and evidence sidecar. It independently reconstructs the expected
experimental subset. This tests reconstruction consistency with the declared
inputs; agreement with the inputs does not prove the inputs are biochemically right.

| Check | Result after correction |
|---|---:|
| Reviewed UniProt source records inspected | 236,245 |
| Mapped proteins | 233,299 |
| Reaction–protein pairs reconstructed | 316,919 |
| ModelSEED reactions represented | 11,071 |
| Evidence rows checked, including multiple supporting Rhea IDs | 372,286 |
| Missing/extra reconstructable pairs | 0 / 0 |
| Evidence-scope, sequence, identifier or alias-join mismatches | 0 |
| Experimental subset missing/extra pairs versus scoped source | 0 / 0 |
| Separate saved JSON cross-check | 2,683 links across 1,436 proteins; no exact-ID or sequence mismatch |

The JSON subset was selected for the earlier focused/random audit of the original
mapping. It is **not a random accuracy sample of the rebuilt mapping**. Neither
that subset nor the complete reconstruction supplies a global catalytic accuracy
estimate. See the [before audit](results/mapping_robustness_20261019/before_audit.json)
and [after audit](results/mapping_robustness_20261019/after_audit.json).

## Defects found and corrected

**Original file: participant IDs were confused with reaction IDs.** In the
saved uniform sample of 200 original mapping pairs, 127 had exact reaction-ID
support, two had EC-only support, 34 showed a RHEA-COMP number collision and
37 remained unresolved. These are annotation-concordance categories, not a
biochemical false-positive estimate. The focused audit found 52 of 53
`rxn02221` links and all 23 `rxn01979` links with component-number collisions;
it also found four regulatory subunits linked to `rxn00646`.
[Rhea distinguishes participant IDs from reaction IDs](https://www.rhea-db.org/help/reaction-participant).
The replacement accepts only exact reaction cross-references. The final merged
legacy file alone does not establish which upstream operation caused the collisions.

**Rebuilt file: reaction evidence was borrowed by direction annotations.** UniProt
stores the undirected reaction and each physiological direction with separate
cross-references and evidence. The parser previously collected all their Rhea IDs
and assigned the first evidence field to all of them. The independent audit found
**9,337 incorrectly scoped evidence rows, including 8,175 with unsupported tiers**.
This count allows a Rhea ID to have multiple valid source annotations; it supersedes
the larger preliminary count that retained only one annotation per ID.
[UniProt documents the separate fields](https://www.uniprot.org/release-notes/2018/12/05/release).

The parser now retains evidence within each reaction or direction field. The
316,919 mapping pairs and their sequences are unchanged. Their best evidence tiers
were recomputed: the strongest tier changes for **297 unique pairs**, since
multiple evidence rows can support the same pair. The experimental subset
increases from 34,597 to **34,637
pairs**, covering **19,000 proteins**. The 40 newly included pairs have experimental
evidence on their own source field. Multiple annotations for the same pair are
combined by the strongest supported tier.

**Evidence integrity was less protected than the mapping table.** The policy now
pins SHA-256 hashes of the evidence sidecar and experimental subset as well as the
main mapping. Audited ingestion rejects a retained pair with missing or invalid
evidence instead of silently assigning a legacy label. Sequence/evidence mismatches
also raise an error. Explicit use of the legacy mapping remains labeled unverified.

## What the evidence tiers mean

The counts below are unique reaction–protein pairs, assigned their strongest
supported tier. They are not counts of independent experiments.

| Best tier | Pairs | Meaning |
|---|---:|---|
| `experimental_direct` | 34,637 | ECO:0000269 on the matching reaction/direction field |
| `curated_direct` | 31,486 | Direct reaction cross-reference with another or absent evidence code; can include curator inference |
| `homology_supported_direct` | 44,176 | Matching annotation supported by ECO:0000250 |
| `computational_direct` | 206,620 | Matching annotation supported by ECO:0000255 or ECO:0000256 |

Here, **direct describes the identifier join**. It does not mean that every
protein was experimentally tested. About 10.9% of pairs have the explicit
experimental tier. Even those annotations concern the recorded native reaction,
not necessarily adipate/HMDA, an isolated monomer or the construct used in an assay.
Read the cited primary experiment before treating a pair as a positive control.

## Remaining limitations

**An exact alias join can inherit a chemically defective equation.** Four Rhea
IDs (`11612`–`11615`, the direction forms of one reaction) each point to both
`rxn00184` and `rxn30679`. The latter omits water and protons and is flagged
`MI:O:1|CI:-2` by ModelSEED; the former has status `OK`.
[Rhea's reaction includes water](https://www.rhea-db.org/rhea/11612).
These aliases produce 152 mapping pairs across the two reactions. Neither reaction
passes the current amide screen, so this example does not enter the screened
candidate route. It demonstrates why the full mapping cannot be called chemically
certified. The upstream equations and aliases are preserved, not silently repaired.
The [fan-out table](results/mapping_robustness_20261019/after_audit_alias_fanout.tsv)
contains the actual definitions.

Across the mapped reactions, 2,662 have a raw ModelSEED status other than exactly
`OK`, accounting for 109,892 pairs. Most of those reaction flags are
`CPDFORMERROR` (2,226 reactions); others include charge or mass issues and one
`OK|CK`. **These flags are not 2,662 proven wrong enzyme assignments.** Symbolic
participants and incomplete formulas can also prevent a balance check. Conversely,
`OK` does not prove that the Rhea alias has the correct substrate specificity or
stereochemistry. A global structure/stoichiometry equivalence audit was not performed.

**Recall is deliberately incomplete.** This route uses reviewed entries and local
Rhea aliases. Unreviewed proteins, reactions without such aliases, promiscuous
activities absent from the annotations and sequence-space neighbors can be missed.
Other retrieval routes may find them, but EC, names, domains and homologs are
hypotheses with their own evidence labels. They must not inherit an experimental
native-reaction claim just because they resemble an annotated enzyme.

**A mapped protein need not be sufficient alone.** Catalytic subunits can require
other subunits, maturation, metal loading or partner proteins. Fragment and
construct checks in the frozen 50-protein panel do not amount to an independent
construct audit of all 233,299 proteins. A cofold assembly check does not establish
catalytic competence or successful amine coupling.

## Current artifacts and reproduction

[Mapping policy](mapping_policy.json) selects the
[reviewed direct mapping](results/colleague_review_20261017/mapping/uniprot-msd-direct-reviewed.tsv),
[experimental subset](results/colleague_review_20261017/mapping/uniprot-msd-experimental-reviewed.tsv)
and [per-link evidence](results/colleague_review_20261017/mapping/direct_mapping_evidence.tsv.gz).
The [source/rebuild summary](results/colleague_review_20261017/mapping/direct_mapping_rebuild_summary.json)
records the source snapshot hash and database hashes. The original
`uniprot-msd-links.tsv` is unchanged.

```bash
cd rxnfp_clustering/query_amide_synthesis_reaction
python -m amide_synthesis evaluate audit-mapping \
  --output-prefix results/mapping_robustness_20261019/recheck
```

This audit reads saved evidence and does not submit jobs or retrieve new protein
records. Rebuilding the mapping or changing the evidence policy requires refreshing
its dependent tables and checking [artifact freshness](results/artifact_freshness.json)
before presenting results as current. The frozen 17- and 50-protein structure
campaigns retain their original inputs; their evidence context is updated separately.

The [pre-correction archive](results/mapping_robustness_20261019/before_correction/provenance.json)
records hashes of the previous mapping artifacts, dependent outputs and documents.
The previous [mapping review](results/mapping_robustness_20261019/before_correction/MAPPING_REVIEW.md)
and [README](results/mapping_robustness_20261019/before_correction/README.md) are
historical records, not current instructions or counts.
