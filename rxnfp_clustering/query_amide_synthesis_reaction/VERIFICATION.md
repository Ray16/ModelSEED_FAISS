# How this project checks itself

Nine re-runnable tools, each written because a claim had been asserted rather
than measured — and in four cases because the assertion turned out to be
wrong. Results below are from 2026-10-13 and every one reproduces by running
the command shown. One further check is not a tool but a guard inside a stage,
described with the others below: it refuses to run on a superseded input.

| check | command | current result |
|---|---|---|
| nothing computed then dropped | `tools/audit_columns.py` | **no column empty**, 52/52 populated |
| every analysis covers every candidate | `tools/audit_coverage.py` | **complete**, after it found 4 stages a route behind |
| what the evidence supports, uncapped | `tools/qualified_set.py` | **79** at ≥ 2 independent lines; 40 invisible to the bench shortlist |
| measured substrate walks | `tools/substrate_walk.py` | 3 enzymes on ≥ 2 rungs; none reaches C6 |
| external recall | `tools/recall_audit.py names` | **100%** on every named truth set; 0/4 negative controls |
| EC-class recall | `tools/recall_audit.py ec` | no actionable gap; every zero verified as correct exclusion |
| retrieval completeness | `tools/probe_unresolved.py` | **0 actionable losses**; 308 resolved, 19 borrowed, 7 via KEGG orthology, 146 with no enzyme |
| annotation ceiling | `tools/verify_annotation_ceiling.py` | **0 of 120** blank fields exist upstream |
| ranking validity | `tools/validate_against_activity.py` | **unvalidatable at n=3**, CI 0.23–0.85 |
| family chemistry | `run_all.py --only famval` | 54 lead / 16 watch / 33 reject |

## What each one caught

None of these was written speculatively; each exists because something was
wrong.

- **`probe_unresolved`** found two reactions (`rxn14123`, `rxn19001`) refused
  by a stage re-litigating an admission already made upstream — the fourth
  score gate. Later it was itself found to be overstating: it counted 19
  reactions as unresolved that carry a borrowed enzyme.
- **`verify_annotation_ceiling`** was sampling with a regex that excludes O, P
  and Q accessions — disproportionately the *reviewed* entries most likely to
  carry the annotation it asks about. The biased sample happened to give the
  right answer; it would not have been evidence for it.
- **`recall_audit`** found the pipeline holding **0 of 26** proteins UniProt
  literally names "Amide synthetase", while 192 internal controls passed.
  Internal tests encode assumptions the project already holds; an external
  list does not.
- **`audit_columns`** exists because two signals were computed and silently
  discarded at the de-duplicating merge: `acylates_free_diamine`, now the most
  discriminating axis in the convergence count, and `borrowed_from_rxn`, at
  0 of 54,199.
- **`substrate_walk`** asks what no scoring axis here can: which enzymes have
  published kinetics on more than one rung of a chain-length series. Every
  chemistry score asks how much a native substrate RESEMBLES adipate + HMDA,
  which is blind to an enzyme whose native substrate is distant but whose
  activity has been walked toward C6. Three enzymes qualify, two of them EC
  6.3.2.x amide ligases. It also demonstrates why the rung count alone must
  not be read as the result: PanC has the longest walk (C2,C3,C4,C5) and the
  values refute the obvious reading of it - glycine at C2 is 0.6 M^-1 s^-1
  against C4 at 0.81 and C5 at 0.42, so everything that is not beta-alanine
  sits on the same non-specific floor whether SHORTER or longer. The flat
  C4-C5 segment is background, not tolerance extending along the chain, and
  PanC is evidence of specificity rather than flexibility. I had written the
  opposite before reading the C2 row.
- **`qualified_set`** exists because the project only ever shipped an answer
  to "which 250 should the bench test", and the question being asked was
  "how many candidates does the evidence support". The shortlist caps per
  family, clade and genus - right for a bench round, wrong for counting -
  and 40 proteins carrying two independent lines appeared in no deliverable
  because a sibling from the same family already held a slot. Writing it
  forced the correction below: the first measurement said 1,817, which is
  what the bar gives before correlated axes are collapsed.
- **The convergence count itself was inflated by a correlated axis, and the
  correction cut it from 28 to 17.** `bifunctional_ligand_in_the_pocket` and
  `own_reaction_chemistry` were counted as independent lines. For 1,753 of the
  1,754 candidates carrying exactly that pair, the curated binding-site ligand
  is a participant in the protein's *own* curated reaction — the same ChEBI id
  appears verbatim in its catalytic-activity line. UniProt curates "binds
  UDP-MurNAc-L-Ala-D-Glu" and "catalyses a reaction consuming
  UDP-MurNAc-L-Ala-D-Glu" from one piece of knowledge, and the count treated
  it as two. The ≥2-line set falls from **1,823 to 84** and the headline from
  **28 to 17**. This module's own docstring already demanded that a new axis
  be checked against the others; the check made when this axis was added was
  crystallographic-vs-curated, which is a different correlation from the one
  that mattered. Two cases are deliberately preserved: a *crystallographic*
  ligand is an observation rather than a restatement (which is why MurE
  P9WJL3 is untouched — UAG really is in the density), and a curated ligand
  the protein's reaction does not involve is a genuinely separate claim. All
  four group-A assay recommendations survive the correction.
- **`audit_coverage`** found that *every* per-protein analysis had stopped one
  route short. The `actives` route added 822 homologs of the measured
  producers; `pocket_capacity`, `ligand_evidence`, `curated_properties` and
  `ec_substrate_ceiling` had all been written before it existed, so each
  covered 0 of the 822 while reporting tens of thousands of healthy rows. The
  consequence was not cosmetic: those are the structural and EC axes, so the
  measured-data hypothesis sat at one evidence line per protein partly **by
  construction**, because the analyses that could have given it a second line
  had never been run on it. `ec_substrate` needed a code fix as well as a
  re-run — it assembles annotations from a hardcoded list of route files, to
  avoid a circular dependency with `master_table`, and that list had fallen
  one route behind. The two lists are now checked against each other.
  - Writing the tool also demonstrated its own thesis. The first draft used a
    coverage share alone and flagged `member_similarity` at 84% as a missed
    route; it is a within-family tie-break with nothing to say about a
    protein admitted by its own reaction. A share cannot distinguish "missed
    a route" from "scoped by design", so each scoped stage now declares its
    scope and is checked for completeness *inside* it — the stronger claim:
    `member_similarity` must cover all 46,254 family-route candidates, and
    does.
- **The stale-input guard in `shortlist`** was added after the stage read an
  enriched table one generation behind the master it was built from. `enrich`
  needs UniProt, so `--offline` drops it; re-running `master` alone leaves the
  enriched copy superseded, and the shortlist silently rebuilt itself from the
  old flags. The symptom was the most dangerous kind available: a fix landed
  in `master`, the shortlist did **not** move, and the unchanged number looked
  like a measured result. A new defect class — not a wrong computation but a
  correct one on a superseded input — so the guard compares mtimes and fails
  the stage rather than choosing.
- **`validate_against_activity`** is the one that has never found a bug,
  because its answer has never changed: with three measured positives the
  bootstrap CI spans chance and sequence length alone outscores every ranking
  this project can build. It is kept to stop anyone, including the author,
  claiming otherwise.

## The claim that cannot be checked here

Nothing above validates that any candidate works. Convergence means several
methods that fail differently agree, which justifies a bench slot and nothing
more. The assay round is the only thing that converts this into a result.
