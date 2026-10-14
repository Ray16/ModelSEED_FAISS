# How this project checks itself

Seven re-runnable tools, each written because a claim had been asserted rather
than measured — and in four cases because the assertion turned out to be
wrong. Results below are from 2026-10-13 and every one reproduces by running
the command shown. One further check is not a tool but a guard inside a stage,
described with the others below: it refuses to run on a superseded input.

| check | command | current result |
|---|---|---|
| nothing computed then dropped | `tools/audit_columns.py` | **no column empty**, 52/52 populated |
| every analysis covers every candidate | `tools/audit_coverage.py` | **complete**, after it found 4 stages a route behind |
| external recall | `tools/recall_audit.py names` | **100%** on every named truth set; 0/4 negative controls |
| EC-class recall | `tools/recall_audit.py ec` | no actionable gap; every zero verified as correct exclusion |
| retrieval completeness | `tools/probe_unresolved.py` | **0 actionable losses**; 308 resolved, 19 borrowed, 153 with no enzyme |
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
