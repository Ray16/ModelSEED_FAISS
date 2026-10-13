# How this project checks itself

Six re-runnable tools, each written because a claim had been asserted rather
than measured — and in three cases because the assertion turned out to be
wrong. Results below are from 2026-10-13 and every one reproduces by running
the command shown.

| check | command | current result |
|---|---|---|
| nothing computed then dropped | `tools/audit_columns.py` | **no column empty**, 52/52 populated |
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
- **`validate_against_activity`** is the one that has never found a bug,
  because its answer has never changed: with three measured positives the
  bootstrap CI spans chance and sequence length alone outscores every ranking
  this project can build. It is kept to stop anyone, including the author,
  claiming otherwise.

## The claim that cannot be checked here

Nothing above validates that any candidate works. Convergence means several
methods that fail differently agree, which justifies a bench slot and nothing
more. The assay round is the only thing that converts this into a result.
