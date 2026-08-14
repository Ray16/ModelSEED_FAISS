# Mass/Charge Balance Correction — Delivery Report

**521 reactions corrected**, every one independently re-verified to mass + charge = 0. Nothing in `ModelSEEDDatabase/` was modified — these are proposals in `corrected_reactions.tsv`.

## Confidence tiers

- **auto-apply: 334** — deterministic (proton/water), EC-corroborated co-substrate/couple, KEGG-matched 2-OG dioxygenase. Safe to apply directly.
- **review: 187** — agent name-based couples + low/review co-substrate. Balance-valid but chemically judged; recommend expert sign-off.

## Breakdown by tier × method

| tier | method | n |
|---|---|---|
| auto-apply | cosubstrate:PPi | 5 |
| auto-apply | cosubstrate:Pi | 1 |
| auto-apply | cosubstrate:Sulfate | 2 |
| auto-apply | couple:acetyl | 3 |
| auto-apply | couple:methyl | 11 |
| auto-apply | proton_water | 309 |
| auto-apply | twoog_demethylase:2OG-dioxygenase-demethylation | 3 |
| review | cosubstrate:CO2 | 19 |
| review | cosubstrate:NH4 | 12 |
| review | cosubstrate:PPi | 18 |
| review | cosubstrate:Pi | 40 |
| review | cosubstrate:Sulfate | 1 |
| review | couple_agent:acetyl | 14 |
| review | couple_agent:glucosyl | 29 |
| review | couple_agent:mannosyl | 3 |
| review | couple_agent:methyl | 51 |

## External validation (KEGG)

- 88 corrections had a KEGG cross-reference and were cofactor-checkable; **85 (97%) consistent** (validated or KEGG-incomplete). Contradictions were retracted upstream; none remain in this set.

## Verification

- `verify_corrections.py` re-parses every corrected stoichiometry from disk and recomputes balance from scratch: **all balanced**.
- Guards applied: degeneracy rejection (no empty reactions), EC prefilter (no transferase couple on an oxidoreductase/hydrolase), stereo tier for glycosyl.

## Scope not covered (see TODO.md)

- Unscoreable R-group/generic-formula reactions (~38.8% of the DB) — refused by design.
- Redox charge-only (~1,500, implicit electrons), oxygen-only hydroxylations (~297), heavy-skeleton errors (~1,500) — flagged for supervised rounds / curation.
