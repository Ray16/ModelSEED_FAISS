# Expanded candidate screen — state of record

Authoritative status of the growing enzyme-candidate pool for adipate + HMDA
(nylon-6,6 diad). Updated as stages complete. The 10-reaction panel and its 13
reference enzymes are the baseline; this effort expands the pool within the
**ligase** families (the chemistry most similar to the nylon diad; the
acyltransferase 2.3.x route is deferred as less target-similar).

## The pool: 124 distinct candidates

Clustered at 50% identity (sequence-distinct enzymes), per-reaction mechanism known.

| Increment | status key | count | what | fold |
|---|---|---:|---|---|
| References | `reuse` | 13 | panel enzymes incl. GlnA3 (Q9KZC7), the best hit | existing step-1 folds reused — DO NOT recofold |
| Expressible distinct | `new` | 86 | E. coli-friendly ligase homologs (GS 58 + NIS + MurE + BacD), 50%-id reps | folding now |
| Hard-host distinct | `new_hardhost` | 25 | actinomycete/archaeal homologs (incl. GlnA3's neighbors); kept because codon-opt handles GC — expressibility is an annotation, not a screen filter | MSAs regenerating → fold next |

Pool definition: `pool.tsv` (status column distinguishes the three increments).

## Pipeline stages

1. Reaction selection → 10 reactions (baseline). DONE.
2. Mapping → candidates; expansion shortlist `../top10_mapping/expansion_shortlist.tsv`. DONE.
3. MSAs:
   - 86 expressible → `msa/` (86 CSVs, index.tsv). DONE, verified query==sequence (86/86).
   - 25 hard-host → `msa_hard/` (fresh dir). IN PROGRESS.
4. Step-1 cofold (protein + ATP + Mg + adipate; activation geometry):
   - 86 expressible → `campaign/` on lambda13, gated. RUNNING (see `campaign/state/*.done.json`).
   - 25 hard-host → `campaign_hard/`. PENDING MSAs.
   - 13 references → reused folds elsewhere (`pool.tsv:existing_fold`).
5. Step-1 geometry across all 124. PENDING step-1.
6. Step-2 (acyl intermediate + neutral-amine HMDA + Mg [+ADP]; accommodation) for
   **ALL candidates, not just step-1 survivors** (no funnel). PENDING step-1.
   Rationale: step-1 and step-2 are independent — BacD activates best (step-1 0.74)
   but cannot fit HMDA (step-2 reachable 0.14), while PuuA is a weak activator
   (step-1 0.10) yet accommodates the amine (step-2 0.44). Funnelling step-2 on
   step-1 hits would miss real accommodators. The 13 references already have step-2;
   step-2 must be built+folded for the 111 new (pocket derived from each one's own
   step-1 adipate pose), so it waits on step-1 completion.

## Known pitfalls (do not repeat)
- **batched MSA output dirs are per-run.** Writing a 2nd batch into a populated MSA dir
  cross-mapped index.tsv/CSVs (hash named for one seq, content of another). Always a
  fresh `--out` dir per run, and verify MSA query==sequence before folding. The 86 set
  was verified clean; the 25 are being regenerated in `msa_hard/`.

## Step-1 ranking metric + transferability (important)

- **Metric:** candidate-workflow rule from the manuscript — distance-filter to poses
  <4 Å, report the mean in-line attack angle among them, mark <5 retained poses
  *inconclusive*. Verified on the SfaB assay panel the proximity-weighted attack angle
  A (and this engaged-angle view) beats near-attack fraction for surfacing hits
  (3-active AUROC 0.94 and all 3 top yielders in the top-10, vs 0.70 / 1-of-3). So we
  do NOT rank by near-attack fraction; it is a context column.
- **Transferability caveat:** that validation is within the **ANL/adenylating
  superfamily** (SfaB homologs, Pα). Our expanded pool is **mostly glutamine-synthetase
  superfamily** (GS; acyl-phosphate, Pγ) — a different fold and mechanism. The proxy's
  calibration does NOT transfer across superfamilies. Therefore:
  - Rank **within family** (GS vs GS, NIS vs NIS); never a cross-family leaderboard.
  - Treat the GS-heavy results as **hypothesis generation**, not validated prediction,
    pending in-family activity validation (GS enzymes with measured amide synthesis).
- `step1_pool_ranking.{tsv,png}`: within-family ranking of the 41 conclusive candidates
  (GS 22, NIS 17, BacD 2); 83 inconclusive (adipate rarely within 4 Å).

## Interpretation guardrails
- Step-1 (activation) is the validated discriminator; step-2 (accommodation) is a
  steric feasibility screen, not an activity ranking.
- `new_hardhost`/`name_and_homology_inferred` candidates are hypotheses, not confirmed
  activities. Expressibility (`pool.tsv` organism) is a downstream ordering annotation.
