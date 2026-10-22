# Expanded candidate screen — state of record

> **Where are the files / cofolded structures?** See [LOCATIONS.md](LOCATIONS.md).

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
   - 25 hard-host → `msa_hard/` (fresh dir). DONE (25/25).
4. Step-1 cofold (protein + ATP + Mg + adipate; activation geometry): DONE — all 124
   folded (86 expressible + 25 hard-host + 13 references), 0 failures.
   **All step-1 ensembles now live in ONE folder: `../../cofold_step1/`** (previously split
   across `campaign/`, `campaign_hard/`, `top10_cofold/campaign/`; consolidated 2026-10-22).
5. Step-1 geometry across all 124. DONE (`step1_pool_ranking.{tsv,png}`; 41 conclusive,
   83 inconclusive because adipate rarely sits within 4 Å).
6. Step-2 (acyl intermediate + neutral-amine HMDA + Mg [+ADP]; accommodation) for
   **ALL candidates, not just step-1 survivors** (no funnel). DONE — all 124 folded,
   0 failures. **All step-2 ensembles now live in ONE folder: `../../cofold_step2/`**
   (previously split across `top10_cofold_step2/` and `step2/`; consolidated 2026-10-22).
   Rationale: step-1 and step-2 are independent — BacD activates best (step-1 161°)
   but cannot fit HMDA (step-2 reachable 0.14), while PuuA is a weaker activator yet
   accommodates the amine (step-2 0.44). Funnelling step-2 on step-1 hits would miss
   real accommodators.
7. Combined step-1 + step-2 view across all 124. DONE — `pool_combined.{tsv}`,
   `pool_combined_within_family.png` (conservative), `pool_combined_cross_family.png`
   (hypothesis). See "Top candidates" below.

## Top candidates (combined step-1 activation + step-2 accommodation)

Only the **GS family** places candidates in the both-feasible corner (step-1 engaged
activation angle ≥150° AND step-2 HMDA reach ≥0.3). NIS enzymes activate adipate at
competitive angles but essentially **cannot** accommodate the C6 diamine (reach ≈ 0);
BacD activates best of all (161°) but also fails step-2 (0.14). This is the clearest
cross-family signal: for the nylon-6,6 diad, GS-fold enzymes are the ones where both
half-reactions look geometrically feasible.

Both-feasible GS candidates (step-1 angle°, step-2 reach):
- GlnA3 (Q9KZC7)                   159°, 0.74   — known best hit; HARD host
- PuuA (P78061)                    162°, 0.44   — reference GS
- A0A4R2L843 (C.halotolerans)      154°, 0.80   — NEW (hard-host)
- A0A1S1H9I3 (E.haloaromaticamans) 153°, 0.88   — NEW (hard-host)
- A0AAN0RL94 (P.temperata)         151°, 0.40   — NEW (expressible)

Caveat: step-1 angle is the validated discriminator only *within* a family (ANL/
adenylating calibration); the cross-family corner is **hypothesis generation**. Step-2
reach is steric feasibility, not an activity ranking. Decisive next test = GS
native-substrate positive control (see transferability memory).

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
