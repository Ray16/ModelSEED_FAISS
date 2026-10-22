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
6. Step-2 (acyl intermediate + neutral-amine HMDA + Mg [+ADP]; accommodation) only on
   step-1 survivors. PENDING.

## Known pitfalls (do not repeat)
- **batched MSA output dirs are per-run.** Writing a 2nd batch into a populated MSA dir
  cross-mapped index.tsv/CSVs (hash named for one seq, content of another). Always a
  fresh `--out` dir per run, and verify MSA query==sequence before folding. The 86 set
  was verified clean; the 25 are being regenerated in `msa_hard/`.

## Interpretation guardrails
- Step-1 (activation) is the validated discriminator; step-2 (accommodation) is a
  steric feasibility screen, not an activity ranking.
- `new_hardhost`/`name_and_homology_inferred` candidates are hypotheses, not confirmed
  activities. Expressibility (`pool.tsv` organism) is a downstream ordering annotation.
