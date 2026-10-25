# GS control matrix — native amine + multi-seed

Follow-up to the amine-free control: does the native amine co-substrate rescue glutamate
engagement, and is the deficit seed-robust? Design: 2 enzymes × (no amine / +putrescine) ×
seeds {42,1,7}, stock Boltz-2 v2.2.1 on lambda1 V100, 50 samples each (150/condition).
Substrate verified from UniProt: **both enzymes are γ-glutamyl-polyamine ligases** —
PuuA = glutamate–putrescine ligase; GlnA3 = γ-glutamyl-polyamine synthetase (putrescine/
cadaverine/spermidine/spermine, prefers long polyamines). **Neither uses ammonia.** Putrescine
is a genuine substrate for both. Metric unchanged: γ-carboxylate O → Pγ → O3B, <4 Å engagement.

## Pooled result (3 seeds, 150 poses)

| enzyme | condition | poses <4 Å | frac <4 Å | median (Å) | in-line angle | near-attack |
|---|---|---:|---:|---:|---:|---:|
| PuuA (P78061)  | no amine     | 0 / 150  | 0.000 | 5.66 | — | 0.000 |
| PuuA (P78061)  | **+putrescine** | **0 / 150** | 0.000 | 5.59 | — | 0.000 |
| GlnA3 (Q9KZC7) | no amine     | 3 / 150  | 0.020 | 4.38 | 159.5° | 0.013 |
| GlnA3 (Q9KZC7) | **+putrescine** | **56 / 150** | **0.373** | 4.09 | 158.6° | 0.293 |

Per-seed (n<4Å/50): PuuA free 0/0/0, +put 0/0/0; GlnA3 free 1/2/0, +put **16/24/16**.
Figure: `matrix_amine_rescue.png`.

## Findings

1. **The deficit is seed-robust, not a fluke.** Amine-free: PuuA 0/0/0, GlnA3 1/2/0 across
   three independent seeds. The single-seed numbers held.

2. **The native amine RESCUES engagement — for GlnA3.** Adding putrescine moves GlnA3 from
   3/150 (2%) to **56/150 (37%)** near-attack, median 4.38→4.09 Å, near-attack fraction
   0.013→0.293, with the in-line angle preserved (~159°) and seed-robust (16/24/16). So the
   complete native-like Michaelis complex (glutamate + diamine + ATP + 2 Mg) **does** assemble
   productive near-attack geometry on this fold. **"The GS fold can't produce near-attack
   geometry" is refuted.**

3. **Putrescine does NOT rescue PuuA.** 0/150 → 0/150; glutamate stays ~5.6 Å from Pγ with the
   full native substrate set, across all three seeds. PuuA is also modeled with **lower Boltz
   confidence** (interface iptm 0.87 vs GlnA3 0.96; ligand_iptm 0.87 vs 0.96), so its
   non-engagement is consistent with a weaker model of its catalytic state — not evidence that
   PuuA (a well-characterized active enzyme) is a poor catalyst.

## Consequences for the pipeline / headline

- **In-line ANGLE remains the robust, transferable signal** (~159–162° whenever engaged, every
  enzyme/condition). Keep using it.
- **Near-attack ENGAGEMENT transfers on this fold, but only given a COMPLETE complex and a
  well-modelled enzyme.** The pool-wide screen used the **amine-free single-ligand** fold, which
  *underestimates* engagement (native glutamate barely engages amine-free but reaches 37% with
  putrescine). So the amine-free engagement counts are not native-calibrated and should be
  caveated; if engagement is used for GS, use the acyl-donor + amine complex and weight by
  confidence.
- **Per-candidate GS engagement is unreliable (model-quality-dependent); family-level
  feasibility is supported.** GlnA3 is the **better-validated** top candidate — its complete
  complex assembles productive, in-line near-attack geometry for a diamine. PuuA's pipeline
  numbers rest on a lower-confidence model and should be flagged, not ranked against GlnA3.
- **Project-level:** both top GS hits are native **diamine/polyamine γ-glutamyl ligases**; they
  natively condense an activated acyl group with a diamine — the nylon chemistry exactly (HMDA is
  C6; native putrescine C4, cadaverine C5). The target is adjacent to their native reaction.

## Remaining caveats
- Static cofold can't model induced-fit/domain closure; PuuA may need that (and a better model).
- GlnA3's preferred substrates are the longer polyamines (spermidine/spermine); putrescine
  (used here for cross-enzyme comparability) is a weaker-but-genuine substrate, so 37% may be a
  floor for GlnA3's true engagement.

Data: `matrix_geometry.json`; inputs `inputs/*.yaml`; models `stage/<name>_s<seed>/...` (gitignored).
