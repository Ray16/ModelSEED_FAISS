# Charge-only imbalance — diagnosis (1,844 reactions)

Reactions where mass (incl. H) balances but net charge ≠ 0. Tested read-only via
`charge_only_explore.py`.

## Verdict: NOT a deterministic fix — this class is redox with implicit electrons

- **58.8% (1,084)** contain a known electron carrier or O₂.
- **|Δq| is overwhelmingly even**: ±2 = 1,212 (66%), ±4 = 302, ±6 = 124, ±8 = 56, ±10 = 42…
  This is the n-electron-transfer signature, not a random charge error.
- Compound charges are **correct**: only **27 / 1,844** have any stored-charge ≠
  structure-charge mismatch, and only **8** are actually zeroed by fixing it.

So "reassign compound charge from ChemAxon" — my initial plan for this class — would
fix ~8 reactions. Rejected as a tier. The imbalance is transferred electrons that
ModelSEED does not represent as mass-bearing species; the real curated fix is usually
a missing electron acceptor/donor or a reconsidered cofactor protonation. That is
per-reaction chemistry → **agent-based repair**, not a heuristic.

## Two things to carry forward

1. **A handful of genuine compound-charge bugs** worth flagging (fix the compound,
   which propagates to all its reactions). Clearest: `cpd00532` superoxide (O2−)
   stored charge **0**, structure **−1** (affects 9 reactions). Others are
   metalloporphyrins/chlorophylls where RDKit-vs-pH7 disagreement may be a
   pH-convention false positive — review, don't auto-apply. NOTE: RDKit formal charge
   from the stored SMILES is NOT a general defect detector, because a neutral SMILES
   legitimately differs from the pH-7 majormicrospecies charge; it only flags cases
   where the SMILES is *explicitly* charged differently than the stored value.

2. **Redox reactions are the dominant remaining defect mode.** They recur across the
   charge-only class AND likely inside the heavy-skeleton class. The agent-repair
   harness should be redox-aware: identify the couple, restore the missing
   acceptor/donor + its protons, then re-verify mass+charge with the same
   `msbio.compute_residual` gate used by the deterministic corrector.

## Where this leaves the deterministic frontier

The heuristic/deterministic yield is essentially exhausted:
- proton/water close: **310** (done, verified)
- genuine compound-charge bugs: **~8** (flag for review)

Everything else — charge-only redox (1,844), oxygen-only (318), H/O/charge-inconsistent
(242), heavy-skeleton (~1,560) — requires reasoning about the specific chemistry.
That is the boundary between the heuristic phase and the agent phase.
