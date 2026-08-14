# Reflection: why the SAM/SAH (and glycosyl) fixes were wrong

External validation against KEGG caught 3 wrong methylations; adding an EC prefilter
then revealed 24 total mechanism-wrong couple picks (12 oxidoreductases, 12 hydrolases).
Root causes, most fundamental first:

## 1. Mass balance is mechanism-blind — and direction-blind
A CH₂ residual is equally consistent with "SAM **adds** a methyl" (methyltransferase) and
"a 2-oxoglutarate dioxygenase **removes** a methyl as formaldehyde" (oxidoreductase). Balance
cannot tell addition from removal, nor which cofactor mediates it. I used balance as the
correctness **gate**, but it is necessary, not sufficient. A wrong-but-balanced fix sails
through it invisibly. The whole agent tier rested on "if it balances and the name fits, keep it."

## 2. I treated EC as soft corroboration, never a hard constraint
The EC number encodes the reaction's **mechanism class** — orthogonal ground truth that
balance and even the reaction name lack. EC 1.14.11 is an oxidoreductase; EC 3.2.1 a
glycoside **hydrolase**; group-transfer couples are class 2 transferases. I used EC only to
*boost confidence when it matched* (`EC_CORROBORATION`), never to *exclude when it
contradicted*. That asymmetry is the core bug: a transferase couple should be **impossible**
on a class-1 or class-3 enzyme, but nothing enforced it. Fix = EC class is a gate, not a bonus
(`ec_compatible` in couple_closure + apply_couples).

## 3. Both filters in the pipeline were mechanism-blind for these cases
Candidate generation offered *any arithmetically-closing couple*; the agent pruned only by
*name*. For a reaction stored as a bare `Morphinone ⇌ Oripavine`, neither arithmetic nor the
name reveals the mechanism — only the EC did, and no stage used it as a filter. Two blind
filters in series are still blind.

## 4. ModelSEED's reversed, cofactor-stripped reactions hid the one real signal
These reactions were stored **backwards** (product gains the methyl) and stripped of their
2-OG/O₂/succinate/CO₂/formaldehyde cofactors, so the equation looked exactly like a
methylation. The agent had no cofactors in the equation to see the dioxygenase; the single
discriminating feature present was the EC number — which I ignored. Degraded input amplified
the blind spot.

## 5. The cheap subagent anchored on a surface cue
"Demethylating" in the enzyme name + a CH₂ residual → "methylation," without asking "what is
enzyme class 1.14.11?" A small model anchors on the salient token. Lesson: don't rely on the
agent to know enzyme taxonomy — encode the hard mechanism constraint (EC class) in code
*around* it.

## Meta-lesson and the architectural fix
Balance-gating makes wrong-but-balanced fixes cheap to propose and invisible to the gate.
Robustness needs an **orthogonal check that sees what balance can't** — here, mechanism class
from EC. External validation (KEGG) is such a check but only covers reactions with xrefs
(it caught 3 of 24). The durable fix is to put the mechanism constraint **upstream** as a
prefilter, so mechanism-impossible candidates are never generated — not to rely on
downstream validation with partial coverage. Defense in depth: prefilter (EC class) →
balance gate → degeneracy guard → external cross-check, each catching what the others miss.

## What changed as a result
- `couple_closure.ec_compatible`: transferase couples offered only if EC is null or shares
  the couple's top-class. Applied in candidate generation AND in apply_couples (defense vs
  stale agent decisions). Removed 24 mechanism-wrong picks (12 EC-1, 12 EC-3.2.1).
- `twoog_demethylase.py`: the correct fix for the 2-OG dioxygenase O-demethylations
  (rebuilds `R-OCH3 + 2OG + O2 → R-OH + formaldehyde + succinate + CO2`, matching KEGG),
  guarded to only touch actually-imbalanced reactions.
- Final: 521 corrections, all balance-verified, and now mechanism-class-consistent.
