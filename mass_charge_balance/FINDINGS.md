# Mass/Charge Balance Autocorrection — Exploration Findings

Goal: build a tool that autocorrects mass & charge balance of ModelSEED reactions.
Separate effort from `thermodynamic_calc/`.

## Data sources (in ModelSEEDDatabase/)

- **Reactions**: `Biochemistry/reaction_*.tsv` (61 shards, 56,012 rows; 48,403 active).
  - Key cols: `stoichiometry` (`n:cpdid:cmpt:comm:"name"`, negative=reactant),
    `status` (**already encodes the imbalance**), `is_obsolete`, `code` (pre-protonation),
    `equation` (post-protonation).
  - `status` vocabulary (REACTIONS.md): `OK`, `MI:C:-1/H:-4/...` (mass imbalance, +=RHS heavy),
    `CI:n` (charge imbalance, +=RHS heavier charge), `HB` (balanced by adding H),
    `EMPTY`, `CPDFORMERROR` (missing/invalid formula). `10000000`-ish = unknown sentinel.
- **Compounds**: `Biochemistry/compound_*.tsv` (45,708). Cols: `formula`, `charge`, `pka`, `pkb`, `smiles`, `inchikey`.
- **ChemAxon protonation results (pH 7.0)** — for the actual correction:
  - Master, cpd-keyed: `Biochemistry/Structures/All_ModelSEED_Structures_updated.txt`
    cols: `cpd | InChI/SMILE | Charged/Original | srcID | srcDB | formula | charge | string`.
    The `Charged` rows are the ChemAxon majormicrospecies formula+charge at pH 7.
  - Per-source (KEGG/MetaCyc/ChEBI/Rhea/MetaNetX): `Structures/<src>/InChI_Charged_Formulas_Charges.txt`
    and `SMILE_Charged_Formulas_Charges.txt` (srcID -> formula, charge), plus
    `pKa_Strings.txt` (per-atom pKa/pKb: `id pKa 1:atom:value;...`).
  - Generator: `Scripts/Structures/ChargeMicrospecies_pKa.java` (ChemAxon pKaPlugin, hardcoded pH=7.0).
  - Marvin/ChemAxon installer already downloaded: `marvinws_unix_26.1.26 (1).sh` (repo root).

## Imbalance landscape (recomputed from formulas, 48,403 active reactions)

| bucket | count | % | meaning | fixable? |
|---|---|---|---|---|
| OK | 25,296 | 52.3% | already balanced | — |
| **A_no_formula** | 18,772 | **38.8%** | ≥1 species has generic/R-group/missing formula | NOT by stoichiometry — needs structure resolution |
| F_heavy_imbalance | 1,927 | 4.0% | skeleton off (C/O/N/P/S…) | genuine error — hard |
| E_charge_only | 1,844 | 3.8% | mass balanced, charge off | **protonation-state / proton fix** |
| D_H_and_O | 499 | 1.0% | only H and O off | **add H2O + H+** |
| B_unknown_charge | 61 | 0.1% | formula ok, charge sentinel/missing | needs charge assignment |
| C_H_only | 4 | 0.0% | only H off | **add H+** |

**Total imbalanced = 47.7%**, but **the "half imbalanced" figure is dominated by bucket A**
(38.8% = generic R-group/polymer formulas). Those are inherently unscoreable by mass balance
and can't be fixed by adding protons/water — they need structural definition of the R-group.

**The stoichiometrically-correctable slice is small and tractable:**
C + D + E (+B) ≈ **2,408 reactions (~5%)** are mass-balanced-or-nearly and just need
proton/water/charge reconciliation — this is the high-value, high-confidence target.
F (1,927, 4%) is genuine heavy-atom imbalance; some fixable (missing cofactor/water),
most are real errors requiring per-reaction chemistry.

## Cross-check vs stored `status`

Recomputed-OK (25,296) vs stored-OK (28,096) agree **81%**. The ~2,800 gap is worth
understanding before trusting either: likely the stored `stoichiometry` field is
pre- vs post-protonation in some rows, or ChemAxon charges differ from the `compound.charge`
column I used. **Next step: reconcile which charge/formula source is authoritative**
(compound.tsv `charge` vs ChemAxon `All_ModelSEED_Structures` charged formula).

## Correction roadmap — one strategy per class, deterministic → non-deterministic

Everything reads ModelSEEDDatabase **read-only**; every corrector writes proposals
into `mass_charge_balance/` and NEVER edits the originals. Counts are from
`proton_water_balance.py` over 48,403 active reactions (self-consistent, honest —
these supersede the earlier optimistic "5% deterministic" estimate).

| # | class | count | strategy | status |
|---|---|---|---|---|
| 0 | already balanced | 25,296 | — | — |
| **1** | **proton/water closable** | **310** | closed-form: w=−ΔO, p=−Δq, check ΔH+2w+p=0 | **DONE — `proton_water_balance.py`, all 310 verified + round-tripped** |
| 2 | charge-only imbalance | 1,844 | **DIAGNOSED (`charge_only_explore.py`): NOT deterministic.** ~99% are redox with implicit electrons (58.8% carry a redox carrier/O₂; |Δq| even, 66% = ±2). Only 8 are genuine compound-charge bugs. → agent phase | diagnosed, deferred to agent |
| 3 | oxygen-only missing species | 318 | H & charge balanced, only O off → a specific O-bearing co-substrate is missing (not water) | agent phase |
| 4 | H/O/charge inconsistent | 242 | H,O off in ratios water/proton can't close → combined missing species | agent phase |
| 5 | heavy-atom skeleton error | ~1,560 | C/N/P/S/halogen off → real stoichiometry errors; likely redox partners + skeleton fixes | agent phase |
| — | needs structure (R-group) | 18,772 | **refuse** — report "requires structure", never fabricate coefficients | policy |
| — | needs charge (unknown charge) | 61 | assign charge from ChemAxon first, then re-route | prereq |

Key correction: a **pure charge imbalance is not proton-fixable** — adding H+ to fix
charge injects an H and breaks mass. And it is NOT a compound-charge defect either:
the charge-only class is dominated by redox reactions missing an electron
acceptor/donor. See `charge_only_report.md`. Deterministic yield ends at proton/water
(310) + ~8 flagged compound bugs (`charge_defect_candidates.tsv`).

## The deterministic frontier is reached — remaining classes go to the agent phase

Architecture (per user): **an agent reasons once about a GROUP/PATTERN of reactions,
emits a code rule, and code applies it across all matching reactions** — never
per-reaction LLM calls (token-wasteful). Every agent-proposed correction passes back
through the same `msbio.compute_residual` gate the deterministic corrector uses:
a patch is only accepted if the reaction verifies mass+charge = 0. This keeps the
non-deterministic phase as safe as the deterministic one — the agent proposes,
`msbio` disposes. Redox-awareness (identify the couple, restore the missing
acceptor/donor + protons) is the dominant pattern to encode first.

## Stages implemented & independently audited (`verify_corrections.py`)

| stage | script | kept | audit |
|---|---|---|---|
| proton/water close (deterministic) | `proton_water_balance.py` | 309 | ✔ |
| co-substrate closure (CO₂/Pᵢ/PPᵢ/NH₄/SO₄) | `cosubstrate_closure.py` | 98 (8 high/13 low/77 review) | ✔ |
| cofactor-couple, auto (EC-corroborated) | `couple_closure.py` | 14 | ✔ |
| cofactor-couple, agent-disambiguated | `apply_couples.py` (3 Sonnet subagents) | 121 | ✔ |
| **total kept** | | **542** | **all balance-verified** |
| phospho precision retractions | `apply_phospho.py` (agent) | −35 | wrong Pᵢ mechanism removed |

`verify_corrections.py` re-reads each stage's output from disk, recomputes mass+charge
from scratch, honors the phospho overrides, and exits ≠0 on any failure. Degenerate
results (fix cancels the reaction to empty — caught in main-agent review) are rejected
by `msbio.is_valid_reaction`; 7 removed.

Agent phase: 199 ambiguous couple reactions → 3 cheap Sonnet subagents disambiguate by
reaction NAME (arithmetic can't; conservative — methyl 63/105, acetyl 14/28, glycosyl
skipped galactose/fucose/fructan). A 4th subagent triaged the phospho ambiguity:
40 confirmed free-Pᵢ, 35 retracted (polyphosphate enzymes, CoA-ligases with AMP already
present, mislabeled terpene "geranyl phosphate" — none fixable by the Pᵢ/NTP menu, so
retracted rather than mis-fixed). Glycosyl fixes carry a structure/stereo tier
(`stereo_check.py`): only 8/46 stereo-verifiable (balance is stereo-blind to hexose id).

## Design principle going forward: CONTEXT decides the correction

Arithmetic balance is necessary, not sufficient. The **same residual has different
correct fixes depending on context** (EC class + which compounds are already present +
reaction name):
- missing phosphate group + EC 3.1.3 phosphatase (has dephospho-product) → add free **Pi** ✔
- missing phosphate group + EC 2.7.1 kinase (has ADP, no ATP) → restore **ATP→ADP couple**, NOT free Pi ✗

Co-substrate closure already uses EC corroboration to gate confidence (`EC_CORROBORATION`
deliberately excludes ATP-dependent kinases from the Pi rule for this reason). The agent
phase generalizes this: the agent reads context, derives a context→fix rule per group,
code applies it, `msbio.compute_residual` verifies. Higher context = higher success rate.
Next context-aware rule to encode: kinase/nucleotidyltransferase donor-couple restoration
(pick ATP/GTP/UTP/CTP from the reaction name).

## Cofactor-couple stage (agent-disambiguated) + known pipeline issue

Couple closure (`couple_closure.py`) generalizes co-substrate closure to donor/acceptor
couples (SAM/SAH, UDPglc/UDP, GDPman/GDP, AcCoA/CoA, ATP/ADP, ...): insert one couple,
close remainder by proton/water, verify. Auto-accepts only unambiguous + EC-corroborated
(16: 12 methyl, 4 acetyl). Ambiguous/no-EC (199) → cheap Sonnet subagents disambiguate by
reaction NAME (which arithmetic can't): methyl(105), glucosyl-vs-mannosyl(66), acetyl(28).
Agent picks flow through `apply_couples.py` → balance-verified → logged. First pass:
acetyl 14/28, glycosyl 36 glc/10 man/20 skip.

Glycosyl gets a targeted STRUCTURE/STEREO tier (`stereo_check.py`) because balance is
stereo-blind to hexose identity: of 46 glycosyl fixes only 8 stereo-verifiable, 5
scaffold-ok-but-stereo-undefined, 33 unverifiable (rest on the name).

**KNOWN PIPELINE ISSUE to fix (phospho ambiguity):** cosubstrate (Pi) runs BEFORE couple
(ATP/ADP) and greedily claims phosphoryl-deficit reactions with free Pi. But a reaction
off by a phosphoryl is closeable by EITHER free Pi (phosphatase) OR ATP→ADP (kinase) — a
context decision. Fix: route Pi-review-confidence reactions that ALSO close via a
nucleotide couple to an agent (phosphatase/phosphorylase→Pi vs kinase→ATP/ADP donor).
This is the next subagent round.

## External validation against KEGG (independent ground truth)

`validate_kegg.py` + `validate_kegg_contradictions.py`: for each corrected reaction with
a KEGG alias, fetch KEGG's reaction (rest.kegg.jp, cached in `kegg_cache.tsv`) and check
whether the cofactor we RESTORED appears in KEGG's participant list. Only co-substrate/
couple fixes are checkable (KEGG omits the H+/H2O that proton-water adds). 200/539 have a
KEGG xref; 104 are cofactor-checkable.

A missing cofactor in KEGG is only a real error if KEGG shows a DIFFERENT mechanism
(a contradicting cofactor we lack); if KEGG merely omits the same cofactor, our fix is
plausibly right and KEGG is incomplete. Refined verdict:

| verdict | n | meaning |
|---|---|---|
| validated | 18 | our cofactor IS in KEGG |
| kegg_incomplete | 83 | KEGG omits the same cofactor, no contradicting marker → keep |
| kegg_contradicts | 3 | KEGG shows another mechanism → **retracted** |

**97% (101/104) consistent with KEGG.** The 3 contradictions were all the same real agent
error: morphine-alkaloid steps (rxn02627/03475/03476) that are 2-oxoglutarate/O2
dioxygenases, which the methyl agent misread as SAM/SAH methylations. Retracted via
`kegg_retractions.tsv` (build_correction_log drops them globally).

## EC prefilter + 2-OG dioxygenase fix (the KEGG failure, fixed at the root)

The 3 KEGG contradictions were one failure mode: a transferase couple offered to an
oxidoreductase. Root cause: EC was used only to BOOST confidence, never to EXCLUDE. Fix:
`couple_closure.ec_compatible` gates transferase couples (class 2) to reactions whose EC is
null or shares the couple's class — applied in candidate generation AND `apply_couples`
(defense vs stale decisions). This removed **24** mechanism-wrong picks: 12 EC-class-1
oxidoreductases (2-OG dioxygenases) + 12 EC-3.2.1 glycoside **hydrolases** the agent had
given sugar-nucleotide donors (they hydrolyse with water, e.g. rxn05973 `H2O ⇌ D-Mannose`).
KEGG had caught only 3 of these 24 (the rest lacked xrefs) — the prefilter generalizes it.

`twoog_demethylase.py` supplies the CORRECT fix for 2-OG dioxygenase O-demethylations:
rebuilds `R-OCH3 + 2OG + O2 → R-OH + formaldehyde + succinate + CO2` (matching KEGG exactly;
rxn02627 == KEGG R03698), guarded to touch only actually-imbalanced reactions. See `REFLECTION.md`.

**FINAL: 521 corrections, all balance-verified, mechanism-class-consistent, 97% KEGG-consistent.**

## Outputs so far
- `landscape.tsv` — per-reaction diagnosis (bucket, residual, charge imbalance).
- `imbalance_landscape.png`, `heavy_imbalance_signatures.png` (also in `figures/` as `mcb_*`).
- `proton_water_corrections.tsv` — 310 proposed corrections: residual, patch
  (e.g. "2 H2O → products"), verified flag, corrected human-readable equation,
  and corrected stoichiometry string (drop-in ModelSEED 4-field format).
- `proton_water_report.txt` — summary + deferral reasons (= the roadmap classes).
- `msbio.py` — shared read-only loaders + formula/stoich parsing (4-field `coeff:cpd:cmpt:"name"`).
