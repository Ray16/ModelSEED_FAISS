# TODO — mass/charge balance autocorrection

Status: **521 corrections, all balance-verified + mechanism-class-consistent + 97% KEGG-consistent.**
Nothing applied to `ModelSEEDDatabase/` (proposals only). Pushed to branch `mass-charge-balance`.
See `FINDINGS.md` (what/why), `REFLECTION.md` (why the wrong fixes happened), `corrections_log.tsv` (before/after per reaction).

Re-run everything: `proton_water_balance.py` → `cosubstrate_closure.py` → `couple_closure.py`
→ (subagents) → `apply_couples.py` → `apply_phospho.py` → `twoog_demethylase.py`
→ `build_correction_log.py` → `verify_corrections.py` → `plot_corrections.py`.

---

## Now / high priority

- [ ] **Open the PR** for `mass-charge-balance` (branch pushed; PR not opened).
- [ ] **Deliver the 521 in a usable form** (track 2): emit a single corrected-reactions file
      in ModelSEED stoichiometry format + a MEMOTE-style before/after balance report.
      Decide the apply path: additive overlay applied at load time vs. upstream PR to
      ModelSEEDDatabase. (User: never overwrite originals.)
- [ ] **Decide what to do with the confidence tiers.** proton/water (309) + high-EC
      co-substrate/couple are solid; the 77 "review" co-substrate + agent tier are proposals.
      Ship all with tier labels, or gate delivery to high-confidence only?

## Validation (extend the KEGG check)

- [ ] **MetaCyc cross-check** — 69 corrected reactions have MetaCyc xrefs not covered by
      KEGG. MetaCyc reaction data isn't in the repo and the site may be gated; needs a data
      source. Same participant-presence + contradiction logic as `validate_kegg*.py`.
- [ ] **Estimate precision on the no-xref tier** (339 reactions have no KEGG xref, incl. most
      plant/secondary-metabolite reactions). Consider an LLM-judge sample or expert spot-check.
- [ ] Rhea cross-check is blocked (Cloudflare); revisit via the EBI/Rhea REST API or a bulk
      download if broader coverage is wanted (only 12 reactions, low priority).

## More correction rounds (supervised — lower confidence)

- [ ] **Oxygen-only hydroxylations** (~297; only 70 have EC). Monooxygenase template
      `S + O2 + NAD(P)H + H+ → S-OH + H2O + NAD(P)+` balances, but the reductant (NADH vs
      NADPH vs other) is ambiguous without EC/name. Restrict to EC 1.14.13 (NADPH) / 1.14.12
      (NADH) subsets; agent picks/skip. Watch for reactions that already contain O2/reductant.
- [ ] **Redox charge-only** (~1,500): missing electron acceptor/donor. Mostly NOT fixable
      (electrons implicit); precision-first says flag, don't fix. Only tackle sub-patterns
      where the acceptor is named.
- [ ] **Heavy-skeleton errors** (~1,500, `agent_triage.tsv`): genuine stoichiometry/compound
      errors; curation-level. Triage for any clean sub-patterns (e.g. C6H10O5 polymer units).

## Robustness / tooling

- [ ] **Extend the EC prefilter into a positive mechanism map** (currently excludes wrong
      classes). E.g. EC 4.1.1 ⇒ decarboxylation (CO2), EC 2.8.2 ⇒ sulfotransfer — route by EC
      instead of trying all couples. Would raise auto-accept rate and cut agent load.
- [ ] **Pre-triage EC before the agent pools** so oxidoreductase/hydrolase reactions never
      reach the methyl/acetyl/glycosyl subagents (the prefilter now blocks them downstream,
      but they still consume agent tokens).
- [ ] **CLI packaging** (track 4): wrap the pipeline into one command that runs on any
      ModelSEED release and emits corrections + report + figures.
- [ ] Add a small **regression test**: the 521 should stay balanced and mechanism-consistent
      across re-runs; assert 0 degenerate / 0 EC-incompatible / 0 balance failures.

## Known issues / watch-list

- [ ] Glycosyl fixes: only 8/46 stereo-verifiable (balance is stereo-blind to hexose id);
      33 rest on the agent's name call. Flagged in `stereo_check.py` output — needs review or
      structure-defined sugars before trusting sugar identity.
- [ ] `couple_closure` still enumerates all couples then filters; the positive EC map above
      would make it cleaner and cheaper.
- [ ] Large regenerable intermediates committed (`landscape.tsv` 2.3M, `agent_triage.tsv`);
      consider gitignoring if repo size matters.

## Done (for reference)

- [x] Imbalance landscape + classification (`explore_landscape.py`, figures).
- [x] Deterministic proton/water + co-substrate closure.
- [x] Cofactor-couple closure + 3 Sonnet subagents (methyl/glycosyl/acetyl) by name.
- [x] Phospho precision triage (40 keep-Pi, 35 retract).
- [x] Degeneracy guard, stereo tier, before/after logging, independent audit.
- [x] KEGG cross-validation (97% consistent; caught 3 errors).
- [x] EC prefilter (removed 24 mechanism-wrong picks) + 2-OG dioxygenase corrector.
- [x] Pushed to branch `mass-charge-balance`.
