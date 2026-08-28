#!/usr/bin/env python
"""
Deliverable-level sanity audit of corrected_reactions.tsv.

Two layers, trusting none of the pipeline's own bookkeeping (everything is recomputed from
the corrected stoichiometry string on disk):

  (1) BALANCE   - re-parse and recompute residual; assert mass+charge = 0 and non-degenerate.
  (2) CHEMISTRY - flag corrections that balance but look chemically suspicious. The two hard
                  guards (indefinite polymer, phosphate mechanism) share mcb.closures with the
                  pipeline, so a clean audit proves the guards actually fired upstream.

Returns the number of hard problems (balance failures + guard violations); 0 = clean.
Writes audit_flags.tsv (soft review flags) for human triage.
"""
from collections import Counter, defaultdict

import pandas as pd

from mcb import io as msbio
from mcb import closures as cl
from mcb.paths import wp

FREEBIES = {"cpd00001", "cpd00067", "cpd00011", "cpd00009", "cpd00012", "cpd00013",
            "cpd00048", "cpd00007", "cpd00024", "cpd00036", "cpd00055"}


def multiset(species, sign):
    c = Counter()
    for s in species:
        if (s["coeff"] < 0) == (sign < 0) and abs(s["coeff"]) > 1e-9:
            c[s["cpd"]] += abs(s["coeff"])
    return c


def run():
    cpd = msbio.load_compounds()
    info = msbio.SpeciesInfo.from_compounds(cpd)
    df = pd.read_csv(wp("corrected_reactions.tsv"), sep="\t", dtype=str, keep_default_na=False)

    balance_fail, degenerate, unscoreable = [], [], []
    guard_violations = []          # HARD: a polymer/phosphate-conflict correction slipped through
    flags = defaultdict(list)      # SOFT: review-only heuristics

    for _, r in df.iterrows():
        rid = r["reaction_id"]
        before = msbio.parse_stoich(r["stoichiometry_before"])
        after = msbio.parse_stoich(r["stoichiometry_corrected"])

        res, fl = msbio.compute_residual(after, info)
        if fl["no_formula"] or fl["unknown_charge"]:
            unscoreable.append(rid)
        if {k: v for k, v in res.items() if abs(v) > 1e-6}:
            balance_fail.append((rid, res))
        if not msbio.is_valid_reaction(after):
            degenerate.append(rid)

        # hard guards (must have been applied upstream). Judge only what the CORRECTION did,
        # not pre-existing curator content: a reaction that already contains Pi/PPi beside a
        # nucleotide couple (dikinases, apyrases) is the curator's business, not our error.
        if cl.is_refused_reaction(after) or cl.is_refused_reaction(before):
            guard_violations.append((rid, "indefinite-polymer species"))
        before_cpds = {s["cpd"] for s in before}
        after_cpds = {s["cpd"] for s in after}
        for pc in (cl.CPD_PI, cl.CPD_PPI):
            if pc in (after_cpds - before_cpds) and \
               cl.phosphate_cosubstrate_conflict(after_cpds - {pc}, pc):
                guard_violations.append((rid, f"phosphate-mechanism conflict ({pc}) ADDED"))

        # soft review flags
        rb, pb = multiset(before, -1), multiset(before, +1)
        ra, pa = multiset(after, -1), multiset(after, +1)
        added = (ra - rb) + (pa - pb)
        n_added_species = len({c for c in added if c not in rb and c not in pb})
        if n_added_species > 2 or sum(added.values()) > 4:
            flags["large_injection"].append(rid)
        res_b, _ = msbio.compute_residual(before, info)
        if abs(res_b.get("charge", 0)) >= 3:
            flags["large_charge_before(|q|>=3)"].append(rid)
        if set(pa - pb) & set(rb):
            flags["added_product_equals_reactant"].append(rid)

    n = len(df)
    print(f"===== BALANCE AUDIT ({n} corrections) =====")
    print(f"  balance failures (mass/charge != 0): {len(balance_fail)}  {balance_fail[:8]}")
    print(f"  degenerate (empty/no-op) after fix : {len(degenerate)}  {degenerate[:8]}")
    print(f"  unscoreable species in result      : {len(unscoreable)}  {unscoreable[:8]}")
    print(f"  GUARD VIOLATIONS (should be 0)      : {len(guard_violations)}  {guard_violations[:8]}")

    print("\n===== CHEMISTRY FLAGS (balanced, review-only) =====")
    for k, ids in sorted(flags.items(), key=lambda kv: -len(kv[1])):
        print(f"  {k:36s} {len(ids):4d}   e.g. {ids[:6]}")

    flagged = defaultdict(list)
    for k, ids in flags.items():
        for rid in ids:
            flagged[rid].append(k)
    dd = df.set_index("reaction_id")
    rows = [{"reaction_id": rid, "flags": "|".join(fs),
             "method": dd.loc[rid]["method"] if rid in dd.index else "",
             "tier": dd.loc[rid]["tier"] if rid in dd.index else "",
             "equation_after": dd.loc[rid]["equation_after"] if rid in dd.index else ""}
            for rid, fs in flagged.items()]
    pd.DataFrame(rows).sort_values("flags").to_csv(wp("audit_flags.tsv"), sep="\t", index=False)
    print(f"\n  wrote audit_flags.tsv ({len(rows)} soft-flagged reactions)")

    hard = len(balance_fail) + len(degenerate) + len(unscoreable) + len(guard_violations)
    print(f"\nAUDIT RESULT: {'CLEAN [OK]' if hard == 0 else f'{hard} HARD PROBLEMS'}")
    return hard


if __name__ == "__main__":
    import sys
    sys.exit(1 if run() else 0)
