#!/usr/bin/env python
"""
Independent per-stage balance audit.

Re-reads each stage's corrections file, re-parses the corrected_stoichiometry FROM
THE FILE (not from any in-memory object), and recomputes mass + charge balance from
scratch. Every corrected reaction must have an empty residual. Also re-confirms the
ORIGINAL reaction was actually imbalanced (so we're not "correcting" balanced rxns).

Run this after any stage. Exit code != 0 if any corrected reaction fails to balance.
"""
import os
import sys
import pandas as pd

import msbio

STAGES = [
    ("proton/water",     "proton_water_corrections.tsv"),
    ("co-substrate",     "cosubstrate_corrections.tsv"),
    ("couple (auto)",    "couple_corrections.tsv"),
    ("2OG-demethylase",  "twoog_corrections.tsv"),
    ("couple (agent)",   "couple_agent_corrections.tsv"),
    ("phospho (agent)",  "phospho_corrections.tsv"),
]


def main():
    cpd = msbio.load_compounds()
    info = msbio.SpeciesInfo.from_compounds(cpd)
    rxn = msbio.load_reactions(True)
    orig_stoich = dict(zip(rxn["id"], rxn["stoichiometry"]))

    # reactions retracted/reassigned by the phospho precision round -> excluded from the
    # co-substrate stage's final count (they still balance, but we chose not to claim them)
    override_ids = set()
    if os.path.exists("phospho_overrides.tsv"):
        try:
            override_ids = set(pd.read_csv("phospho_overrides.tsv", sep="\t", dtype=str)["id"])
        except pd.errors.EmptyDataError:
            pass

    overall_fail = 0
    reconciled_total = 0
    print("===== PER-STAGE MASS/CHARGE BALANCE AUDIT =====\n")
    for label, path in STAGES:
        try:
            df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        except FileNotFoundError:
            print(f"[{label}] file missing: {path}\n")
            continue
        except pd.errors.EmptyDataError:
            print(f"[{label}] no corrections (empty): {path}\n")
            continue

        # drop rows retracted/reassigned by the phospho round (co-substrate stage only)
        if "cosubstrate" in path:
            df = df[~df["id"].isin(override_ids)]
        n = len(df)
        reconciled_total += n
        bad_corrected = []
        orig_was_balanced = []
        for _, r in df.iterrows():
            # 1) corrected reaction must balance to zero
            sp = msbio.parse_stoich(r["corrected_stoichiometry"])
            res, flags = msbio.compute_residual(sp, info)
            if res or flags["no_formula"] or flags["unknown_charge"]:
                bad_corrected.append((r["id"], res, flags))
            # 2) original must have been imbalanced (sanity: we fixed a real defect)
            o = orig_stoich.get(r["id"])
            if o is not None:
                ores, oflags = msbio.compute_residual(msbio.parse_stoich(o), info)
                if not ores and not oflags["no_formula"] and not oflags["unknown_charge"]:
                    orig_was_balanced.append(r["id"])

        ok = n - len(bad_corrected)
        status = "PASS" if not bad_corrected else "FAIL"
        print(f"[{label}]  {path}")
        print(f"    corrected reactions: {n}")
        print(f"    verified mass+charge balanced: {ok}/{n}   -> {status}")
        if bad_corrected:
            overall_fail += len(bad_corrected)
            for rid, res, fl in bad_corrected[:10]:
                print(f"      FAIL {rid}  residual={res} flags={fl}")
        if orig_was_balanced:
            print(f"    WARNING: {len(orig_was_balanced)} originals were already balanced "
                  f"(unexpected): {orig_was_balanced[:5]}")
        # confidence split if present
        if "confidence" in df.columns:
            print(f"    confidence: " +
                  ", ".join(f"{k}={v}" for k, v in df['confidence'].value_counts().items()))
        print()

    print(f"TOTAL corrected (reconciled, after {len(override_ids)} phospho retractions): {reconciled_total}")
    print(f"AUDIT RESULT: {'ALL BALANCED [OK]' if overall_fail == 0 else f'{overall_fail} FAILURES'}")
    sys.exit(1 if overall_fail else 0)


if __name__ == "__main__":
    main()
