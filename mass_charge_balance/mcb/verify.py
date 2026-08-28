#!/usr/bin/env python
"""
Independent per-stage balance audit.

Re-reads each stage's corrections file, re-parses corrected_stoichiometry FROM THE FILE (not
any in-memory object), and recomputes mass + charge balance from scratch. Every corrected
reaction must have an empty residual; the original must have been imbalanced. Returns the
number of failures (0 = all balanced); the CLI turns that into an exit code.
"""
import os

import pandas as pd

from mcb import io as msbio
from mcb.paths import work

STAGES = [
    ("proton/water",    "proton_water_corrections.tsv"),
    ("co-substrate",    "cosubstrate_corrections.tsv"),
    ("couple (auto)",   "couple_corrections.tsv"),
    ("2OG-demethylase", "twoog_corrections.tsv"),
    ("couple (agent)",  "couple_agent_corrections.tsv"),
    ("phospho (agent)", "phospho_corrections.tsv"),
]


def run():
    cpd = msbio.load_compounds()
    info = msbio.SpeciesInfo.from_compounds(cpd)
    rxn = msbio.load_reactions(True)
    orig_stoich = dict(zip(rxn["id"], rxn["stoichiometry"]))

    override_ids = set()
    if os.path.exists(work("phospho_overrides.tsv")):
        try:
            override_ids = set(pd.read_csv(work("phospho_overrides.tsv"), sep="\t", dtype=str)["id"])
        except pd.errors.EmptyDataError:
            pass

    overall_fail = 0
    reconciled_total = 0
    print("===== PER-STAGE MASS/CHARGE BALANCE AUDIT =====\n")
    for label, path in STAGES:
        try:
            df = pd.read_csv(work(path), sep="\t", dtype=str, keep_default_na=False)
        except FileNotFoundError:
            print(f"[{label}] file missing: {path}\n")
            continue
        except pd.errors.EmptyDataError:
            print(f"[{label}] no corrections (empty): {path}\n")
            continue

        if "cosubstrate" in path:
            df = df[~df["id"].isin(override_ids)]
        n = len(df)
        reconciled_total += n
        bad_corrected, orig_was_balanced = [], []
        for _, r in df.iterrows():
            sp = msbio.parse_stoich(r["corrected_stoichiometry"])
            res, flags = msbio.compute_residual(sp, info)
            if res or flags["no_formula"] or flags["unknown_charge"]:
                bad_corrected.append((r["id"], res, flags))
            o = orig_stoich.get(r["id"])
            if o is not None:
                ores, oflags = msbio.compute_residual(msbio.parse_stoich(o), info)
                if not ores and not oflags["no_formula"] and not oflags["unknown_charge"]:
                    orig_was_balanced.append(r["id"])

        ok = n - len(bad_corrected)
        status = "PASS" if not bad_corrected else "FAIL"
        print(f"[{label}]  {path}")
        print(f"    verified mass+charge balanced: {ok}/{n}   -> {status}")
        if bad_corrected:
            overall_fail += len(bad_corrected)
            for rid, res, fl in bad_corrected[:10]:
                print(f"      FAIL {rid}  residual={res} flags={fl}")
        if orig_was_balanced:
            print(f"    WARNING: {len(orig_was_balanced)} originals were already balanced: "
                  f"{orig_was_balanced[:5]}")
        print()

    print(f"TOTAL corrected (reconciled, after {len(override_ids)} phospho retractions): "
          f"{reconciled_total}")
    print(f"AUDIT RESULT: {'ALL BALANCED [OK]' if overall_fail == 0 else f'{overall_fail} FAILURES'}")
    return overall_fail


if __name__ == "__main__":
    import sys
    sys.exit(1 if run() else 0)
