#!/usr/bin/env python
"""
Stage 6 - 2-oxoglutarate/O2-dependent dioxygenase O-DEMETHYLATION corrector (EC 1.14.11.x).

These enzymes strip an O-methyl group as formaldehyde:
    R-OCH3 + 2-oxoglutarate + O2  ->  R-OH + formaldehyde + succinate + CO2
ModelSEED often stores only "R-OH <=> R-OCH3" (a bare CH2 difference), which a naive CH2
balancer mistakes for a SAM methylation -- the failure KEGG caught. Here we detect the class
by EC (an oxidoreductase, NOT a transferase) and emit the correct dioxygenase reaction from
scratch in the demethylation direction, then balance-verify. This stage has priority over the
couple-agent stage in build_log, so it overrides any methyl mislabel on the same reaction.

Output (work dir): twoog_corrections.tsv
"""
import pandas as pd

from mcb import io as msbio
from mcb.paths import wp

TWOOG = "cpd00024"   # 2-oxoglutarate
O2 = "cpd00007"
SUCC = "cpd00036"    # succinate
CO2 = "cpd00011"
HCHO = "cpd00055"    # formaldehyde
COFACTORS = {TWOOG, O2, SUCC, CO2, HCHO, "cpd00001", "cpd00067"}


def formula_diff(fa, fb):
    keys = set(fa) | set(fb)
    return {k: fa.get(k, 0) - fb.get(k, 0) for k in keys if fa.get(k, 0) - fb.get(k, 0)}


def main():
    cpd = msbio.load_compounds()
    rxn = msbio.load_reactions(True)
    info = msbio.SpeciesInfo.from_compounds(cpd)
    name_of = dict(zip(cpd["id"], cpd["name"]))

    rows = []
    skipped = 0
    for _, r in rxn.iterrows():
        ec = r["ec_numbers"]
        if ec in ("", "null") or not any(e.strip().startswith("1.14.11") for e in ec.split("|")):
            continue
        sp = msbio.parse_stoich(r["stoichiometry"])
        res0, fl0 = msbio.compute_residual(sp, info)
        if not res0 or fl0["no_formula"] or fl0["unknown_charge"]:   # only fix imbalanced
            skipped += 1
            continue
        organic = [s for s in sp if s["cpd"] not in COFACTORS
                   and (info.formula(s["cpd"]) or {}).get("C", 0) > 0]
        if len(organic) != 2:
            skipped += 1
            continue
        fa, fb = info.formula(organic[0]["cpd"]), info.formula(organic[1]["cpd"])
        if not fa or not fb:
            continue
        diff = formula_diff(fa, fb)
        if diff not in ({"C": 1, "H": 2}, {"C": -1, "H": -2}):   # differ by exactly one CH2
            continue
        if diff == {"C": 1, "H": 2}:
            methylated, demethylated = organic[0]["cpd"], organic[1]["cpd"]
        else:
            methylated, demethylated = organic[1]["cpd"], organic[0]["cpd"]

        cmpt = organic[0]["cmpt"]

        def S(coeff, c):
            return {"coeff": coeff, "cpd": c, "cmpt": cmpt, "comm": "", "name": name_of.get(c, c)}
        new_sp = [S(-1, methylated), S(-1, TWOOG), S(-1, O2),
                  S(+1, demethylated), S(+1, HCHO), S(+1, SUCC), S(+1, CO2)]
        res, fl = msbio.compute_residual(new_sp, info)
        if res or fl["no_formula"] or fl["unknown_charge"] or not msbio.is_valid_reaction(new_sp):
            skipped += 1
            continue
        rows.append({
            "id": r["id"], "orig_status": r["status"], "couple": "2OG-dioxygenase-demethylation",
            "confidence": "high", "ec": ec,
            "patch": "rebuilt: R-OCH3 + 2OG + O2 -> R-OH + formaldehyde + succinate + CO2",
            "corrected_stoichiometry": msbio.serialize_stoich(new_sp),
        })

    pd.DataFrame(rows).to_csv(wp("twoog_corrections.tsv"), sep="\t", index=False)
    print(f"2-OG dioxygenase demethylations corrected & balance-verified: {len(rows)}")
    print(f"EC 1.14.11 reactions skipped (not a 2-species CH2 demethylation): {skipped}")


if __name__ == "__main__":
    main()
