#!/usr/bin/env python
"""
2-oxoglutarate/O2-dependent dioxygenase O-DEMETHYLATION corrector (EC 1.14.11.x).

These enzymes strip an O-methyl group as formaldehyde:
    R-OCH3 + 2-oxoglutarate + O2  ->  R-OH + formaldehyde + succinate + CO2
ModelSEED often stores only "R-OH <=> R-OCH3" (a bare CH2 difference, frequently written
in the reverse/methylation-looking direction and missing all four cofactors). A naive
CH2 balancer mistakes this for a SAM methylation -- the failure that KEGG caught. Here we
detect the class by EC (an oxidoreductase, NOT a transferase) and emit the correct
dioxygenase reaction FROM SCRATCH in the demethylation direction, then balance-verify.

Detection: EC contains 1.14.11.- and the reaction has exactly two organic species
differing by exactly CH2 (the O-methyl vs O-H pair).

Output: twoog_corrections.tsv  (schema compatible with build_correction_log)
Reads DB read-only.
"""
import pandas as pd
import msbio

TWOOG = "cpd00024"   # 2-oxoglutarate  C5H4O5 -2
O2 = "cpd00007"      # O2
SUCC = "cpd00036"    # succinate C4H4O4 -2
CO2 = "cpd00011"     # CO2
HCHO = "cpd00055"    # formaldehyde CH2O
COFACTORS = {TWOOG, O2, SUCC, CO2, HCHO, "cpd00001", "cpd00067"}


def formula_diff(fa, fb):
    keys = set(fa) | set(fb)
    return {k: fa.get(k, 0) - fb.get(k, 0) for k in keys if fa.get(k, 0) - fb.get(k, 0)}


def main():
    cpd = msbio.load_compounds(); rxn = msbio.load_reactions(True)
    info = msbio.SpeciesInfo.from_compounds(cpd)
    name_of = dict(zip(cpd["id"], cpd["name"]))

    rows = []
    skipped = 0
    for _, r in rxn.iterrows():
        ec = r["ec_numbers"]
        if ec in ("", "null") or not any(e.strip().startswith("1.14.11")
                                         for e in ec.split("|")):
            continue
        sp = msbio.parse_stoich(r["stoichiometry"])
        # only fix reactions that are actually imbalanced
        res0, fl0 = msbio.compute_residual(sp, info)
        if not res0 or fl0["no_formula"] or fl0["unknown_charge"]:
            skipped += 1
            continue
        organic = [s for s in sp if s["cpd"] not in COFACTORS
                   and (info.formula(s["cpd"]) or {}).get("C", 0) > 0]
        if len(organic) != 2:
            skipped += 1
            continue
        fa = info.formula(organic[0]["cpd"]); fb = info.formula(organic[1]["cpd"])
        if not fa or not fb:
            continue
        diff = formula_diff(fa, fb)  # organic[0] - organic[1]
        # must differ by exactly one CH2 (the O-methyl)
        if diff not in ({"C": 1, "H": 2}, {"C": -1, "H": -2}):
            continue
        # methylated = more carbon
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

    pd.DataFrame(rows).to_csv("twoog_corrections.tsv", sep="\t", index=False)
    print(f"2-OG dioxygenase demethylations corrected & balance-verified: {len(rows)}")
    print(f"EC 1.14.11 reactions skipped (not a 2-species CH2 demethylation): {skipped}")
    for r in rows[:12]:
        print(f"  {r['id']} EC={r['ec']}")


if __name__ == "__main__":
    main()
