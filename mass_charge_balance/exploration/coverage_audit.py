#!/usr/bin/env python
"""
Coverage audit: of every imbalanced-but-scoreable reaction, which are closeable by our
machinery, and did we actually correct them? Distinguishes:
  - MISSED: closeable by proton/water but NOT in corrections_log  -> a real bug (should be 0)
  - skipped-by-design: cosubstrate/couple-closeable but agent SKIP / retracted (intentional)
  - not-in-menu: closeable by a SINGLE cofactor we don't stock (a coverage-expansion gap)
  - not-closeable: redox/skeleton/etc. (out of scope)
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import Counter, defaultdict
import pandas as pd
import msbio
from mcb.closures import HEAVY, proton_water_close, residual_after
from mcb.closures import COUPLES, ec_compatible, add, pw_close

# Extra single-species cofactors NOT in our co-substrate menu, to size the coverage gap.
EXTRA_SINGLE = {
    "cpd00242": "H2CO3", "cpd00047": "Formate", "cpd00100": "Glycerol",
    "cpd00033": "Glycine", "cpd00029": "Acetate", "cpd00041": "Aspartate",
    "cpd00023": "Glutamate", "cpd11640": "H2", "cpd00239": "H2S",
    "cpd00075": "Nitrite", "cpd00528": "N2", "cpd00073": "Urea",
}
# Extra couples not stocked (donor, acceptor)
EXTRA_COUPLES = {
    "succinyl (SucCoA/CoA)": ("cpd00078", "cpd00010"),
    "glucuronosyl (UDPGlcA/UDP)": ("cpd00035", "cpd00014"),
    "dTDP-glucosyl": ("cpd00072", "cpd00014"),
    "glutathionyl (GSH/GSSG)": ("cpd00042", "cpd00111"),
}


def single_closes(res0, formula, charge, info):
    for k in (-3, -2, -1, 1, 2, 3):
        if proton_water_close(residual_after(res0, formula, charge, k)) is not None:
            return True
    return False


def couple_closes(res0, donor, acc, info):
    df_ = info.formula(donor); dq_ = info.charge(donor)
    af_ = info.formula(acc); aq_ = info.charge(acc)
    if None in (df_, dq_, af_, aq_):
        return False
    for k in (1, -1):
        if pw_close(add(add(res0, af_, aq_, k), df_, dq_, -k)) is not None:
            return True
    return False


def main():
    cpd = msbio.load_compounds(); rxn = msbio.load_reactions(True)
    info = msbio.SpeciesInfo.from_compounds(cpd)
    log = pd.read_csv("corrections_log.tsv", sep="\t", dtype=str, keep_default_na=False)
    corrected = set(log["reaction_id"])

    cls = Counter()
    missed_pw = []
    not_in_menu = Counter()
    gap_examples = defaultdict(list)

    for _, r in rxn.iterrows():
        sp = msbio.parse_stoich(r["stoichiometry"])
        res0, fl = msbio.compute_residual(sp, info)
        if fl["no_formula"] or fl["unknown_charge"]:
            cls["unscoreable(no formula/charge)"] += 1
            continue
        if not res0:
            cls["already balanced"] += 1
            continue
        # imbalanced & scoreable:
        is_corr = r["id"] in corrected
        if proton_water_close(res0) is not None:
            cls["closeable: proton/water"] += 1
            if not is_corr:
                missed_pw.append(r["id"])
            continue
        # single stocked co-substrate?
        if any(single_closes(res0, f, q, info) for _, f, q in HEAVY.values()):
            cls["closeable: co-substrate(stocked)"] += 1
            continue
        # stocked couple (EC-compatible)?
        if any(ec_compatible(r["ec_numbers"], cn) and
               couple_closes(res0, COUPLES[cn][0], COUPLES[cn][1], info) for cn in COUPLES):
            cls["closeable: couple(stocked)"] += 1
            continue
        # ---- would an UNSTOCKED cofactor close it? (coverage-expansion gap) ----
        hit = None
        for c, nm in EXTRA_SINGLE.items():
            f = info.formula(c); q = info.charge(c)
            if f and q is not None and single_closes(res0, f, q, info):
                hit = f"single:{nm}"; break
        if not hit:
            for nm, (d, a) in EXTRA_COUPLES.items():
                if couple_closes(res0, d, a, info):
                    hit = f"couple:{nm}"; break
        if hit:
            cls["closeable: NOT in our menu"] += 1
            not_in_menu[hit] += 1
            if len(gap_examples[hit]) < 3:
                gap_examples[hit].append((r["id"], r["name"][:40], r["ec_numbers"]))
            continue
        cls["not closeable (redox/skeleton/other)"] += 1

    print("===== COVERAGE AUDIT (active reactions) =====")
    for k, v in cls.most_common():
        print(f"  {k:38s} {v}")
    print(f"\n  >>> proton/water-closeable but NOT corrected (BUG if >0): {len(missed_pw)}")
    if missed_pw:
        print("      ", missed_pw[:15])
    print(f"\n  >>> closeable by a cofactor NOT in our menu (expansion opportunity):")
    for k, v in not_in_menu.most_common():
        print(f"        {k:28s} {v}")
        for ex in gap_examples[k]:
            print(f"            {ex[0]} {ex[1]:40s} EC={ex[2]}")


if __name__ == "__main__":
    main()
