#!/usr/bin/env python
"""
Menu-expansion candidates: reactions closeable by a SINGLE cofactor (or couple) not in
our stocked menu. Coverage audit found ~2,251 such, but 2,155 are the H2 redox trap
(EXCLUDED here). This enumerates the *plausible* remainder with EC + name so we can
triage which cofactors are genuinely worth adding (vs. arithmetic false positives like
glutamate-needs-ATP or acetate-from-a-hydrolase-that-needs-water).

Each candidate is EC-gated where the cofactor has a natural enzyme class, and results are
grouped by cofactor so each class's real-vs-false rate is visible.
Output: menu_expansion_candidates.tsv  (read-only).
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from mcb.paths import work, result, inp, figure
from collections import Counter, defaultdict
import pandas as pd
import msbio
from mcb.closures import proton_water_close, residual_after
from mcb.closures import add, pw_close

# extra SINGLE cofactors (cpd -> (name, EC-prefixes that make it plausible; () = any))
EXTRA_SINGLE = {
    "cpd00047": ("Formate",   ("2.1.2", "6.3.4", "1.2.1", "3.5.1")),
    "cpd00023": ("Glutamate", ("6.3.2",)),           # ligases (but usually need ATP too)
    "cpd00029": ("Acetate",   ("3.1.1", "3.5.1")),   # esterases/amidohydrolases (need water)
    "cpd00073": ("Urea",      ("3.5.3", "3.5.1")),   # ureohydrolases
    "cpd00239": ("H2S",       ("4.4.1", "2.5.1")),   # lyases
    "cpd00075": ("Nitrite",   ("1.7", "1.9")),
    "cpd00033": ("Glycine",   ("6.3.2", "2.1.2")),
    "cpd00041": ("Aspartate", ("6.3", "2.6.1")),
    "cpd00100": ("Glycerol",  ()),
}
# extra COUPLES (name -> (donor, acceptor, EC-prefixes))
EXTRA_COUPLES = {
    "succinyl (SucCoA/CoA)":     ("cpd00078", "cpd00010", ("2.3.1",)),
    "glucuronosyl (UDPGlcA/UDP)": ("cpd00035", "cpd00014", ("2.4.1",)),
    "glutamyl+ATP (Glu/-,ATP/ADP)": None,  # placeholder: needs 2-species, handled elsewhere
}


def single_closes(res0, formula, charge):
    for k in (-3, -2, -1, 1, 2, 3):
        if proton_water_close(residual_after(res0, formula, charge, k)) is not None:
            return k
    return None


def couple_closes(res0, donor, acc, info):
    df_ = info.formula(donor); dq_ = info.charge(donor)
    af_ = info.formula(acc); aq_ = info.charge(acc)
    if None in (df_, dq_, af_, aq_):
        return False
    for k in (1, -1):
        if pw_close(add(add(res0, af_, aq_, k), df_, dq_, -k)) is not None:
            return True
    return False


def ec_hit(ec, prefixes):
    if not prefixes:
        return "any"
    if ec in ("", "null"):
        return "no-ec"
    return "yes" if any(e.strip().startswith(p) for e in ec.split("|") for p in prefixes) else "no"


def main():
    cpd = msbio.load_compounds(); rxn = msbio.load_reactions(True)
    info = msbio.SpeciesInfo.from_compounds(cpd)
    corrected = set(pd.read_csv(result("corrections_log.tsv"), sep="\t", dtype=str)["reaction_id"])

    rows = []
    for _, r in rxn.iterrows():
        if r["id"] in corrected:
            continue
        sp = msbio.parse_stoich(r["stoichiometry"])
        res0, fl = msbio.compute_residual(sp, info)
        if fl["no_formula"] or fl["unknown_charge"] or not res0:
            continue
        # skip if H2 closes it (redox trap) -> not a real menu gap
        h2 = info.formula("cpd11640")
        if h2 and single_closes(res0, h2, 0) is not None:
            continue
        ec = r["ec_numbers"]
        for c, (nm, pref) in EXTRA_SINGLE.items():
            f = info.formula(c); q = info.charge(c)
            if f is None or q is None:
                continue
            k = single_closes(res0, f, q)
            if k is not None:
                rows.append({"reaction_id": r["id"], "cofactor": f"single:{nm}",
                             "ec_hit": ec_hit(ec, pref), "ec": ec,
                             "name": r["name"][:45], "definition": r["definition"][:90]})
                break
        else:
            for nm, spec in EXTRA_COUPLES.items():
                if spec is None:
                    continue
                d, a, pref = spec
                if couple_closes(res0, d, a, info):
                    rows.append({"reaction_id": r["id"], "cofactor": f"couple:{nm}",
                                 "ec_hit": ec_hit(ec, pref), "ec": ec,
                                 "name": r["name"][:45], "definition": r["definition"][:90]})
                    break

    df = pd.DataFrame(rows)
    df.to_csv(work("menu_expansion_candidates.tsv"), sep="\t", index=False)
    print(f"plausible menu-expansion candidates (H2 trap excluded): {len(df)}\n")
    print("by cofactor x EC-hit (yes = EC corroborates -> real; no = likely false):")
    print(pd.crosstab(df["cofactor"], df["ec_hit"]).to_string())
    print("\nEC-corroborated candidates (the ones worth acting on):")
    for _, r in df[df["ec_hit"] == "yes"].iterrows():
        print(f"  {r['reaction_id']} {r['cofactor']:22s} EC={r['ec'][:16]:16s} {r['definition']}")


if __name__ == "__main__":
    main()
