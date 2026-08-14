#!/usr/bin/env python
"""
Proton/water balancer: the deterministic, highest-confidence correction class.

The only two "free" reagents a reaction can gain/lose without changing its
chemistry are water and protons. They span exactly three balance axes:

    reagent   H    O    charge
    H+        +1    0     +1
    H2O       +2   +1      0

Given a reaction's residual (products - reactants) in H, O and charge, the fix is
FULLY DETERMINED (no search):

    waters  to close O:      w = -residual[O]
    protons to close charge: p = -residual[charge]
    then H must independently close: residual[H] + 2*w + p == 0   (consistency)

A reaction is correctable here iff:
  * every species has a real formula and known charge (else -> needs-structure /
    needs-charge, handled elsewhere),
  * the residual involves only H, O, charge (any other heavy element -> skeleton
    error, handled elsewhere), and
  * the H consistency check holds.

A pure charge imbalance (only 'charge' off, H and O balanced) is intentionally NOT
handled here: adding a proton to fix charge injects an H and breaks mass. Those are
compound-level protonation defects handled by the charge-reassignment corrector.

INPUT : ModelSEEDDatabase (read-only -- never modified).
OUTPUT (all written into THIS folder, originals untouched):
    proton_water_corrections.tsv  -- proposed patch + corrected reaction, per fixable rxn
    proton_water_report.txt       -- summary counts, deferral reasons, examples
"""
import os
from collections import Counter

import pandas as pd

import msbio

OUT = os.path.dirname(os.path.abspath(__file__))

REAGENT = {  # cpd id -> (name, contribution incl 'charge')
    msbio.CPD_PROTON: ("H+",  {"H": 1, "charge": 1}),
    msbio.CPD_WATER:  ("H2O", {"H": 2, "O": 1, "charge": 0}),
}


def add_reagent(species, cpd, net_coeff, cmpt="0"):
    name = REAGENT[cpd][0]
    for sp in species:
        if sp["cpd"] == cpd and sp["cmpt"] == cmpt:
            sp["coeff"] += net_coeff
            return species
    species.append({"coeff": net_coeff, "cpd": cpd, "cmpt": cmpt, "comm": "", "name": name})
    return species


def solve(residual):
    """Return (patch, reason). patch = {'H2O': w, 'H+': p} net product-side coeffs."""
    heavy = {el for el in residual if el not in ("H", "O", "charge")}
    if heavy:
        return None, f"skeleton_error:{','.join(sorted(heavy))}"
    dH = residual.get("H", 0.0)
    dO = residual.get("O", 0.0)
    dq = residual.get("charge", 0.0)
    if any(abs(x - round(x)) > 1e-6 for x in (dH, dO, dq)):
        return None, "non_integer_residual"
    w = -round(dO)
    p = -round(dq)
    if abs(dH + 2 * w + p) > 1e-6:
        # Distinguish the diagnostic sub-cases for reporting.
        if set(residual) == {"charge"}:
            return None, "charge_only_protonation_defect"
        if set(residual) == {"O"}:
            return None, "oxygen_only_missing_species"
        return None, "H_O_charge_inconsistent"
    if w == 0 and p == 0:
        return None, "no_change_needed"
    return {"H2O": w, "H+": p}, "ok"


def apply_patch(species, patch):
    species = [dict(sp) for sp in species]
    cmpt = Counter(sp["cmpt"] for sp in species).most_common(1)[0][0]
    if patch["H2O"]:
        add_reagent(species, msbio.CPD_WATER, patch["H2O"], cmpt)
    if patch["H+"]:
        add_reagent(species, msbio.CPD_PROTON, patch["H+"], cmpt)
    return [sp for sp in species if abs(sp["coeff"]) > 1e-9]


def describe(patch):
    bits = []
    for reagent in ("H2O", "H+"):
        n = patch[reagent]
        if n:
            bits.append(f"{abs(n)} {reagent} -> {'products' if n > 0 else 'reactants'}")
    return "; ".join(bits)


def render_equation(species):
    """Human-readable name equation for review."""
    react = [sp for sp in species if sp["coeff"] < 0]
    prod = [sp for sp in species if sp["coeff"] > 0]
    def side(items):
        return " + ".join(f'({msbio.format_coeff(abs(sp["coeff"]))}) {sp["name"] or sp["cpd"]}'
                          for sp in items)
    return f"{side(react)} <=> {side(prod)}"


def main():
    print("Loading ModelSEED biochemistry (read-only) ...")
    cpd = msbio.load_compounds()
    rxn = msbio.load_reactions(active_only=True)
    info = msbio.SpeciesInfo.from_compounds(cpd)
    print(f"  {len(rxn)} active reactions")

    rows = []
    reasons = Counter()
    verified_ok = verify_fail = 0

    for _, r in rxn.iterrows():
        species = msbio.parse_stoich(r["stoichiometry"])
        residual, flags = msbio.compute_residual(species, info)

        if flags["no_formula"]:
            reasons["defer_needs_structure"] += 1
            continue
        if flags["unknown_charge"]:
            reasons["defer_needs_charge"] += 1
            continue
        if not residual:
            reasons["already_balanced"] += 1
            continue

        patch, why = solve(residual)
        if patch is None:
            reasons[f"defer_{why}"] += 1
            continue

        new_species = apply_patch(species, patch)
        new_residual, _ = msbio.compute_residual(new_species, info)
        if not msbio.is_valid_reaction(new_species):
            reasons["defer_degenerate_result"] += 1
            continue
        ok = not new_residual
        verified_ok += ok
        verify_fail += (not ok)
        reasons["CORRECTED"] += 1

        rows.append({
            "id": r["id"],
            "orig_status": r["status"],
            "residual": ";".join(f"{k}:{int(v)}" for k, v in sorted(residual.items())),
            "patch": describe(patch),
            "verified": "OK" if ok else "FAIL",
            "corrected_equation": render_equation(new_species),
            "orig_stoichiometry": r["stoichiometry"],
            "corrected_stoichiometry": msbio.serialize_stoich(new_species),
        })

    df = pd.DataFrame(rows)
    out_tsv = os.path.join(OUT, "proton_water_corrections.tsv")
    df.to_csv(out_tsv, sep="\t", index=False)

    lines = ["===== PROTON/WATER BALANCE — CORRECTION REPORT =====",
             "(reads ModelSEEDDatabase read-only; writes proposals into mass_charge_balance/)\n"]
    for k, v in reasons.most_common():
        lines.append(f"  {k:34s} {v:7d}")
    lines.append(f"\n  Reactions CORRECTED (patch applied): {reasons['CORRECTED']}")
    lines.append(f"  Verified balanced after patch:       {verified_ok}")
    lines.append(f"  Verification FAILURES:               {verify_fail}")
    if len(df):
        lines.append("\n  Patch-type distribution:")
        for k, v in Counter(df["patch"]).most_common(12):
            lines.append(f"    {k:34s} {v}")
        lines.append("\n  Worked examples:")
        for _, ex in df.head(8).iterrows():
            lines.append(f"    {ex['id']}  status={ex['orig_status']}  "
                         f"residual[{ex['residual']}]  ->  {ex['patch']}  [{ex['verified']}]")
    report = "\n".join(lines)
    with open(os.path.join(OUT, "proton_water_report.txt"), "w") as fh:
        fh.write(report + "\n")
    print("\n" + report)
    print(f"\nWrote {out_tsv}")


if __name__ == "__main__":
    main()
