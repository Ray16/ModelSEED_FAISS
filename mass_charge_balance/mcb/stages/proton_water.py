#!/usr/bin/env python
"""
Stage 1 - proton/water balancer: the deterministic, highest-confidence class.

The only two "free" reagents a reaction can gain/lose without changing its chemistry are
water and protons; they span exactly three balance axes (H, O, charge), so given the
residual the fix is fully determined (no search): w = -dO waters, p = -dq protons, then H
must independently close (dH + 2w + p == 0). See mcb.closures.proton_water_close.

Refuses reactions containing an indefinite-polymer species (mcb.closures guard) and
non-H/O/charge (skeleton) residuals -> handled by later stages / left for curation.

Reads ModelSEEDDatabase read-only; writes proposals into the work directory only.
"""
from collections import Counter

import pandas as pd

from mcb import io as msbio
from mcb import closures as cl
from mcb.paths import wp

REAGENT = {
    cl.CPD_PROTON: ("H+",  {"H": 1, "charge": 1}),
    cl.CPD_WATER:  ("H2O", {"H": 2, "O": 1, "charge": 0}),
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
        add_reagent(species, cl.CPD_WATER, patch["H2O"], cmpt)
    if patch["H+"]:
        add_reagent(species, cl.CPD_PROTON, patch["H+"], cmpt)
    return [sp for sp in species if abs(sp["coeff"]) > 1e-9]


def describe(patch):
    bits = []
    for reagent in ("H2O", "H+"):
        n = patch[reagent]
        if n:
            bits.append(f"{abs(n)} {reagent} -> {'products' if n > 0 else 'reactants'}")
    return "; ".join(bits)


def render_equation(species):
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
        if cl.is_refused_reaction(species):           # indefinite-polymer / generic species
            reasons["defer_generic_polymer"] += 1
            continue
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
    df.to_csv(wp("proton_water_corrections.tsv"), sep="\t", index=False)

    lines = ["===== PROTON/WATER BALANCE - CORRECTION REPORT =====",
             "(reads ModelSEEDDatabase read-only; writes proposals into the work dir)\n"]
    for k, v in reasons.most_common():
        lines.append(f"  {k:34s} {v:7d}")
    lines.append(f"\n  Reactions CORRECTED (patch applied): {reasons['CORRECTED']}")
    lines.append(f"  Verified balanced after patch:       {verified_ok}")
    lines.append(f"  Verification FAILURES:               {verify_fail}")
    report = "\n".join(lines)
    with open(wp("proton_water_report.txt"), "w") as fh:
        fh.write(report + "\n")
    print("\n" + report)


if __name__ == "__main__":
    main()
