#!/usr/bin/env python
"""
Co-substrate closure: deterministic generalization of the proton/water rule.

Curators routinely omit a *small* vocabulary of "free" small molecules. If a
reaction's residual is closed EXACTLY by adding at most one such heavy co-substrate
(plus the free proton/water pair), that is the same physically-meaningful signal as
a missing water -- the reaction is simply "off by one CO2 / one phosphate / ...".

Precision guard (per user: correct only with a meaningful check):
  * at most ONE heavy co-substrate type per reaction (|k| <= 3),
  * remaining {H,O,charge} must then close by the deterministic proton/water rule,
  * every accepted patch re-verified to mass+charge = 0 via msbio.compute_residual,
  * results SEGREGATED by reagent so each species class can be trusted/rejected
    independently (CO2-missing is far more trustworthy than a bare charge shuffle).

Reads ModelSEEDDatabase read-only; writes proposals into this folder only.
"""
from collections import Counter, defaultdict

import pandas as pd

import msbio

# Commonly-omitted "free" heavy co-substrates (cpd id -> (name, formula dict, charge)).
# pH-7 microspecies as stored in ModelSEED.
HEAVY = {
    "cpd00011": ("CO2",     {"C": 1, "O": 2},          0),
    "cpd00009": ("Pi",      {"H": 1, "O": 4, "P": 1}, -2),
    "cpd00012": ("PPi",     {"H": 1, "O": 7, "P": 2}, -3),
    "cpd00013": ("NH4",     {"H": 4, "N": 1},          1),
    "cpd00048": ("Sulfate", {"O": 4, "S": 1},         -2),
}

# EC top-classes whose enzyme genuinely RELEASES/CONSUMES the FREE co-substrate, so a
# single-species closure is biochemically corroborated (not just arithmetically balanced).
# Deliberately EXCLUDES ATP-dependent kinases (EC 2.7.1/2.7.4): they transfer phosphate
# FROM ATP, so a "missing Pi" there is really a missing ATP->ADP pair -> agent phase.
EC_CORROBORATION = {
    "CO2":     ("4.1.1", "6.4.1"),          # carboxy-lyases / biotin carboxylases
    "Pi":      ("3.1.3", "3.6.1", "3.1.4"),  # phosphomonoesterases / anhydride hydrolases
    "PPi":     ("6.", "2.7.7", "3.6.1"),     # ligases / nucleotidyltransferases (release PPi)
    "NH4":     ("3.5.1", "3.5.4", "4.3.1"),  # amidohydrolases / deaminases / ammonia-lyases
    "Sulfate": ("2.8.2", "3.1.6"),           # sulfotransferases / sulfatases
}
WATER = ({"H": 2, "O": 1}, 0)
PROTON = ({"H": 1}, 1)


def proton_water_close(res):
    """Return (w, p) closing residual res in {H,O,charge}, or None. res is a dict."""
    heavy = {e for e in res if e not in ("H", "O", "charge")}
    if heavy:
        return None
    dH = res.get("H", 0.0); dO = res.get("O", 0.0); dq = res.get("charge", 0.0)
    if any(abs(x - round(x)) > 1e-6 for x in (dH, dO, dq)):
        return None
    w = -round(dO); p = -round(dq)
    if abs(dH + 2 * w + p) > 1e-6:
        return None
    return w, p


def residual_after(res, formula, charge, k):
    """residual after adding k copies (product side) of a species."""
    out = dict(res)
    for el, n in formula.items():
        out[el] = out.get(el, 0) + k * n
    out["charge"] = out.get("charge", 0) + k * charge
    return {e: v for e, v in out.items() if abs(v) > 1e-6}


def main():
    cpd = msbio.load_compounds()
    rxn = msbio.load_reactions(True)
    info = msbio.SpeciesInfo.from_compounds(cpd)
    name = dict(zip(cpd["id"], cpd["name"]))

    rows = []
    by_reagent = Counter()

    for _, r in rxn.iterrows():
        sp = msbio.parse_stoich(r["stoichiometry"])
        res0, fl = msbio.compute_residual(sp, info)
        if fl["no_formula"] or fl["unknown_charge"] or not res0:
            continue
        # skip pure proton/water-closable (already handled by proton_water_balance)
        if proton_water_close(res0) is not None:
            continue

        best = None
        for cid, (nm, formula, charge) in HEAVY.items():
            for k in (-3, -2, -1, 1, 2, 3):
                res1 = residual_after(res0, formula, charge, k)
                wp = proton_water_close(res1)
                if wp is None:
                    continue
                w, p = wp
                # total added molecules (prefer the most parsimonious)
                cost = abs(k) + abs(w) + abs(p)
                if best is None or cost < best[0]:
                    best = (cost, cid, nm, k, w, p)
        if best is None:
            continue

        _, cid, nm, k, w, p = best
        # build patch: k heavy (product side if k>0), w water, p proton
        patch = defaultdict(int)
        patch[cid] += k
        if w: patch["cpd00001"] += w
        if p: patch["cpd00067"] += p

        new_sp = [dict(s) for s in sp]
        cmpt = Counter(s["cmpt"] for s in sp).most_common(1)[0][0]
        for pc, pk in patch.items():
            found = False
            for s in new_sp:
                if s["cpd"] == pc and s["cmpt"] == cmpt:
                    s["coeff"] += pk; found = True; break
            if not found:
                nm2 = {"cpd00001": "H2O", "cpd00067": "H+"}.get(pc, HEAVY.get(pc, (pc,))[0])
                new_sp.append({"coeff": pk, "cpd": pc, "cmpt": cmpt, "comm": "", "name": nm2})
        new_sp = [s for s in new_sp if abs(s["coeff"]) > 1e-9]
        res_final, _ = msbio.compute_residual(new_sp, info)
        if res_final or not msbio.is_valid_reaction(new_sp):   # verify to zero + non-degenerate
            continue

        by_reagent[nm] += 1
        desc = []
        for pc, pk in patch.items():
            lbl = {"cpd00001": "H2O", "cpd00067": "H+"}.get(pc, nm)
            if pk:
                desc.append(f"{abs(pk)} {lbl} -> {'products' if pk>0 else 'reactants'}")
        # EC corroboration -> confidence
        ec = r["ec_numbers"]
        corroborated = ec not in ("", "null") and any(
            e.strip().startswith(pref) for e in ec.split("|")
            for pref in EC_CORROBORATION.get(nm, ()))
        # single free-species helper (only water/proton besides the heavy) is cleaner
        n_heavy_only = (patch.get("cpd00001", 0) == 0 and patch.get("cpd00067", 0) == 0)
        confidence = "high" if corroborated else ("review" if ec in ("", "null") else "low")
        rows.append({
            "id": r["id"], "orig_status": r["status"],
            "residual": ",".join(f"{k2}:{int(v):+d}" for k2, v in sorted(res0.items())),
            "co_substrate": nm, "patch": "; ".join(desc),
            "confidence": confidence, "ec": ec, "definition": r["definition"][:100],
            "corrected_stoichiometry": msbio.serialize_stoich(new_sp),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(["confidence", "co_substrate"])
    df.to_csv("cosubstrate_corrections.tsv", sep="\t", index=False)
    print(f"co-substrate single-species closures found & verified: {len(df)}")
    print("\nby confidence (EC corroborates the missing species):")
    for k, v in df["confidence"].value_counts().items():
        print(f"  {k:8s} {v}")
    print("\nby co-substrate x confidence:")
    print(pd.crosstab(df["co_substrate"], df["confidence"]))
    print("\nHIGH-confidence examples (EC predicts the co-substrate):")
    for _, e in df[df["confidence"] == "high"].head(10).iterrows():
        print(f"  {e['id']} {e['co_substrate']:8s} res={e['residual']:22s} EC={e['ec'][:18]:18s} {e['definition'][:40]}")
    print("\nwrote cosubstrate_corrections.tsv")


if __name__ == "__main__":
    main()
