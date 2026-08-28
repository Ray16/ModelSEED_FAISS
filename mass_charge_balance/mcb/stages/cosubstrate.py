#!/usr/bin/env python
"""
Stage 2 - co-substrate closure: deterministic generalization of the proton/water rule.

Curators routinely omit a small vocabulary of "free" heavy molecules (CO2, Pi, PPi, NH4,
sulfate). If a residual is closed exactly by adding at most one such species (|k| <= 3) plus
the deterministic proton/water pair, that is the same physically-meaningful signal as a
missing water. Results are segregated by reagent so each class can be trusted/rejected on
its own, and EC corroboration (the enzyme releases the free species) sets confidence.

Guards (mcb.closures):
  * refuse reactions with an indefinite-polymer species (placeholder formula),
  * reject a phosphate co-substrate that contradicts the nucleotide couple present
    (ATP->ADP releases Pi, not PPi; ATP->AMP releases PPi, not Pi).

Reads ModelSEEDDatabase read-only; writes proposals into the work directory only.
"""
from collections import Counter, defaultdict

import pandas as pd

from mcb import io as msbio
from mcb import closures as cl
from mcb.paths import work


def main():
    cpd = msbio.load_compounds()
    rxn = msbio.load_reactions(True)
    info = msbio.SpeciesInfo.from_compounds(cpd)

    rows = []
    by_reagent = Counter()

    for _, r in rxn.iterrows():
        sp = msbio.parse_stoich(r["stoichiometry"])
        if cl.is_refused_reaction(sp):                       # indefinite-polymer guard
            continue
        res0, fl = msbio.compute_residual(sp, info)
        if fl["no_formula"] or fl["unknown_charge"] or not res0:
            continue
        # skip pure proton/water-closable (already handled by stage 1)
        if cl.proton_water_close(res0) is not None:
            continue

        reaction_cpds = {s["cpd"] for s in sp}
        best = None
        for cid, (nm, formula, charge) in cl.HEAVY.items():
            # phosphate-mechanism guard: don't add Pi/PPi that contradicts the ATP couple
            if cl.phosphate_cosubstrate_conflict(reaction_cpds, cid):
                continue
            for k in (-3, -2, -1, 1, 2, 3):
                res1 = cl.add_species(res0, formula, charge, k)
                wp_ = cl.proton_water_close(res1)
                if wp_ is None:
                    continue
                w, p = wp_
                cost = abs(k) + abs(w) + abs(p)   # prefer the most parsimonious patch
                if best is None or cost < best[0]:
                    best = (cost, cid, nm, k, w, p)
        if best is None:
            continue

        _, cid, nm, k, w, p = best
        patch = defaultdict(int)
        patch[cid] += k
        if w:
            patch["cpd00001"] += w
        if p:
            patch["cpd00067"] += p

        new_sp = [dict(s) for s in sp]
        cmpt = Counter(s["cmpt"] for s in sp).most_common(1)[0][0]
        for pc, pk in patch.items():
            found = False
            for s in new_sp:
                if s["cpd"] == pc and s["cmpt"] == cmpt:
                    s["coeff"] += pk
                    found = True
                    break
            if not found:
                nm2 = {"cpd00001": "H2O", "cpd00067": "H+"}.get(pc, cl.HEAVY.get(pc, (pc,))[0])
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
                desc.append(f"{abs(pk)} {lbl} -> {'products' if pk > 0 else 'reactants'}")
        ec = r["ec_numbers"]
        corroborated = ec not in ("", "null") and any(
            e.strip().startswith(pref) for e in ec.split("|")
            for pref in cl.EC_CORROBORATION.get(nm, ()))
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
    df.to_csv(work("cosubstrate_corrections.tsv"), sep="\t", index=False)
    print(f"co-substrate single-species closures found & verified: {len(df)}")
    if len(df):
        print("\nby co-substrate x confidence:")
        print(pd.crosstab(df["co_substrate"], df["confidence"]))


if __name__ == "__main__":
    main()
