#!/usr/bin/env python
"""
Cofactor-couple closure: generalize co-substrate closure to a DONOR/ACCEPTOR couple
(the group-transfer partner curators drop, e.g. SAM->SAH for a methylation).

For each imbalanced reaction we try inserting one couple (donor on one side, acceptor
on the other), then close the remainder with the deterministic proton/water rule and
verify mass+charge = 0. EC corroboration gates confidence.

Crucially, several couples can be arithmetically valid for the same residual (any
hexose-NDP donor closes a C6H10O5 deficit). Those AMBIGUOUS reactions are emitted for
agent disambiguation by reaction name; the UNAMBIGUOUS + EC-corroborated ones are
auto-accepted.

Output:
  couple_corrections.tsv         auto-accepted (unambiguous + corroborated), balance-verified
  couple_candidates_for_agent.tsv  reactions with >1 valid couple or no EC -> agent decides
Reads DB read-only; writes into this folder.
"""
import os
from collections import Counter, defaultdict
import pandas as pd
import msbio

# Cofactor couples: name -> (donor_cpd, acceptor_cpd, EC-prefix corroboration tuple)
# donor is the "loaded" carrier (consumed), acceptor is the "unloaded" product.
COUPLES = {
    "methyl (SAM/SAH)":        ("cpd00017", "cpd00019", ("2.1.1",)),
    "glucosyl (UDPglc/UDP)":   ("cpd00026", "cpd00014", ("2.4.1",)),
    "mannosyl (GDPman/GDP)":   ("cpd00083", "cpd00031", ("2.4.1",)),
    "acetyl (AcCoA/CoA)":      ("cpd00022", "cpd00010", ("2.3.1",)),
    "phospho ATP/ADP":         ("cpd00002", "cpd00008", ("2.7.1", "2.7.2", "2.7.4")),
    "phospho GTP/GDP":         ("cpd00038", "cpd00031", ("2.7.1", "2.7.4")),
    "phospho UTP/UDP":         ("cpd00062", "cpd00014", ("2.7.1", "2.7.4")),
    "adenylyl ATP/AMP":        ("cpd00002", "cpd00018", ("2.7.7", "6.")),
}


def couple_ec_classes(cname):
    """EC top-classes where this couple's chemistry lives (derived from its corroboration
    prefixes). Group-transfer couples live in class 2 (transferases); adenylyl also 6."""
    return {p.split(".")[0] for p in COUPLES[cname][2]}


def ec_compatible(ec, cname):
    """A couple may be offered only if the reaction's EC is null OR shares the couple's
    top-class. This PREVENTS offering a transferase couple (class 2) to an oxidoreductase
    (class 1) -- the 2-OG-dioxygenase-as-methylation failure KEGG caught."""
    if ec in ("", "null"):
        return True
    home = couple_ec_classes(cname)
    return any(e.strip().split(".")[0] in home for e in ec.split("|"))


def pw_close(res):
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


def add(res, formula, charge, k):
    out = dict(res)
    for el, n in formula.items():
        out[el] = out.get(el, 0) + k * n
    out["charge"] = out.get("charge", 0) + k * charge
    return {e: v for e, v in out.items() if abs(v) > 1e-6}


def main():
    cpd = msbio.load_compounds(); rxn = msbio.load_reactions(True)
    info = msbio.SpeciesInfo.from_compounds(cpd)
    name_of = dict(zip(cpd["id"], cpd["name"]))

    # skip reactions already fixed by earlier stages
    done = set()
    for f in ("proton_water_corrections.tsv", "cosubstrate_corrections.tsv"):
        if os.path.exists(f):
            done |= set(pd.read_csv(f, sep="\t", dtype=str)["id"])

    auto, ambiguous = [], []
    for _, r in rxn.iterrows():
        if r["id"] in done:
            continue
        sp = msbio.parse_stoich(r["stoichiometry"])
        res0, fl = msbio.compute_residual(sp, info)
        if fl["no_formula"] or fl["unknown_charge"] or not res0:
            continue
        if pw_close(res0) is not None:
            continue

        ec = r["ec_numbers"]
        valid = []   # (couple_name, k, w, p, corroborated)
        for cname, (donor, acceptor, ecpref) in COUPLES.items():
            if not ec_compatible(ec, cname):   # EC prefilter: no transferase couple on an oxidoreductase
                continue
            df_ = info.formula(donor); dq_ = info.charge(donor)
            af_ = info.formula(acceptor); aq_ = info.charge(acceptor)
            if df_ is None or af_ is None or dq_ is None or aq_ is None:
                continue
            for k in (1, -1):
                # donor consumed on reactant side, acceptor produced -> net to residual:
                # +k*acceptor (products) - k*donor ... express as adding donor with -k, acceptor with +k
                res1 = add(res0, af_, aq_, k)
                res1 = add(res1, df_, dq_, -k)
                wp = pw_close(res1)
                if wp is None:
                    continue
                w, p = wp
                corrob = ec not in ("", "null") and any(e.strip().startswith(pref)
                            for e in ec.split("|") for pref in ecpref)
                valid.append((cname, k, w, p, corrob))
                break
        if not valid:
            continue
        corr = [v for v in valid if v[4]]
        rec = {"id": r["id"], "name": r["name"], "ec": ec,
               "residual": ",".join(f"{k}:{int(v):+d}" for k, v in sorted(res0.items())),
               "definition": r["definition"][:110],
               "candidates": "|".join(v[0] for v in valid)}
        # unambiguous auto-accept: exactly one EC-corroborated couple
        if len(corr) == 1:
            cname, k, w, p, _ = corr[0]
            donor, acceptor, _ = COUPLES[cname]
            new_sp = [dict(s) for s in sp]
            cmpt = Counter(s["cmpt"] for s in sp).most_common(1)[0][0]
            patch = {donor: -k, acceptor: k, "cpd00001": w, "cpd00067": p}
            for pc, pk in patch.items():
                if not pk: continue
                hit = next((s for s in new_sp if s["cpd"] == pc and s["cmpt"] == cmpt), None)
                if hit: hit["coeff"] += pk
                else: new_sp.append({"coeff": pk, "cpd": pc, "cmpt": cmpt, "comm": "",
                                     "name": name_of.get(pc, pc)})
            new_sp = [s for s in new_sp if abs(s["coeff"]) > 1e-9]
            rfin, _ = msbio.compute_residual(new_sp, info)
            if rfin or not msbio.is_valid_reaction(new_sp):
                continue
            rec.update({"couple": cname, "confidence": "high",
                        "patch": f"{cname} couple + water/proton",
                        "corrected_stoichiometry": msbio.serialize_stoich(new_sp)})
            auto.append(rec)
        else:
            ambiguous.append(rec)

    pd.DataFrame(auto).to_csv("couple_corrections.tsv", sep="\t", index=False)
    pd.DataFrame(ambiguous).to_csv("couple_candidates_for_agent.tsv", sep="\t", index=False)
    print(f"auto-accepted (unambiguous, EC-corroborated, balance-verified): {len(auto)}")
    if auto:
        print(Counter(a["couple"] for a in auto))
    print(f"ambiguous / no-EC -> for agent disambiguation: {len(ambiguous)}")
    if ambiguous:
        print("  candidate-set frequency:", Counter(a["candidates"] for a in ambiguous).most_common(8))


if __name__ == "__main__":
    main()
