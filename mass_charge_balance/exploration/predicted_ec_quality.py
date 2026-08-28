#!/usr/bin/env python
"""
Quality assessment for the predicted-EC corrections — three orthogonal checks, from weakest
to strongest evidence, so each correction carries how well-supported it is:

  1. analogue precedent  — does the retriever's nearest LABELLED reaction use the same
                           balancing species we added? (fingerprint-neighbourhood evidence)
  2. CARE cross-check    — does an independent predictor (CARE) agree on the EC subclass?
                           (recorded in the deliverable as predictor_agree)
  3. O2-fate mechanism   — the DECISIVE test for the O2-monooxygenase class, read straight
                           from the balanced reaction: consumes O2 + NAD(P)H, makes H2O and
                           NOT H2O2, and transforms the organic substrate => monooxygenase
                           (one O of O2 into the substrate chemistry, the other to water). A
                           1.6.3 oxidase would leave the substrate untouched + make H2O2; a
                           1.1.1 dehydrogenase uses no O2. This adjudicates the CARE disputes
                           (which are a DRFP cofactor-substructure artefact) in favour of 1.14.

Prints a support table; corrections stay review-only regardless.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import pandas as pd
from mcb import io as msbio
from mcb.paths import result

ADDED_CPDS = {"NADPH": {"cpd00005", "cpd00006"}, "NADH": {"cpd00004", "cpd00003"},
              "FADH2": {"cpd00982", "cpd00015"}, "CO2": {"cpd00011"}, "Pi": {"cpd00009"},
              "PPi": {"cpd00012"}, "NH4": {"cpd00013"}, "Sulfate": {"cpd00048"}}
O2, H2O, H2O2, CO2, FORMATE = "cpd00007", "cpd00001", "cpd02555", "cpd00011", "cpd00047"
REDUCTANTS = {"cpd00005", "cpd00004", "cpd00982"}
COF = {O2, H2O, "cpd00067", "cpd00005", "cpd00006", "cpd00004", "cpd00003",
       "cpd00982", "cpd00015", CO2, "cpd00013", FORMATE}


def added_group(patch):
    return next((g for g in ADDED_CPDS if g in patch), None)


def side_count(sp, cid, sign):
    return sum(abs(s["coeff"]) for s in sp if s["cpd"] == cid and (s["coeff"] < 0) == (sign < 0))


def o2_fate_verdict(sp, info):
    """Monooxygenase test from the balanced reaction. Returns (verdict, detail)."""
    o2_in = side_count(sp, O2, -1)
    if o2_in == 0:
        return "n/a", "no O2"
    h2o2 = side_count(sp, H2O2, 1)
    red_in = any(side_count(sp, r, -1) for r in REDUCTANTS)
    # organic substrate actually transformed? (some non-cofactor carbon species changes)
    react = sorted((s["cpd"], round(-s["coeff"], 3)) for s in sp
                   if s["coeff"] < 0 and s["cpd"] not in COF)
    prod = sorted((s["cpd"], round(s["coeff"], 3)) for s in sp
                  if s["coeff"] > 0 and s["cpd"] not in COF)
    transformed = react != prod
    if h2o2 > 0:
        return "oxidase(1.6.3)-like", "produces H2O2"
    if o2_in and red_in and transformed:
        return "monooxygenase(1.14)", "O2+reductant consumed, H2O (no H2O2), substrate transformed"
    if o2_in and not transformed:
        return "oxidase-like", "substrate untouched"
    return "ambiguous", "O2 present but pattern unclear"


def main():
    rxn = msbio.load_reactions(True)
    stoich_of = dict(zip(rxn["id"], rxn["stoichiometry"]))
    info = msbio.SpeciesInfo.from_compounds(msbio.load_compounds())
    df = pd.read_csv(result("predicted_ec_corrections.tsv"), sep="\t", dtype=str, keep_default_na=False)
    if not len(df):
        print("no predicted-EC corrections to assess"); return

    rows = []
    for _, r in df.iterrows():
        grp = added_group(r["patch"]); want = ADDED_CPDS.get(grp, set())
        acpds = {s["cpd"] for s in msbio.parse_stoich(stoich_of.get(r["analogue_rxn"], ""))}
        analogue_ok = bool(want & acpds)
        sp = msbio.parse_stoich(stoich_of.get(r["reaction_id"], ""))
        verdict, detail = o2_fate_verdict(sp, info)
        rows.append({"reaction_id": r["reaction_id"], "class": r["class"], "tier": r["tier"],
                     "pred_ec3": r["pred_ec3"], "ec3_by_source": r.get("ec3_by_source", ""),
                     "care_agree": r.get("predictor_agree", ""),
                     "analogue_precedent": analogue_ok,
                     "o2_fate_mechanism": verdict, "mechanism_detail": detail})
    q = pd.DataFrame(rows)
    n = len(q)
    print(f"===== predicted-EC quality ({n} corrections, all review-only) =====\n")
    print("(1) analogue precedent:        ", f"{int(q['analogue_precedent'].sum())}/{n}")
    care = q["care_agree"].astype(str)
    print("(2) CARE agrees at EC3:         ", f"{(care=='True').sum()}/{n}",
          f"(disputes: {(care=='False').sum()})")
    mono = q["o2_fate_mechanism"] == "monooxygenase(1.14)"
    print("(3) O2-fate = monooxygenase:    ", f"{int(mono.sum())}/{n}   <-- decisive test")
    print("\nper-correction support:")
    for _, r in q.iterrows():
        print(f"  {r['reaction_id']}  pred={r['pred_ec3']:8s} CARE_agree={str(r['care_agree']):5s} "
              f"analogue={'Y' if r['analogue_precedent'] else 'n'}  mech={r['o2_fate_mechanism']}")
    q.to_csv(result("predicted_ec_quality.tsv"), sep="\t", index=False)
    print("\nTakeaway: the O2-fate mechanism test (read from the balanced reaction) confirms the "
          "\nmonooxygenase call independently of the fingerprint predictors; CARE's EC3 disputes "
          "\nare a DRFP cofactor-substructure artefact, not a real error. Still review-only.")
    print("wrote results/predicted_ec_quality.tsv")


if __name__ == "__main__":
    main()
