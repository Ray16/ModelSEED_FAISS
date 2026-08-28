#!/usr/bin/env python
"""
Predicted-EC stage (OPT-IN, review-only) — corrections for EC=null reactions unlocked by a
*predicted* EC, kept strictly separate from the curated-EC deliverable.

Not part of `mcb run`. Run explicitly: `python -m mcb predict_ec`. Reads the frozen
predictions in inputs/predicted_ec*.tsv (see mcb.ec_predict), and only for reactions where
ModelSEED has NO curated EC and a predictor is confident + in-domain. Two well-defined
classes, each balance-verified and gated on the PREDICTED EC corroborating the mechanism:

  C. co-substrate menu-expansion — predicted EC3 corroborates a stocked co-substrate
     (CO2<-4.1.1, Pi<-3.1.3/3.6.1, PPi<-2.7.7, NH4<-3.5.1, sulfate<-2.8.2): run the same
     single-co-substrate + proton/water close.
  A. O2 monooxygenase — O2 present, only H/O/charge off, predicted EC3 pins the reductant
     (1.14.13 NADPH / 1.14.12 NADH / 1.14.14 FAD): insert reductant->oxidized couple + p/w.

Output (SEPARATE from the 516): results/predicted_ec_corrections.tsv, every row tagged
tier=review-predicted with ec_source, pred_ec3, confidence and the nearest labelled analogue.
These are candidates for expert review — never auto-apply.
"""
from collections import Counter

import os

import pandas as pd

from mcb import io as msbio
from mcb import closures as cl
from mcb import ec_predict
from mcb.paths import result

O2 = "cpd00007"
# predicted EC3 prefix -> reductant couple (reduced donor, oxidized acceptor)
REDUCTANT = {
    "1.14.13": ("cpd00005", "cpd00006", "NADPH/NADP"),
    "1.14.12": ("cpd00004", "cpd00003", "NADH/NAD"),
    "1.14.14": ("cpd00982", "cpd00015", "FADH2/FAD"),
}
# predicted EC3 prefix -> stocked co-substrate cpd id (mirror of closures.EC_CORROBORATION)
COSUB_EC = {
    "4.1.1": "cpd00011", "6.4.1": "cpd00011",          # CO2
    "3.1.3": "cpd00009", "3.1.4": "cpd00009",          # Pi
    "3.6.1": "cpd00009", "2.7.7": "cpd00012",          # Pi / PPi
    "3.5.1": "cpd00013", "3.5.4": "cpd00013", "4.3.1": "cpd00013",  # NH4
    "2.8.2": "cpd00048", "3.1.6": "cpd00048",          # sulfate
}


def _apply(sp, info, patch, name_of):
    """Apply {cpd: net_coeff} to a species list; return verified species or None."""
    new_sp = [dict(s) for s in sp]
    cmpt = Counter(s["cmpt"] for s in sp).most_common(1)[0][0]
    for pc, pk in patch.items():
        if not pk:
            continue
        hit = next((s for s in new_sp if s["cpd"] == pc and s["cmpt"] == cmpt), None)
        if hit:
            hit["coeff"] += pk
        else:
            new_sp.append({"coeff": pk, "cpd": pc, "cmpt": cmpt, "comm": "",
                           "name": name_of.get(pc, pc)})
    new_sp = [s for s in new_sp if abs(s["coeff"]) > 1e-9]
    res, fl = msbio.compute_residual(new_sp, info)
    if res or fl["no_formula"] or fl["unknown_charge"] or not msbio.is_valid_reaction(new_sp):
        return None
    return new_sp


def _cosub_fix(res0, sp, info, ec3, name_of):
    for pre, cid in COSUB_EC.items():
        if not ec3.startswith(pre):
            continue
        nm, formula, charge = cl.HEAVY[cid]
        k = cl.single_closes(res0, formula, charge)
        if k is None:
            continue
        w, p = cl.proton_water_close(cl.add_species(res0, formula, charge, k))
        new = _apply(sp, info, {cid: k, "cpd00001": w, "cpd00067": p}, name_of)
        if new:
            return new, f"co-substrate {nm} + water/proton (predicted-EC {pre})"
    return None, None


def _o2_fix(res0, sp, info, ec3, name_of):
    for pre, (donor, acceptor, lbl) in REDUCTANT.items():
        if not ec3.startswith(pre):
            continue
        df_, dq_ = info.formula(donor), info.charge(donor)
        af_, aq_ = info.formula(acceptor), info.charge(acceptor)
        if None in (df_, dq_, af_, aq_):
            continue
        for k in (1, -1):
            res1 = cl.add_species(cl.add_species(res0, af_, aq_, k), df_, dq_, -k)
            wp = cl.pw_close(res1)
            if wp is None:
                continue
            w, p = wp
            new = _apply(sp, info, {donor: -k, acceptor: k, "cpd00001": w, "cpd00067": p}, name_of)
            if new:
                return new, f"monooxygenase reductant {lbl} + water/proton (predicted-EC {pre})"
    return None, None


def main():
    cpd = msbio.load_compounds()
    rxn = msbio.load_reactions(True)
    info = msbio.SpeciesInfo.from_compounds(cpd)
    name_of = dict(zip(cpd["id"], cpd["name"]))
    preds = ec_predict.load_predicted_ec()
    print(f"predicted-EC table: {len(preds)} reactions, "
          f"{sum(1 for p in preds.values() if p['usable'])} usable (confident+in-domain)")

    # never re-surface a reaction the curated pipeline already corrected (some EC=null
    # reactions are already fixed as no-EC 'review' co-substrate in the 516)
    already = set()
    if os.path.exists(result("corrected_reactions.tsv")):
        already = set(pd.read_csv(result("corrected_reactions.tsv"), sep="\t",
                                  dtype=str)["reaction_id"])

    rows = []
    stats = Counter()
    for _, r in rxn.iterrows():
        if r["ec_numbers"] not in ("", "null"):
            continue                                   # curated EC -> handled by the main pipeline
        if r["id"] in already:
            continue                                   # already in the curated 516 -> not new
        ec3, source, conf, corr = ec_predict.effective_ec(r["id"], r["ec_numbers"], preds)
        if source == "none":
            continue
        sp = msbio.parse_stoich(r["stoichiometry"])
        if cl.is_refused_reaction(sp):
            continue
        res0, fl = msbio.compute_residual(sp, info)
        if fl["no_formula"] or fl["unknown_charge"] or not res0:
            continue
        if cl.proton_water_close(res0) is not None:
            continue                                   # deterministic, needs no EC

        heavy = {k for k in res0 if k not in ("H", "O", "charge")}
        new_sp, patch, klass = None, None, None
        if O2 in {s["cpd"] for s in sp} and not heavy:
            new_sp, patch = _o2_fix(res0, sp, info, ec3, name_of); klass = "O2-monooxygenase"
        if new_sp is None and heavy:
            new_sp, patch = _cosub_fix(res0, sp, info, ec3, name_of); klass = "cosubstrate"
        if new_sp is None:
            continue

        p = preds[r["id"]]
        stats[klass] += 1
        # cross-predictor corroboration (rxnfp drives; CARE etc. corroborate at EC3)
        by = corr["by_source"]
        agree = corr["agree"]
        tier = ("review-predicted-corroborated" if agree is True
                else "review-predicted-disputed" if agree is False
                else "review-predicted-single")
        rows.append({
            "reaction_id": r["id"], "name": r["name"], "class": klass,
            "tier": tier, "ec_source": source,
            "pred_ec3": ec3, "pred_conf": conf,
            "predictor_agree": agree, "n_predictors": corr["n_sources"],
            "ec3_by_source": ";".join(f"{s}={e}" for s, e in sorted(by.items())),
            "analogue_rxn": p["analogue"], "analogue_ec": p["analogue_ec"],
            "patch": patch,
            "equation_before": msbio.render_equation(sp, name_of),
            "equation_after": msbio.render_equation(new_sp, name_of),
            "stoichiometry_before": r["stoichiometry"],
            "stoichiometry_corrected": msbio.serialize_stoich(new_sp),
        })

    df = pd.DataFrame(rows).sort_values(["class", "reaction_id"]) if rows else pd.DataFrame(rows)
    df.to_csv(result("predicted_ec_corrections.tsv"), sep="\t", index=False)
    print(f"\npredicted-EC corrections (review-predicted, balance-verified): {len(df)}")
    print("by class:", dict(stats))
    if len(df):
        print("by tier (cross-predictor corroboration):")
        print(df["tier"].value_counts().to_string())
    print("ALL are review-predicted — tagged by ec_source + predictor agreement, NOT in the 516.")
    print("wrote results/predicted_ec_corrections.tsv")


if __name__ == "__main__":
    main()
