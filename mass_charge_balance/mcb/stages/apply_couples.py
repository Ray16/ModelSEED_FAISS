#!/usr/bin/env python
"""
Stage 4 - apply agent couple-decisions -> balance-verified, logged corrections.

Reads agent_decisions_*.tsv (id, decision, reason) where `decision` is a couple name from
mcb.closures.COUPLES or "SKIP". For each non-SKIP decision we insert the chosen
donor/acceptor couple, close the remainder with proton/water, and ACCEPT ONLY IF mass+charge
verifies to zero -- the agent's pick is never trusted blind. The EC prefilter still applies
(no transferase couple on an oxidoreductase), and glycosyl picks get a stereo tier.

Output (work dir): couple_agent_corrections.tsv
"""
import glob
import os
from collections import Counter

import pandas as pd

from mcb import io as msbio
from mcb import closures as cl
from mcb.stereo import glycosyl_confidence
from mcb.paths import work, INPUTS


def apply_couple(sp, info, cname, name_of):
    """Try inserting couple cname into species list sp; return corrected species or None."""
    donor, acceptor, _ = cl.COUPLES[cname]
    df_, dq_ = info.formula(donor), info.charge(donor)
    af_, aq_ = info.formula(acceptor), info.charge(acceptor)
    if None in (df_, dq_, af_, aq_):
        return None
    res0, fl = msbio.compute_residual(sp, info)
    if fl["no_formula"] or fl["unknown_charge"]:
        return None
    for k in (1, -1):
        res1 = cl.add_species(cl.add_species(res0, af_, aq_, k), df_, dq_, -k)
        wpr = cl.pw_close(res1)
        if wpr is None:
            continue
        w, p = wpr
        new_sp = [dict(s) for s in sp]
        cmpt = Counter(s["cmpt"] for s in sp).most_common(1)[0][0]
        for pc, pk in {donor: -k, acceptor: k, "cpd00001": w, "cpd00067": p}.items():
            if not pk:
                continue
            hit = next((s for s in new_sp if s["cpd"] == pc and s["cmpt"] == cmpt), None)
            if hit:
                hit["coeff"] += pk
            else:
                new_sp.append({"coeff": pk, "cpd": pc, "cmpt": cmpt, "comm": "",
                               "name": name_of.get(pc, pc)})
        new_sp = [s for s in new_sp if abs(s["coeff"]) > 1e-9]
        rfin, _ = msbio.compute_residual(new_sp, info)
        if not rfin and msbio.is_valid_reaction(new_sp):
            return new_sp
    return None


def main():
    cpd = msbio.load_compounds()
    rxn = msbio.load_reactions(True)
    info = msbio.SpeciesInfo.from_compounds(cpd)
    name_of = dict(zip(cpd["id"], cpd["name"]))
    smiles_of = dict(zip(cpd["id"], cpd["smiles"]))
    orig = {r["id"]: r for _, r in rxn.iterrows()}

    decisions = {}
    for f in sorted(glob.glob(os.path.join(str(INPUTS), "agent_decisions_*.tsv"))):
        if "phospho" in os.path.basename(f):
            continue
        d = pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False)
        for _, r in d.iterrows():
            decisions[r["id"]] = (r["decision"].strip(), r.get("reason", ""))
    print(f"agent decisions loaded: {len(decisions)}")

    rows = []
    stats = Counter()
    for rid, (decision, reason) in decisions.items():
        if decision in ("SKIP", ""):
            stats["skip"] += 1
            continue
        if decision not in cl.COUPLES:
            stats[f"unknown_decision:{decision}"] += 1
            continue
        o = orig.get(rid)
        if o is None:
            stats["missing_rxn"] += 1
            continue
        if not cl.ec_compatible(o["ec_numbers"], decision):   # EC prefilter defense
            stats["ec_incompatible_skipped"] += 1
            continue
        sp = msbio.parse_stoich(o["stoichiometry"])
        new_sp = apply_couple(sp, info, decision, name_of)
        if new_sp is None:
            stats["verify_failed"] += 1
            continue
        stats["applied"] += 1
        stereo_tier, stereo_detail = "n/a", ""
        if "glucosyl" in decision or "mannosyl" in decision:
            stereo_tier, stereo_detail = glycosyl_confidence(sp, info, smiles_of)
            stats[f"stereo:{stereo_tier}"] += 1
        rows.append({
            "id": rid, "orig_status": o["status"], "couple": decision,
            "confidence": "agent", "stereo_tier": stereo_tier,
            "stereo_detail": stereo_detail, "agent_reason": reason,
            "patch": f"{decision} couple + water/proton",
            "corrected_stoichiometry": msbio.serialize_stoich(new_sp),
        })

    pd.DataFrame(rows).to_csv(work("couple_agent_corrections.tsv"), sep="\t", index=False)
    print("stats:", dict(stats))
    print(f"wrote couple_agent_corrections.tsv ({len(rows)} balance-verified)")
    if rows:
        print("by couple:", Counter(r["couple"] for r in rows))


if __name__ == "__main__":
    main()
