#!/usr/bin/env python
"""
Stage 3 - cofactor-couple closure: generalize co-substrate closure to a DONOR/ACCEPTOR
couple (the group-transfer partner curators drop, e.g. SAM->SAH for a methylation).

For each imbalanced reaction we try inserting one couple (donor consumed, acceptor produced),
close the remainder with the deterministic proton/water rule, and verify mass+charge = 0.
The EC prefilter (mcb.closures.ec_compatible) blocks a transferase couple on an
oxidoreductase/hydrolase. Several couples can be arithmetically valid for the same residual
(any hexose-NDP donor closes a C6H10O5 deficit); those AMBIGUOUS reactions are emitted for
agent disambiguation by name, while UNAMBIGUOUS + EC-corroborated ones are auto-accepted.

Outputs (work dir):
  couple_corrections.tsv           auto-accepted, balance-verified
  couple_candidates_for_agent.tsv  >1 valid couple or no EC -> agent decides
"""
import os
from collections import Counter

import pandas as pd

from mcb import io as msbio
from mcb import closures as cl
from mcb.paths import wp


def main():
    cpd = msbio.load_compounds()
    rxn = msbio.load_reactions(True)
    info = msbio.SpeciesInfo.from_compounds(cpd)
    name_of = dict(zip(cpd["id"], cpd["name"]))

    # skip reactions already fixed by earlier stages
    done = set()
    for f in ("proton_water_corrections.tsv", "cosubstrate_corrections.tsv"):
        if os.path.exists(wp(f)):
            done |= set(pd.read_csv(wp(f), sep="\t", dtype=str)["id"])

    auto, ambiguous = [], []
    for _, r in rxn.iterrows():
        if r["id"] in done:
            continue
        sp = msbio.parse_stoich(r["stoichiometry"])
        if cl.is_refused_reaction(sp):                       # indefinite-polymer guard
            continue
        res0, fl = msbio.compute_residual(sp, info)
        if fl["no_formula"] or fl["unknown_charge"] or not res0:
            continue
        if cl.pw_close(res0) is not None:
            continue

        ec = r["ec_numbers"]
        valid = []   # (couple_name, k, w, p, corroborated)
        for cname, (donor, acceptor, ecpref) in cl.COUPLES.items():
            if not cl.ec_compatible(ec, cname):   # no transferase couple on an oxidoreductase
                continue
            df_, dq_ = info.formula(donor), info.charge(donor)
            af_, aq_ = info.formula(acceptor), info.charge(acceptor)
            if df_ is None or af_ is None or dq_ is None or aq_ is None:
                continue
            for k in (1, -1):
                res1 = cl.add_species(res0, af_, aq_, k)
                res1 = cl.add_species(res1, df_, dq_, -k)
                wpr = cl.pw_close(res1)
                if wpr is None:
                    continue
                w, p = wpr
                corrob = ec not in ("", "null") and any(
                    e.strip().startswith(pref) for e in ec.split("|") for pref in ecpref)
                valid.append((cname, k, w, p, corrob))
                break
        if not valid:
            continue
        corr = [v for v in valid if v[4]]
        rec = {"id": r["id"], "name": r["name"], "ec": ec,
               "residual": ",".join(f"{k}:{int(v):+d}" for k, v in sorted(res0.items())),
               "definition": r["definition"][:110],
               "candidates": "|".join(v[0] for v in valid)}
        if len(corr) == 1:                        # unambiguous auto-accept
            cname, k, w, p, _ = corr[0]
            donor, acceptor, _ = cl.COUPLES[cname]
            new_sp = [dict(s) for s in sp]
            cmpt = Counter(s["cmpt"] for s in sp).most_common(1)[0][0]
            patch = {donor: -k, acceptor: k, "cpd00001": w, "cpd00067": p}
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
            rfin, _ = msbio.compute_residual(new_sp, info)
            if rfin or not msbio.is_valid_reaction(new_sp):
                continue
            rec.update({"couple": cname, "confidence": "high",
                        "patch": f"{cname} couple + water/proton",
                        "corrected_stoichiometry": msbio.serialize_stoich(new_sp)})
            auto.append(rec)
        else:
            ambiguous.append(rec)

    pd.DataFrame(auto).to_csv(wp("couple_corrections.tsv"), sep="\t", index=False)
    pd.DataFrame(ambiguous).to_csv(wp("couple_candidates_for_agent.tsv"), sep="\t", index=False)
    print(f"auto-accepted (unambiguous, EC-corroborated, balance-verified): {len(auto)}")
    if auto:
        print(Counter(a["couple"] for a in auto))
    print(f"ambiguous / no-EC -> for agent disambiguation: {len(ambiguous)}")


if __name__ == "__main__":
    main()
