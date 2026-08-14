#!/usr/bin/env python
"""
Apply the phospho-disambiguation agent decisions:
  keep Pi / keep PPi -> leave the co-substrate correction as-is.
  ATP/ADP|GTP/GDP|UTP/UDP -> reassign: apply the nucleotide couple, balance-verify,
      write to phospho_corrections.tsv, and mark the reaction to be REMOVED from the
      free-Pi/PPi co-substrate corrections (it was the wrong mechanism).
  RETRACT -> remove the co-substrate correction (wrong Pi/PPi, no confident fix).

Writes:
  phospho_corrections.tsv         reassigned, balance-verified couple corrections
  phospho_overrides.tsv           id, action (reassigned|retracted) -> consumed by
                                  build_correction_log to drop those cosubstrate rows
Reads DB read-only.
"""
import pandas as pd
import msbio
from couple_closure import COUPLES
from apply_couples import apply_couple

DECISION_TO_COUPLE = {
    "ATP/ADP": "phospho ATP/ADP",
    "GTP/GDP": "phospho GTP/GDP",
    "UTP/UDP": "phospho UTP/UDP",
}


def main():
    cpd = msbio.load_compounds(); rxn = msbio.load_reactions(True)
    info = msbio.SpeciesInfo.from_compounds(cpd)
    name_of = dict(zip(cpd["id"], cpd["name"]))
    orig = {r["id"]: r for _, r in rxn.iterrows()}

    dec = pd.read_csv("agent_decisions_phospho.tsv", sep="\t", dtype=str, keep_default_na=False)

    corr_rows, override_rows = [], []
    from collections import Counter
    stats = Counter()
    for _, d in dec.iterrows():
        rid, decision = d["id"], d["decision"].strip()
        stats[decision] += 1
        if decision in ("keep Pi", "keep PPi"):
            continue
        if decision == "RETRACT":
            override_rows.append({"id": rid, "action": "retracted", "reason": d.get("reason", "")})
            continue
        cname = DECISION_TO_COUPLE.get(decision)
        if cname is None:
            stats[f"unknown:{decision}"] += 1
            continue
        o = orig.get(rid)
        if o is None:
            continue
        sp = msbio.parse_stoich(o["stoichiometry"])
        new_sp = apply_couple(sp, info, cname, name_of)
        if new_sp is None:
            stats["reassign_verify_failed"] += 1
            continue
        corr_rows.append({
            "id": rid, "orig_status": o["status"], "couple": cname,
            "confidence": "agent", "agent_reason": d.get("reason", ""),
            "patch": f"{cname} couple + water/proton (reassigned from free Pi/PPi)",
            "corrected_stoichiometry": msbio.serialize_stoich(new_sp),
        })
        override_rows.append({"id": rid, "action": "reassigned", "reason": d.get("reason", "")})

    pd.DataFrame(corr_rows).to_csv("phospho_corrections.tsv", sep="\t", index=False)
    pd.DataFrame(override_rows).to_csv("phospho_overrides.tsv", sep="\t", index=False)
    print("decisions:", dict(stats))
    print(f"reassigned (balance-verified): {len(corr_rows)}  written to phospho_corrections.tsv")
    print(f"overrides (drop from cosubstrate): {len(override_rows)} -> phospho_overrides.tsv")


if __name__ == "__main__":
    main()
