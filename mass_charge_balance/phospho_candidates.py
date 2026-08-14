#!/usr/bin/env python
"""
Phospho disambiguation candidates: the co-substrate stage greedily assigned FREE Pi/PPi
to phosphoryl-deficit reactions, but many are really nucleotide-dependent (a kinase
transfers phosphate FROM ATP, a nucleotidyltransferase FROM a (d)NTP). For each Pi/PPi
correction, test which nucleotide-couple mechanisms ALSO close the reaction. Those with
an alternative are genuinely ambiguous -> agent decides from EC/name.

Mechanisms tested (besides the current free Pi/PPi):
  ATP/ADP, GTP/GDP, UTP/UDP, CTP/CDP  (phosphoryl transfer, kinases 2.7.-)
  ATP/AMP + PPi                        (adenylyl transfer, 2.7.7 / ligases)

Output: phospho_candidates_for_agent.tsv  (id, name, ec, definition, current, alternatives)
Reads DB read-only.
"""
import pandas as pd
import msbio
from couple_closure import pw_close, add

# name -> list of (cpd, signed_multiplicity) added together as one mechanism unit
MECHS = {
    "free Pi":        [("cpd00009", +1)],
    "free PPi":       [("cpd00012", +1)],
    "ATP/ADP":        [("cpd00002", -1), ("cpd00008", +1)],
    "GTP/GDP":        [("cpd00038", -1), ("cpd00031", +1)],
    "UTP/UDP":        [("cpd00062", -1), ("cpd00014", +1)],
    "CTP/CDP":        [("cpd00052", -1), ("cpd00016", +1)],
    "ATP/AMP+PPi":    [("cpd00002", -1), ("cpd00018", +1), ("cpd00012", +1)],
}


def closes(res0, mech, info):
    """Does adding this mechanism (with an overall +/-1 orientation) + water/proton close?"""
    for orient in (1, -1):
        res = dict(res0)
        ok = True
        for cpd, mult in mech:
            f = info.formula(cpd); q = info.charge(cpd)
            if f is None or q is None:
                ok = False; break
            res = add(res, f, q, orient * mult)
        if ok and pw_close(res) is not None:
            return True
    return False


def main():
    cpd = msbio.load_compounds(); rxn = msbio.load_reactions(True)
    info = msbio.SpeciesInfo.from_compounds(cpd)
    orig = {r["id"]: r for _, r in rxn.iterrows()}

    cs = pd.read_csv("cosubstrate_corrections.tsv", sep="\t", dtype=str, keep_default_na=False)
    cs = cs[cs["co_substrate"].isin(["Pi", "PPi"])]

    rows = []
    for _, c in cs.iterrows():
        r = orig.get(c["id"])
        if r is None:
            continue
        sp = msbio.parse_stoich(r["stoichiometry"])
        res0, fl = msbio.compute_residual(sp, info)
        if not res0:
            continue
        alts = [m for m in ("ATP/ADP", "GTP/GDP", "UTP/UDP", "CTP/CDP", "ATP/AMP+PPi")
                if closes(res0, MECHS[m], info)]
        if not alts:
            continue  # only free Pi/PPi works -> keep as-is, not ambiguous
        rows.append({
            "id": c["id"], "name": r["name"], "ec": r["ec_numbers"],
            "current_assignment": f"free {c['co_substrate']}",
            "alternatives": "|".join(alts),
            "residual": ",".join(f"{k}:{int(v):+d}" for k, v in sorted(res0.items())),
            "definition": r["definition"][:120],
        })

    df = pd.DataFrame(rows)
    df.to_csv("phospho_candidates_for_agent.tsv", sep="\t", index=False)
    print(f"Pi/PPi corrections total: {len(cs)}")
    print(f"ambiguous (a nucleotide couple also closes): {len(df)}")
    if len(df):
        from collections import Counter
        print("alternative-set frequency:", Counter(df["alternatives"]).most_common())
        print("\nexamples:")
        for _, r in df.head(8).iterrows():
            print(f"  {r['id']} EC={r['ec'][:14]:14s} cur={r['current_assignment']:8s} "
                  f"alt={r['alternatives']:20s} {r['name'][:40]}")


if __name__ == "__main__":
    main()
