#!/usr/bin/env python
"""Read a few reactions from each non-deterministic category with full context, so we
can judge how much reasoning each needs (-> cheap subagent vs main model)."""
import pandas as pd
import msbio

cpd = msbio.load_compounds(); rxn = msbio.load_reactions(True)
info = msbio.SpeciesInfo.from_compounds(cpd)
name = dict(zip(cpd["id"], cpd["name"]))
tri = pd.read_csv("agent_triage.tsv", sep="\t", dtype=str, keep_default_na=False)
rxn_by_id = {r["id"]: r for _, r in rxn.iterrows()}

# (label, filter) -> pick first N ids
TARGETS = [
    ("charge_only ±2, NO redox carrier", lambda d: (d["class"]=="charge_only") & (d["signature"].isin(["charge:+2","charge:-2"])) & (d["redox"]=="")),
    ("charge_only ±2, NAD/NADP",         lambda d: (d["class"]=="charge_only") & (d["redox"].str.contains("NAD"))),
    ("oxygen_only O:+1",                 lambda d: (d["class"]=="oxygen_only") & (d["signature"]=="O:+1")),
    ("H_O_charge",                        lambda d: (d["class"]=="H_O_charge")),
    ("skeleton:C  CH2 (C:+1,H:+2)",      lambda d: (d["signature"]=="C:+1,H:+2")),
    ("skeleton:C  hexose (C:+6,H:+10,O:+5)", lambda d: (d["signature"]=="C:+6,H:+10,O:+5")),
    ("skeleton:P  phosphate",            lambda d: (d["class"]=="skeleton:P")),
    ("skeleton big C,N,S",               lambda d: (d["class"]=="skeleton:C,N,S")),
]

for label, filt in TARGETS:
    sub = tri[filt(tri)]
    print("\n" + "="*90)
    print(f"### {label}   (group size ~{len(sub)})")
    for rid in sub["id"].head(2):
        r = rxn_by_id[rid]
        sp = msbio.parse_stoich(r["stoichiometry"])
        res,_ = msbio.compute_residual(sp, info)
        print(f"\n  {rid}  | {r['name']}")
        print(f"    EC={r['ec_numbers']}  status={r['status']}  residual={ {k:int(v) for k,v in res.items()} }")
        print(f"    def: {r['definition']}")
        for s in sp:
            print(f"       {s['coeff']:+g} {s['cpd']:10s} {name.get(s['cpd'],'')[:26]:26s} "
                  f"{info.formula_of.get(s['cpd']):12s} q={info.charge(s['cpd'])}")
