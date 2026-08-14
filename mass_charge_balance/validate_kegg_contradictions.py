#!/usr/bin/env python
"""
Refine the KEGG check: a "not_in_kegg" correction is only WRONG if KEGG shows a
DIFFERENT mechanism -- i.e. KEGG's reaction contains a cofactor implying another
chemistry that our corrected reaction lacks. If KEGG merely omits the same cofactor
(no contradicting cofactor present), our fix is plausibly right and KEGG is incomplete.

For each KEGG-checkable correction:
  map our CORRECTED participants -> KEGG C-numbers (OURS)
  MARKERS_in_kegg_not_ours = (KEGG participant set ∩ cofactor markers) − OURS
  verdict:
    validated          added cofactor present in KEGG
    kegg_contradicts   KEGG has a mechanism cofactor we lack -> LIKELY WRONG (retract)
    kegg_incomplete    KEGG lacks our cofactor AND no contradicting marker -> keep
"""
import os
from collections import defaultdict, Counter
import pandas as pd
import msbio

ALIAS = os.path.join(msbio.BIOCHEM, "Aliases")
EXCLUDE = {"cpd00001", "cpd00067"}

# KEGG C-numbers whose presence implies a specific non-trivial mechanism
MARKERS = {
    "C00026": "2-oxoglutarate", "C00042": "succinate",   # 2-OG dioxygenases
    "C00007": "O2", "C00003": "NAD", "C00004": "NADH",
    "C00005": "NADPH", "C00006": "NADP", "C00016": "FAD", "C01352": "FADH2",
    "C00028": "acceptor", "C00030": "reduced-acceptor",
    "C00138": "reduced-ferredoxin", "C00139": "oxidized-ferredoxin",
}


def main():
    cpd = msbio.load_compounds(); info = msbio.SpeciesInfo.from_compounds(cpd)
    log = pd.read_csv("corrections_log.tsv", sep="\t", dtype=str, keep_default_na=False)
    val = pd.read_csv("kegg_validation.tsv", sep="\t", dtype=str, keep_default_na=False)
    kegg = pd.read_csv("kegg_cache.tsv", sep="\t", dtype=str, keep_default_na=False)
    kmap = {r["R"]: set(r["compounds"].split(",")) if r["compounds"] else set()
            for _, r in kegg.iterrows()}

    ra = pd.read_csv(f"{ALIAS}/Unique_ModelSEED_Reaction_Aliases.txt", sep="\t", dtype=str)
    ra = ra[ra["Source"] == "KEGG"]
    r2k = defaultdict(list)
    for _, r in ra.iterrows():
        if r["External ID"].startswith("R"):
            r2k[r["ModelSEED ID"]].append(r["External ID"])
    ca = pd.read_csv(f"{ALIAS}/Unique_ModelSEED_Compound_Aliases.txt", sep="\t", dtype=str)
    ca = ca[ca["Source"] == "KEGG"]
    cpd2k = defaultdict(set)
    for _, r in ca.iterrows():
        if r["External ID"].startswith("C"):
            cpd2k[r["ModelSEED ID"]].add(r["External ID"])

    log_by = {r["reaction_id"]: r for _, r in log.iterrows()}
    verdicts = Counter()
    out = []
    for _, v in val.iterrows():
        if v["verdict"] not in ("validated", "not_in_kegg", "partial"):
            continue
        rid = v["reaction_id"]
        keggset = set()
        for R in r2k.get(rid, []):
            keggset |= kmap.get(R, set())
        if not keggset:
            continue
        # our corrected participants -> KEGG C-numbers
        ours = set()
        for s in msbio.parse_stoich(log_by[rid]["stoichiometry_after"]):
            ours |= cpd2k.get(s["cpd"], set())
        contradicting = {c for c in keggset if c in MARKERS} - ours
        if v["verdict"] == "validated":
            verdict = "validated"
        elif contradicting:
            verdict = "kegg_contradicts"
        else:
            verdict = "kegg_incomplete"
        verdicts[verdict] += 1
        out.append({"reaction_id": rid, "category": v["category"],
                    "confidence": v["confidence"], "kegg_verdict": verdict,
                    "contradicting_markers": ",".join(sorted(MARKERS[c] for c in contradicting)),
                    "added": v["added"]})

    odf = pd.DataFrame(out)
    odf.to_csv("kegg_verdicts_refined.tsv", sep="\t", index=False)
    print("===== refined KEGG verdicts (checkable corrections) =====")
    for k, n in verdicts.most_common():
        print(f"  {k:18s} {n}")
    tot = sum(verdicts.values())
    good = verdicts["validated"] + verdicts["kegg_incomplete"]
    print(f"\n  consistent with KEGG (validated + kegg_incomplete): {good}/{tot} ({100*good/tot:.0f}%)")
    print(f"  KEGG CONTRADICTS (likely wrong -> retract): {verdicts['kegg_contradicts']}")
    print("\n  by category:")
    for cat in sorted(odf["category"].unique()):
        c = odf[odf["category"] == cat]
        print(f"    {cat:22s} contradicts={ (c['kegg_verdict']=='kegg_contradicts').sum() }/{len(c)}")
    print("\n  contradictions to retract:")
    for _, r in odf[odf["kegg_verdict"] == "kegg_contradicts"].iterrows():
        print(f"    {r['reaction_id']} [{r['category']}] added={r['added']} "
              f"kegg_has={r['contradicting_markers']}")
    print("\nwrote kegg_verdicts_refined.tsv")


if __name__ == "__main__":
    main()
