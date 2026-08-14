#!/usr/bin/env python
"""
External validation: does KEGG's version of the reaction list the cofactor/co-substrate
we RESTORED? This is ground truth independent of our pipeline. Applies to co-substrate
and couple corrections (proton/water adds only H+/H2O, which KEGG omits, so it can't be
checked this way).

Method:
  1. map each corrected reaction -> KEGG R-number(s)  (reaction aliases)
  2. diff before/after stoichiometry -> the ADDED heavy species (exclude H2O/H+)
  3. map added species -> KEGG C-number(s)  (compound aliases)
  4. fetch KEGG reaction EQUATION (cached), collect its C-number set
  5. verdict: validated (all added species appear in KEGG) / partial / not_in_kegg /
     no_mapping / no_kegg_xref
Reads DB read-only + KEGG REST (cached to kegg_cache.tsv).
"""
import os
import subprocess
import time
from collections import defaultdict, Counter
import pandas as pd
import msbio

ALIAS = os.path.join(msbio.BIOCHEM, "Aliases")
CACHE = "kegg_cache.tsv"
EXCLUDE = {"cpd00001", "cpd00067"}  # water, proton: KEGG omits


def load_kegg_maps():
    ra = pd.read_csv(f"{ALIAS}/Unique_ModelSEED_Reaction_Aliases.txt", sep="\t", dtype=str)
    ra = ra[ra["Source"] == "KEGG"]
    rxn2k = defaultdict(set)
    for _, r in ra.iterrows():
        if r["External ID"].startswith("R"):
            rxn2k[r["ModelSEED ID"]].add(r["External ID"])
    ca = pd.read_csv(f"{ALIAS}/Unique_ModelSEED_Compound_Aliases.txt", sep="\t", dtype=str)
    ca = ca[ca["Source"] == "KEGG"]
    cpd2k = defaultdict(set)
    for _, r in ca.iterrows():
        if r["External ID"].startswith("C"):
            cpd2k[r["ModelSEED ID"]].add(r["External ID"])
    return rxn2k, cpd2k


def fetch_kegg(rnums):
    """Fetch KEGG reactions in batches of 10; return {R: set(C-numbers)}. Cached."""
    cache = {}
    if os.path.exists(CACHE):
        cdf = pd.read_csv(CACHE, sep="\t", dtype=str, keep_default_na=False)
        cache = {r["R"]: set(r["compounds"].split(",")) if r["compounds"] else set()
                 for _, r in cdf.iterrows()}
    todo = [r for r in rnums if r not in cache]
    for i in range(0, len(todo), 10):
        batch = todo[i:i + 10]
        url = "https://rest.kegg.jp/get/" + "+".join("rn:" + b for b in batch)
        try:
            out = subprocess.run(["curl", "-s", "--max-time", "30", url],
                                 capture_output=True, text=True, timeout=40).stdout
        except Exception:
            out = ""
        # parse entries split by ///
        cur, eq = None, None
        for entry in out.split("///"):
            R = None; comps = set()
            for line in entry.splitlines():
                if line.startswith("ENTRY"):
                    R = line.split()[1]
                if line.startswith("EQUATION"):
                    comps = set(t for t in line.replace("EQUATION", "").split()
                                if t.startswith("C") and t[1:].isdigit())
            if R:
                cache[R] = comps
        for b in batch:
            cache.setdefault(b, set())  # mark fetched even if empty
        time.sleep(0.34)
    # write cache
    pd.DataFrame([{"R": k, "compounds": ",".join(sorted(v))} for k, v in cache.items()]
                 ).to_csv(CACHE, sep="\t", index=False)
    return cache


def main():
    cpd = msbio.load_compounds(); info = msbio.SpeciesInfo.from_compounds(cpd)
    log = pd.read_csv("corrections_log.tsv", sep="\t", dtype=str, keep_default_na=False)
    rxn2k, cpd2k = load_kegg_maps()

    # collect KEGG R-numbers to fetch (only for reactions with a KEGG xref that are
    # co-substrate/couple i.e. added a heavy species)
    def added_heavy(r):
        before = {s["cpd"]: s["coeff"] for s in msbio.parse_stoich(r["stoichiometry_before"])}
        after = {s["cpd"]: s["coeff"] for s in msbio.parse_stoich(r["stoichiometry_after"])}
        added = []
        for c, coeff in after.items():
            if c in EXCLUDE:
                continue
            if c not in before or abs(after[c] - before.get(c, 0)) > 1e-9:
                # species newly present OR coeff changed -> candidate added cofactor
                if c not in before:
                    added.append(c)
        return added

    rows = []
    need = set()
    for _, r in log.iterrows():
        rs = rxn2k.get(r["reaction_id"], set())
        if not rs:
            rows.append((r, [], "no_kegg_xref", set())); continue
        added = added_heavy(r)
        if not added:
            rows.append((r, [], "no_heavy_added", set())); continue
        need |= rs
        rows.append((r, added, None, rs))

    kegg = fetch_kegg(sorted(need))

    out = []
    verdicts = Counter()
    for r, added, pre, rs in rows:
        if pre:
            verdicts[pre] += 1
            out.append({**base(r), "verdict": pre, "added": "", "detail": ""})
            continue
        kegg_comps = set()
        for R in rs:
            kegg_comps |= kegg.get(R, set())
        if not kegg_comps:
            v = "kegg_no_equation"
            detail = ";".join(sorted(rs))
        else:
            checks = []
            for c in added:
                cks = cpd2k.get(c, set())
                if not cks:
                    checks.append((c, "no_mapping"))
                elif cks & kegg_comps:
                    checks.append((c, "present"))
                else:
                    checks.append((c, "absent"))
            statuses = [s for _, s in checks]
            if all(s == "present" for s in statuses):
                v = "validated"
            elif any(s == "present" for s in statuses):
                v = "partial"
            elif any(s == "absent" for s in statuses):
                v = "not_in_kegg"
            else:
                v = "no_mapping"
            detail = ",".join(f"{info.formula_of.get(c,c)}:{s}" for c, s in checks)
        verdicts[v] += 1
        out.append({**base(r), "verdict": v, "added": ",".join(added), "detail": detail})

    odf = pd.DataFrame(out)
    odf.to_csv("kegg_validation.tsv", sep="\t", index=False)

    print("===== KEGG participant validation =====")
    for k, v in verdicts.most_common():
        print(f"  {k:18s} {v}")
    checkable = odf[odf["verdict"].isin(["validated", "partial", "not_in_kegg"])]
    if len(checkable):
        good = (checkable["verdict"] == "validated").sum()
        print(f"\n  Among KEGG-checkable ({len(checkable)}): validated={good} "
              f"({100*good/len(checkable):.0f}%), "
              f"partial={(checkable['verdict']=='partial').sum()}, "
              f"not_in_kegg={(checkable['verdict']=='not_in_kegg').sum()}")
    print("\n  by category (validated / checkable):")
    for cat in sorted(odf["category"].unique()):
        c = odf[(odf["category"] == cat) & odf["verdict"].isin(["validated","partial","not_in_kegg"])]
        if len(c):
            print(f"    {cat:22s} {(c['verdict']=='validated').sum()}/{len(c)}")
    print("\n  DISAGREEMENTS (not_in_kegg) to review:")
    for _, r in odf[odf["verdict"] == "not_in_kegg"].head(12).iterrows():
        print(f"    {r['reaction_id']} [{r['category']}] added={r['added']}  {r['detail']}")
    print("\nwrote kegg_validation.tsv")


def base(r):
    return {"reaction_id": r["reaction_id"], "category": r["category"],
            "confidence": r["confidence"]}


if __name__ == "__main__":
    main()
