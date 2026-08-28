#!/usr/bin/env python
"""
Stage 7 - canonical correction log with BEFORE and AFTER for every fixed reaction.

Reads each stage's raw corrections file, reconstructs the original reaction from the
(read-only) DB, and writes a uniform record per fix so all categories are auditable and
visualizable side-by-side. Applies phospho overrides (drop reassigned/retracted cosubstrate
rows) and any external retractions (kegg_retractions.tsv), and keeps one correction per
reaction (earlier, higher-priority stage wins -- twoog before couple_agent).

Outputs (work dir): fixes/<category>.tsv  and  corrections_log.tsv
"""
import os

import pandas as pd

from mcb import io as msbio
from mcb.paths import work, inp, result, WORK

FIXDIR = os.path.join(str(WORK), "fixes")

# stage file -> (category label, confidence column or None). Order = priority on overlap.
STAGES = [
    ("proton_water_corrections.tsv", "proton_water", None),
    ("cosubstrate_corrections.tsv",  "cosubstrate",  "confidence"),
    ("couple_corrections.tsv",       "couple",       "confidence"),
    ("twoog_corrections.tsv",        "twoog_demethylase", "confidence"),
    ("couple_agent_corrections.tsv", "couple_agent", "confidence"),
    ("phospho_corrections.tsv",      "phospho_agent", "confidence"),
]


def resid_str(res):
    return ",".join(f"{k}:{int(v):+d}" for k, v in sorted(res.items())) if res else ""


def main():
    os.makedirs(FIXDIR, exist_ok=True)
    cpd = msbio.load_compounds()
    info = msbio.SpeciesInfo.from_compounds(cpd)
    name_of = dict(zip(cpd["id"], cpd["name"]))
    rxn = msbio.load_reactions(True)
    orig = {r["id"]: r for _, r in rxn.iterrows()}

    override_ids = set()
    if os.path.exists(work("phospho_overrides.tsv")):
        ov = pd.read_csv(work("phospho_overrides.tsv"), sep="\t", dtype=str, keep_default_na=False)
        override_ids = set(ov["id"])
        print(f"phospho overrides (dropped from cosubstrate): {len(override_ids)}")

    global_retract = set()
    if os.path.exists(inp("kegg_retractions.tsv")):
        rt = pd.read_csv(inp("kegg_retractions.tsv"), sep="\t", dtype=str, keep_default_na=False)
        global_retract = set(rt["reaction_id"])
        print(f"global retractions (KEGG contradicts): {len(global_retract)}")

    all_rows = []
    seen = set()
    for path, category, confcol in STAGES:
        if not os.path.exists(work(path)):
            print(f"skip missing {path}")
            continue
        try:
            df = pd.read_csv(work(path), sep="\t", dtype=str, keep_default_na=False)
        except pd.errors.EmptyDataError:
            print(f"skip empty {path}")
            continue
        rows = []
        for _, r in df.iterrows():
            rid = r["id"]
            if category == "cosubstrate" and rid in override_ids:
                continue
            if rid in global_retract:
                continue
            if rid in seen:
                continue
            seen.add(rid)
            o = orig.get(rid)
            if o is None:
                continue
            before_sp = msbio.parse_stoich(o["stoichiometry"])
            after_sp = msbio.parse_stoich(r["corrected_stoichiometry"])
            rb, fb = msbio.compute_residual(before_sp, info)
            ra, fa = msbio.compute_residual(after_sp, info)
            subcat = category
            if "co_substrate" in r and r["co_substrate"]:
                subcat = f"{category}:{r['co_substrate']}"
            elif "couple" in r and r["couple"]:
                subcat = f"{category}:{r['couple'].split()[0]}"
            rows.append({
                "reaction_id": rid,
                "category": subcat,
                "confidence": r[confcol] if confcol and confcol in r else "deterministic",
                "ec": o["ec_numbers"],
                "orig_status": o["status"],
                "residual_before": resid_str(rb),
                "balanced_before": bool(not rb and not fb["no_formula"] and not fb["unknown_charge"]),
                "residual_after": resid_str(ra),
                "balanced_after": bool(not ra and not fa["no_formula"] and not fa["unknown_charge"]),
                "patch": r.get("patch", ""),
                "equation_before": msbio.render_equation(before_sp, name_of),
                "equation_after": msbio.render_equation(after_sp, name_of),
                "stoichiometry_before": o["stoichiometry"],
                "stoichiometry_after": r["corrected_stoichiometry"],
            })
        pd.DataFrame(rows).to_csv(os.path.join(FIXDIR, f"{category}.tsv"), sep="\t", index=False)
        all_rows.extend(rows)
        print(f"[{category}] {len(rows)} fixes")

    master = pd.DataFrame(all_rows)
    master.to_csv(result("corrections_log.tsv"), sep="\t", index=False)

    print("\n===== correction log summary =====")
    print(f"total fixes logged: {len(master)}")
    print(f"any already-balanced originals (=bug): {int(master['balanced_before'].sum())}")
    print(f"any still-imbalanced after (=bug):     {int((~master['balanced_after']).sum())}")
    print("\nby category:")
    print(master["category"].value_counts().to_string())


if __name__ == "__main__":
    main()
