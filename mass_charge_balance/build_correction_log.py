#!/usr/bin/env python
"""
Canonical correction log with BEFORE and AFTER for every fixed reaction.

Reads each stage's raw corrections file, reconstructs the original reaction from the
(read-only) DB, and writes a uniform record per fix so all categories are auditable
and visualizable side-by-side:

    fixes/<category>.tsv    per-category log
    corrections_log.tsv     combined master log

Columns: reaction_id, category, confidence, ec, orig_status,
         residual_before, balanced_before, residual_after, balanced_after,
         patch, equation_before, equation_after,
         stoichiometry_before, stoichiometry_after

Extend by adding an entry to STAGES. Idempotent; rebuilds from scratch each run.
"""
import os
import json
import pandas as pd

import msbio

OUT = os.path.dirname(os.path.abspath(__file__))
FIXDIR = os.path.join(OUT, "fixes")
os.makedirs(FIXDIR, exist_ok=True)

# stage file -> (category label, confidence column or None)
STAGES = [
    ("proton_water_corrections.tsv", "proton_water", None),
    ("cosubstrate_corrections.tsv",  "cosubstrate",  "confidence"),
    ("couple_corrections.tsv",       "couple",       "confidence"),
    ("twoog_corrections.tsv",        "twoog_demethylase", "confidence"),  # before couple_agent: wins on overlap
    ("couple_agent_corrections.tsv", "couple_agent", "confidence"),
    ("phospho_corrections.tsv",      "phospho_agent", "confidence"),
]


def resid_str(res):
    return ",".join(f"{k}:{int(v):+d}" for k, v in sorted(res.items())) if res else ""


def main():
    cpd = msbio.load_compounds()
    info = msbio.SpeciesInfo.from_compounds(cpd)
    name_of = dict(zip(cpd["id"], cpd["name"]))
    rxn = msbio.load_reactions(True)
    orig = {r["id"]: r for _, r in rxn.iterrows()}

    # phospho overrides: cosubstrate rows reassigned/retracted by the phospho agent
    override_ids = set()
    ov_path = os.path.join(OUT, "phospho_overrides.tsv")
    if os.path.exists(ov_path):
        ov = pd.read_csv(ov_path, sep="\t", dtype=str, keep_default_na=False)
        override_ids = set(ov["id"])
        print(f"phospho overrides (dropped from cosubstrate): {len(override_ids)}")

    # global retractions: corrections contradicted by external validation (KEGG etc.)
    global_retract = set()
    rt_path = os.path.join(OUT, "kegg_retractions.tsv")
    if os.path.exists(rt_path):
        rt = pd.read_csv(rt_path, sep="\t", dtype=str, keep_default_na=False)
        global_retract = set(rt["reaction_id"])
        print(f"global retractions (KEGG contradicts): {len(global_retract)}")

    all_rows = []
    seen = set()   # one correction per reaction; earlier (higher-priority) stage wins
    for path, category, confcol in STAGES:
        full = os.path.join(OUT, path)
        if not os.path.exists(full):
            print(f"skip missing {path}")
            continue
        try:
            df = pd.read_csv(full, sep="\t", dtype=str, keep_default_na=False)
        except pd.errors.EmptyDataError:
            print(f"skip empty {path}")
            continue
        rows = []
        for _, r in df.iterrows():
            rid = r["id"]
            # drop cosubstrate rows the phospho agent reassigned/retracted
            if category == "cosubstrate" and rid in override_ids:
                continue
            # drop any row an external validation contradicted
            if rid in global_retract:
                continue
            # one correction per reaction; a higher-priority stage already claimed it
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
            # sub-category for cosubstrate = which co-substrate
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
        cat_df = pd.DataFrame(rows)
        cat_path = os.path.join(FIXDIR, f"{category}.tsv")
        cat_df.to_csv(cat_path, sep="\t", index=False)
        all_rows.extend(rows)
        print(f"[{category}] {len(rows)} fixes -> {cat_path}")

    master = pd.DataFrame(all_rows)
    master.to_csv(os.path.join(OUT, "corrections_log.tsv"), sep="\t", index=False)

    # integrity summary
    print("\n===== correction log summary =====")
    print(f"total fixes logged: {len(master)}")
    print(f"balanced_before all False? {not master['balanced_before'].any()}  "
          f"(any already-balanced originals = bug: {int(master['balanced_before'].sum())})")
    print(f"balanced_after  all True?  {master['balanced_after'].all()}  "
          f"(any still-imbalanced = bug: {int((~master['balanced_after']).sum())})")
    print("\nby category:")
    print(master["category"].value_counts().to_string())


if __name__ == "__main__":
    main()
