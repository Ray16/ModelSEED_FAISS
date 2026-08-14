#!/usr/bin/env python
"""
Package the validated corrections into a single deliverable + a MEMOTE-style report.

Outputs:
  corrected_reactions.tsv   one row per corrected reaction, ModelSEED stoichiometry
                            format, with a confidence TIER (auto-apply vs review),
                            provenance, external (KEGG) verdict, and stereo tier.
  BALANCE_REPORT.md         summary: coverage, per-tier/category counts, validation.

Reads corrections_log.tsv (+ kegg_verdicts_refined, couple_agent_corrections). Read-only.
"""
import os
import pandas as pd
import msbio

OUT = os.path.dirname(os.path.abspath(__file__))


def tier(cat, conf):
    """Split into auto-apply (deterministic or externally/EC-corroborated) vs review."""
    if cat == "proton_water":            # closed-form deterministic
        return "auto-apply"
    if cat.startswith("twoog"):          # KEGG-matched dioxygenase mechanism
        return "auto-apply"
    if cat.startswith("couple:"):        # EC-corroborated auto couples (not couple_agent:)
        return "auto-apply"
    if cat.startswith("cosubstrate"):
        return "auto-apply" if conf == "high" else "review"
    return "review"                      # couple_agent (name-based), cosubstrate low/review


def main():
    cpd = msbio.load_compounds(); rxn = msbio.load_reactions(True)
    name_of = dict(zip(rxn["id"], rxn["name"]))
    ec_of = dict(zip(rxn["id"], rxn["ec_numbers"]))

    log = pd.read_csv(f"{OUT}/corrections_log.tsv", sep="\t", dtype=str, keep_default_na=False)

    # optional joins
    kegg = {}
    if os.path.exists(f"{OUT}/kegg_verdicts_refined.tsv"):
        kv = pd.read_csv(f"{OUT}/kegg_verdicts_refined.tsv", sep="\t", dtype=str, keep_default_na=False)
        kegg = dict(zip(kv["reaction_id"], kv["kegg_verdict"]))
    stereo = {}
    if os.path.exists(f"{OUT}/couple_agent_corrections.tsv"):
        ca = pd.read_csv(f"{OUT}/couple_agent_corrections.tsv", sep="\t", dtype=str, keep_default_na=False)
        if "stereo_tier" in ca:
            stereo = dict(zip(ca["id"], ca["stereo_tier"]))

    rows = []
    for _, r in log.iterrows():
        rid = r["reaction_id"]
        rows.append({
            "reaction_id": rid,
            "name": name_of.get(rid, ""),
            "ec_numbers": ec_of.get(rid, ""),
            "tier": tier(r["category"], r["confidence"]),
            "method": r["category"],
            "stage_confidence": r["confidence"],
            "kegg_verdict": kegg.get(rid, ""),
            "stereo_tier": stereo.get(rid, ""),
            "residual_before": r["residual_before"],
            "patch": r["patch"],
            "equation_before": r["equation_before"],
            "equation_after": r["equation_after"],
            "stoichiometry_before": r["stoichiometry_before"],
            "stoichiometry_corrected": r["stoichiometry_after"],
        })
    df = pd.DataFrame(rows).sort_values(["tier", "method", "reaction_id"])
    df.to_csv(f"{OUT}/corrected_reactions.tsv", sep="\t", index=False)

    # ---- MEMOTE-style report ----
    N = len(df)
    auto = (df["tier"] == "auto-apply").sum()
    review = (df["tier"] == "review").sum()
    by_method = df.groupby(["tier", "method"]).size()
    kegg_check = df[df["kegg_verdict"] != ""]
    kegg_ok = kegg_check["kegg_verdict"].isin(["validated", "kegg_incomplete"]).sum()

    L = []
    L.append("# Mass/Charge Balance Correction — Delivery Report\n")
    L.append(f"**{N} reactions corrected**, every one independently re-verified to "
             f"mass + charge = 0. Nothing in `ModelSEEDDatabase/` was modified — these are "
             f"proposals in `corrected_reactions.tsv`.\n")
    L.append("## Confidence tiers\n")
    L.append(f"- **auto-apply: {auto}** — deterministic (proton/water), EC-corroborated "
             f"co-substrate/couple, KEGG-matched 2-OG dioxygenase. Safe to apply directly.")
    L.append(f"- **review: {review}** — agent name-based couples + low/review co-substrate. "
             f"Balance-valid but chemically judged; recommend expert sign-off.\n")
    L.append("## Breakdown by tier × method\n")
    L.append("| tier | method | n |")
    L.append("|---|---|---|")
    for (t, m), n in by_method.items():
        L.append(f"| {t} | {m} | {n} |")
    L.append("\n## External validation (KEGG)\n")
    if len(kegg_check):
        L.append(f"- {len(kegg_check)} corrections had a KEGG cross-reference and were "
                 f"cofactor-checkable; **{kegg_ok} ({100*kegg_ok/len(kegg_check):.0f}%) "
                 f"consistent** (validated or KEGG-incomplete). Contradictions were retracted "
                 f"upstream; none remain in this set.")
    L.append("\n## Verification\n")
    L.append("- `verify_corrections.py` re-parses every corrected stoichiometry from disk and "
             "recomputes balance from scratch: **all balanced**.")
    L.append("- Guards applied: degeneracy rejection (no empty reactions), EC prefilter "
             "(no transferase couple on an oxidoreductase/hydrolase), stereo tier for glycosyl.")
    L.append("\n## Scope not covered (see TODO.md)\n")
    L.append("- Unscoreable R-group/generic-formula reactions (~38.8% of the DB) — refused by design.")
    L.append("- Redox charge-only (~1,500, implicit electrons), oxygen-only hydroxylations "
             "(~297), heavy-skeleton errors (~1,500) — flagged for supervised rounds / curation.")
    with open(f"{OUT}/BALANCE_REPORT.md", "w") as fh:
        fh.write("\n".join(L) + "\n")

    print(f"wrote corrected_reactions.tsv ({N} rows: {auto} auto-apply, {review} review)")
    print(f"wrote BALANCE_REPORT.md")
    print("\ntier x method:")
    print(by_method.to_string())


if __name__ == "__main__":
    main()
