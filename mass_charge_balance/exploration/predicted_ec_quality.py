#!/usr/bin/env python
"""
Quality check for the predicted-EC corrections: is a fix chemically plausible, not just
balanced? The strongest network-free signal is the retriever's own nearest labelled
analogue — a real, EC-labelled, balanced reaction. If that analogue actually CONTAINS the
species we added (the reductant / co-substrate), our fix matches a real precedent.

For each predicted-EC correction we report:
  - retrieval similarity to the analogue (higher = more trustworthy prediction),
  - whether the analogue reaction contains the same balancing species we injected,
  - predicted EC1 consistency with the fix class (O2->oxidoreductase, etc.).
Prints a pass rate; these stay review-only regardless.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from collections import Counter
import pandas as pd
from mcb import io as msbio
from mcb.paths import result

# species a fix injects, by the balancing group named in the patch
ADDED_CPDS = {
    "NADPH": {"cpd00005", "cpd00006"}, "NADH": {"cpd00004", "cpd00003"},
    "FADH2": {"cpd00982", "cpd00015"},
    "CO2": {"cpd00011"}, "Pi": {"cpd00009"}, "PPi": {"cpd00012"},
    "NH4": {"cpd00013"}, "sulfate": {"cpd00048"}, "Sulfate": {"cpd00048"},
}


def added_group(patch):
    for g in ADDED_CPDS:
        if g in patch:
            return g
    return None


def main():
    rxn = msbio.load_reactions(True)
    stoich_of = dict(zip(rxn["id"], rxn["stoichiometry"]))
    df = pd.read_csv(result("predicted_ec_corrections.tsv"), sep="\t", dtype=str, keep_default_na=False)
    if not len(df):
        print("no predicted-EC corrections to assess"); return

    rows = []
    for _, r in df.iterrows():
        grp = added_group(r["patch"])
        want = ADDED_CPDS.get(grp, set())
        analogue = stoich_of.get(r["analogue_rxn"], "")
        acpds = {s["cpd"] for s in msbio.parse_stoich(analogue)}
        analogue_has = bool(want & acpds)                       # analogue uses same species
        ec1_ok = (r["class"] == "O2-monooxygenase" and str(r["pred_ec3"]).startswith("1")) or \
                 (r["class"] == "cosubstrate")
        rows.append({**r.to_dict(), "added_group": grp,
                     "analogue_has_same_species": analogue_has, "ec1_consistent": ec1_ok})

    q = pd.DataFrame(rows)
    n = len(q)
    print(f"===== predicted-EC quality ({n} corrections, all review-predicted) =====\n")
    print("by class:", q["class"].value_counts().to_dict())
    print(f"\nanalogue reaction uses the SAME balancing species we added: "
          f"{int(q['analogue_has_same_species'].sum())}/{n} "
          f"({q['analogue_has_same_species'].mean()*100:.0f}%)")
    print(f"predicted EC1 consistent with fix class:                    "
          f"{int(q['ec1_consistent'].sum())}/{n}")
    # confidence / similarity distribution
    q["pred_conf"] = pd.to_numeric(q["pred_conf"], errors="coerce")
    print(f"\npred_ec3 confidence: min {q['pred_conf'].min():.2f}  "
          f"median {q['pred_conf'].median():.2f}  max {q['pred_conf'].max():.2f}")
    strong = q[q["analogue_has_same_species"]]
    print(f"\n--- {len(strong)} STRONG (analogue precedent confirms the added species) ---")
    for _, r in strong.iterrows():
        print(f"  {r['reaction_id']} [{r['class']}] +{r['added_group']}  "
              f"pred_ec3={r['pred_ec3']}(conf {r['pred_conf']:.2f})  "
              f"analogue={r['analogue_rxn']}(EC {r['analogue_ec']})")
    weak = q[~q["analogue_has_same_species"]]
    if len(weak):
        print(f"\n--- {len(weak)} WEAKER (no analogue precedent for the added species — extra scrutiny) ---")
        for _, r in weak.iterrows():
            print(f"  {r['reaction_id']} [{r['class']}] +{r['added_group']}  "
                  f"pred_ec3={r['pred_ec3']}  analogue={r['analogue_rxn']}(EC {r['analogue_ec']})")

    q.to_csv(result("predicted_ec_quality.tsv"), sep="\t", index=False)
    print(f"\nwrote results/predicted_ec_quality.tsv")


if __name__ == "__main__":
    main()
