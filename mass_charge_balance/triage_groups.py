#!/usr/bin/env python
"""
Triage the non-deterministic imbalanced reactions into PATTERN GROUPS so the agent
phase reasons once per group (then code applies the rule to every member), instead
of one LLM call per reaction.

Grouping key per reaction:
  - defect class (charge_only / oxygen_only / H_O_charge / skeleton:<elems>)
  - imbalance signature (the residual, e.g. "charge:+2" or "C:-1,O:-2")
  - redox context (which known electron carriers / O2 are present)
  - EC number top-level class (enzyme family)

Output (read-only; writes into this folder):
  agent_triage.tsv   : one row per reaction with its group keys + definition
  triage_summary.txt : groups ranked by size (highest-leverage patterns first)
"""
from collections import Counter, defaultdict

import pandas as pd

import msbio

REDOX_CPDS = {
    "cpd00007": "O2", "cpd00003": "NAD", "cpd00004": "NADH",
    "cpd00005": "NADPH", "cpd00006": "NADP", "cpd00015": "FAD",
    "cpd00982": "FADH2", "cpd11621": "UQ", "cpd11620": "UQH2",
    "cpd00109": "cytc_ox", "cpd00110": "cytc_red", "cpd00025": "H2O2",
    "cpd00418": "NO", "cpd00528": "N2", "cpd00075": "NO2", "cpd00013": "NH3",
}


def sig_str(residual):
    return ",".join(f"{k}:{int(v):+d}" for k, v in sorted(residual.items()))


def defect_class(residual):
    keys = set(residual) - {"charge"}
    heavy = keys - {"H", "O"}
    if heavy:
        return "skeleton:" + ",".join(sorted(heavy))
    if set(residual) == {"charge"}:
        return "charge_only"
    if set(residual) == {"O"}:
        return "oxygen_only"
    return "H_O_charge"


def main():
    cpd = msbio.load_compounds()
    rxn = msbio.load_reactions(True)
    info = msbio.SpeciesInfo.from_compounds(cpd)

    rows = []
    for _, r in rxn.iterrows():
        sp = msbio.parse_stoich(r["stoichiometry"])
        residual, flags = msbio.compute_residual(sp, info)
        if flags["no_formula"] or flags["unknown_charge"] or not residual:
            continue
        # skip the deterministically-solved proton/water class
        dH = residual.get("H", 0); dO = residual.get("O", 0); dq = residual.get("charge", 0)
        heavy = set(residual) - {"H", "O", "charge"}
        if not heavy and abs(dH + 2 * (-round(dO)) + (-round(dq))) < 1e-6:
            continue  # proton/water-fixable -> already handled

        carriers = sorted({REDOX_CPDS[s["cpd"]] for s in sp if s["cpd"] in REDOX_CPDS})
        ec = r["ec_numbers"].split(".")[0] if r["ec_numbers"] not in ("", "null") else ""
        rows.append({
            "id": r["id"],
            "class": defect_class(residual),
            "signature": sig_str(residual),
            "redox": "+".join(carriers),
            "ec_class": ec,
            "n_species": len(sp),
            "definition": r["definition"][:120],
        })

    df = pd.DataFrame(rows)
    df.to_csv("agent_triage.tsv", sep="\t", index=False)

    lines = [f"NON-DETERMINISTIC IMBALANCED REACTIONS: {len(df)}\n"]
    lines.append("=== by defect class ===")
    for k, v in df["class"].value_counts().items():
        lines.append(f"  {k:22s} {v}")

    lines.append("\n=== largest (class, signature, redox) pattern groups — top 25 ===")
    grp = df.groupby(["class", "signature", "redox"]).size().sort_values(ascending=False)
    for (cls, sig, redox), n in grp.head(25).items():
        lines.append(f"  {n:5d}  {cls:16s} [{sig:14s}] redox={redox or '-'}")

    lines.append("\n=== redox involvement ===")
    has_redox = (df["redox"] != "").sum()
    lines.append(f"  reactions with a known redox carrier: {has_redox} ({100*has_redox/len(df):.1f}%)")

    lines.append("\n=== by EC top-level class (1=oxidoreductase) ===")
    for k, v in df["ec_class"].value_counts().items():
        lines.append(f"  EC {k or '(none)'}: {v}")

    report = "\n".join(lines)
    with open("triage_summary.txt", "w") as fh:
        fh.write(report + "\n")
    print(report)
    print(f"\nwrote agent_triage.tsv ({len(df)} rows) + triage_summary.txt")


if __name__ == "__main__":
    main()
