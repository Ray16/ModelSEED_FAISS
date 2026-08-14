#!/usr/bin/env python
"""
Explore the mass/charge imbalance landscape of the ModelSEED reaction database.

We DO NOT trust the stored `status` field blindly -- we recompute element and
charge balance from the compound formulas/charges, and cross-check against the
stored status. Then we classify every non-OK reaction into actionable buckets.

Output: prints a summary table and writes landscape.tsv (per-reaction diagnosis).
"""
import glob
import os
import re
import json
from collections import Counter, defaultdict

import pandas as pd

BIOCHEM = "/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/ModelSEEDDatabase/Biochemistry"
OUT = "/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/mass_charge_balance"

UNKNOWN_SENTINELS = {10000000, 10000001, 10000002, 9999999, 9999998,
                     -10000000, -10000001, -10000002, -9999999}

# ---------------------------------------------------------------------------
# Load compounds
# ---------------------------------------------------------------------------
def load_compounds():
    frames = []
    for f in sorted(glob.glob(os.path.join(BIOCHEM, "compound_*.tsv"))):
        frames.append(pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False))
    cpd = pd.concat(frames, ignore_index=True)
    return cpd

def load_reactions():
    frames = []
    for f in sorted(glob.glob(os.path.join(BIOCHEM, "reaction_*.tsv"))):
        frames.append(pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False))
    rxn = pd.concat(frames, ignore_index=True)
    return rxn

# ---------------------------------------------------------------------------
# Formula parsing
# ---------------------------------------------------------------------------
FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")

def parse_formula(formula):
    """Return dict element->count. Returns None if formula is empty/generic.

    Generic = contains R, X, *, ), or 'null'/empty -> cannot be balanced by mass.
    """
    if formula is None:
        return None
    f = formula.strip()
    if f in ("", "null", "noformula"):
        return None
    # Generic polymer / R-group markers
    if any(sym in f for sym in ("R", "X", "*", "(", ")", "n")):
        # 'n' as subscript variable also indicates polymer; but note real
        # elements don't use lowercase alone here except within [A-Z][a-z].
        # We flag anything with these tokens as generic.
        # Careful: 'Na', 'Ni', 'Sn' contain n/... handle below.
        pass
    counts = defaultdict(int)
    pos = 0
    generic = False
    # tokenize handling R-groups: treat R, X as generic
    for m in re.finditer(r"([A-Z][a-z]?|\*|\(|\)|\.)(\d*)", f):
        sym = m.group(1)
        num = m.group(2)
        n = int(num) if num else 1
        if sym in ("*", "(", ")", "."):
            generic = True
            continue
        if sym in ("R", "X"):
            generic = True
            continue
        counts[sym] += n
    if generic:
        return None
    # detect leftover characters not consumed (e.g. lowercase 'n')
    consumed = "".join(m.group(0) for m in re.finditer(r"([A-Z][a-z]?|\*|\(|\)|\.)(\d*)", f))
    # crude leftover check
    stripped = re.sub(r"[A-Z][a-z]?\d*|[\*\(\)\.]\d*", "", f)
    if stripped.strip():
        return None
    return dict(counts)

# ---------------------------------------------------------------------------
# Stoichiometry parsing:  n:cpdid:cmpt:comm:"name"  separated by ;
# ---------------------------------------------------------------------------
def parse_stoich(s):
    out = []
    if not s or s == "null":
        return out
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        fields = part.split(":")
        try:
            coeff = float(fields[0])
        except ValueError:
            continue
        cpdid = fields[1]
        out.append((coeff, cpdid))
    return out

def main():
    print("Loading compounds & reactions ...")
    cpd = load_compounds()
    rxn = load_reactions()
    print(f"  compounds: {len(cpd)}   reactions: {len(rxn)}")

    # compound lookup
    formula_of = dict(zip(cpd["id"], cpd["formula"]))
    charge_of = {}
    for cid, ch in zip(cpd["id"], cpd["charge"]):
        try:
            charge_of[cid] = int(float(ch))
        except (ValueError, TypeError):
            charge_of[cid] = None

    parsed_formula = {cid: parse_formula(f) for cid, f in formula_of.items()}

    # Only consider non-obsolete reactions
    rxn_active = rxn[rxn["is_obsolete"].isin(["0", "false", "False", ""])].copy()
    print(f"  active (non-obsolete) reactions: {len(rxn_active)}")

    rows = []
    recompute_mismatch = 0
    for _, r in rxn_active.iterrows():
        rid = r["id"]
        stoich = parse_stoich(r["stoichiometry"])
        status = r["status"]

        # recompute mass & charge balance
        elem_bal = defaultdict(float)
        charge_bal = 0.0
        has_formula_err = False
        has_charge_err = False
        n_generic = 0
        for coeff, cid in stoich:
            pf = parsed_formula.get(cid)
            if pf is None:
                has_formula_err = True
                n_generic += 1
            else:
                for el, cnt in pf.items():
                    elem_bal[el] += coeff * cnt
            ch = charge_of.get(cid)
            if ch is None or ch in UNKNOWN_SENTINELS:
                has_charge_err = True
            else:
                charge_bal += coeff * ch

        elem_imbalance = {el: v for el, v in elem_bal.items() if abs(v) > 1e-6}
        charge_imbalance = charge_bal if abs(charge_bal) > 1e-6 else 0.0

        # Classify
        if has_formula_err:
            bucket = "A_no_formula"          # generic/R-group/missing formula
        elif has_charge_err:
            bucket = "B_unknown_charge"      # formula ok but charge sentinel/missing
        elif not elem_imbalance and charge_imbalance == 0:
            bucket = "OK"
        else:
            # what elements are off (excluding H)?
            heavy_off = {el: v for el, v in elem_imbalance.items() if el != "H"}
            if not heavy_off and "H" in elem_imbalance:
                bucket = "C_H_only"          # only hydrogen off -> proton/water fixable
            elif set(heavy_off) <= {"O"} and "H" in elem_imbalance:
                bucket = "D_H_and_O"         # water fixable
            elif not heavy_off and not elem_imbalance and charge_imbalance != 0:
                bucket = "E_charge_only"     # mass ok, charge off -> protonation state
            else:
                bucket = "F_heavy_imbalance" # genuine skeleton imbalance

        # refine: pure charge-only when mass fully balanced
        if bucket not in ("A_no_formula", "B_unknown_charge") and not elem_imbalance and charge_imbalance != 0:
            bucket = "E_charge_only"

        rows.append({
            "id": rid,
            "status": status,
            "bucket": bucket,
            "n_species": len(stoich),
            "n_generic": n_generic,
            "elem_imbalance": json.dumps(elem_imbalance, sort_keys=True) if elem_imbalance else "",
            "charge_imbalance": charge_imbalance,
            "is_transport": r["is_transport"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "landscape.tsv"), sep="\t", index=False)

    print("\n===== BUCKET BREAKDOWN (recomputed, active reactions) =====")
    tot = len(df)
    bc = df["bucket"].value_counts()
    for b, c in bc.items():
        print(f"  {b:22s} {c:7d}  {100*c/tot:5.1f}%")
    print(f"  {'TOTAL':22s} {tot:7d}")

    imbalanced = df[df["bucket"] != "OK"]
    print(f"\n  imbalanced (non-OK): {len(imbalanced)}  ({100*len(imbalanced)/tot:.1f}%)")

    # Cross-check vs stored status
    print("\n===== CROSS-CHECK vs stored `status` =====")
    stored_ok = df["status"].str.startswith("OK")
    recomp_ok = df["bucket"] == "OK"
    agree = (stored_ok == recomp_ok).mean()
    print(f"  stored-OK vs recomputed-OK agreement: {100*agree:.1f}%")
    print(f"  stored OK count: {stored_ok.sum()}   recomputed OK: {recomp_ok.sum()}")

    # Among F_heavy_imbalance, what elements dominate?
    print("\n===== F_heavy_imbalance: top element signatures =====")
    fh = df[df["bucket"] == "F_heavy_imbalance"]
    sig = Counter()
    for e in fh["elem_imbalance"]:
        d = json.loads(e)
        sig[tuple(sorted(set(d) - {"H"}))] += 1
    for k, v in sig.most_common(20):
        print(f"  {','.join(k) if k else '(H only?)':20s} {v}")

if __name__ == "__main__":
    main()
