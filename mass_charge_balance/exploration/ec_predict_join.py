#!/usr/bin/env python
"""
Join the rxnfp_clustering EC predictions (results/predicted_missing_ec.tsv) onto the
mass/charge-balance EC=null tractable sub-patterns, to size how much a predicted EC unlocks.

For each EC=null reaction in a tractable class (O2-hydroxylation / redox charge-only /
co-substrate menu-expansion), we ask: does the retriever give a CONFIDENT, in-domain EC3
call, and does that predicted EC CORROBORATE a specific fix (pin the reductant / redox
couple / released co-substrate)?

Read-only diagnostic. Prints counts.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from collections import Counter
import pandas as pd
from mcb import io as msbio
from mcb import closures as cl

PRED = "/nfs/lambda_stor_01/homes/rzhu/ModelSEED_FAISS/rxnfp_clustering/results/predicted_missing_ec.tsv"
O2 = "cpd00007"
# predicted-EC3 -> which fix it corroborates
O2_REDUCTANT_EC = {"1.14.13": "NADPH", "1.14.14": "FAD", "1.14.12": "NADH", "1.14.11": "2OG(twoog)"}
REDOX_ACCEPTOR_EC1 = "1"           # oxidoreductase => a redox couple is the missing piece
COSUB_EC = {  # EC3 prefix -> co-substrate it corroborates (mirror of closures.EC_CORROBORATION)
    "4.1.1": "CO2", "6.4.1": "CO2",
    "3.1.3": "Pi", "3.6.1": "Pi/PPi", "3.1.4": "Pi",
    "2.7.7": "PPi",
    "3.5.1": "NH4", "3.5.4": "NH4", "4.3.1": "NH4",
    "2.8.2": "Sulfate", "3.1.6": "Sulfate",
}


def ec3_prefix(pred_ec3):
    # pred_ec3 like "1.14.13" already; take first 3 levels
    return ".".join(str(pred_ec3).split(".")[:3])


def main():
    pred = pd.read_csv(PRED, sep="\t", dtype=str, keep_default_na=False)
    pred["conf_ec3f"] = pd.to_numeric(pred["conf_ec3"], errors="coerce").fillna(0)
    P = {r["rxn_id"]: r for _, r in pred.iterrows()}
    n_conf = int(((pred["confident"] == "True") & (pred["ood"] == "False")).sum())
    print(f"prediction table: {len(pred)} missing-EC reactions, "
          f"{n_conf} confident+in-domain EC3 calls\n")

    cpd = msbio.load_compounds(); rxn = msbio.load_reactions(True)
    info = msbio.SpeciesInfo.from_compounds(cpd)

    stats = Counter()
    unlock = Counter()   # tractable EC=null reactions that get a corroborating predicted EC
    examples = {}

    for _, r in rxn.iterrows():
        if r["ec_numbers"] not in ("", "null"):
            continue                                   # only EC=null
        sp = msbio.parse_stoich(r["stoichiometry"])
        if cl.is_refused_reaction(sp):
            continue
        res, fl = msbio.compute_residual(sp, info)
        if fl["no_formula"] or fl["unknown_charge"] or not res:
            continue
        if cl.proton_water_close(res) is not None:
            continue
        heavy = {k for k in res if k not in ("H", "O", "charge")}
        cpds = {s["cpd"] for s in sp}

        # classify tractable sub-pattern
        cls = None
        if O2 in cpds and not heavy:
            cls = "A:O2-hydroxylation"
        elif not heavy and res.get("charge") and abs(res["charge"]) % 2 == 0 \
                and not res.get("H") and not res.get("O"):
            cls = "B:redox-charge-only"
        else:
            # co-substrate menu-expansion? one stocked/extra co-substrate + p/w closes it
            for cid, (nm, formula, charge) in cl.HEAVY.items():
                if cl.single_closes(res, formula, charge) is not None:
                    cls = "C:cosubstrate-menu"; break
        if cls is None:
            continue
        stats[cls] += 1

        p = P.get(r["id"])
        if p is None:
            stats[f"{cls} | no prediction"] += 1
            continue
        confident = p["confident"] == "True" and p["ood"] == "False"
        if not confident:
            stats[f"{cls} | pred abstain/OOD"] += 1
            continue
        stats[f"{cls} | confident pred"] += 1

        # does the predicted EC corroborate a concrete fix?
        e3 = ec3_prefix(p["pred_ec3"]); e1 = str(p["pred_ec1"])
        corrob = None
        if cls.startswith("A") and e3[:7] in O2_REDUCTANT_EC:
            corrob = f"reductant={O2_REDUCTANT_EC[e3[:7]]}"
        elif cls.startswith("A") and any(e3.startswith(k) for k in O2_REDUCTANT_EC):
            corrob = "reductant(1.14.x)"
        elif cls.startswith("B") and e1 == REDOX_ACCEPTOR_EC1:
            corrob = "redox-couple(EC1=oxidoreductase)"
        elif cls.startswith("C"):
            for pre, sub in COSUB_EC.items():
                if e3.startswith(pre):
                    corrob = f"co-substrate={sub}"; break
        if corrob:
            unlock[cls] += 1
            examples.setdefault(cls, []).append(
                f"{r['id']} pred_ec3={p['pred_ec3']}(conf {p['conf_ec3']}) -> {corrob}")

    print("===== tractable EC=null reactions x prediction status =====")
    for k in sorted(stats):
        print(f"  {k:42s} {stats[k]}")
    print("\n===== UNLOCKED (confident predicted EC corroborates a concrete fix) =====")
    for k in sorted(unlock):
        print(f"  {k:24s} {unlock[k]}")
        for ex in examples[k][:4]:
            print(f"       {ex}")
    print(f"\n  >>> total EC=null reactions newly corroborated by predicted EC: {sum(unlock.values())}")


if __name__ == "__main__":
    main()
