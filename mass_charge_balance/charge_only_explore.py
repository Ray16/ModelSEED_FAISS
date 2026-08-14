#!/usr/bin/env python
"""
Diagnose the 1,844 charge-only-imbalanced reactions (mass incl. H balanced, net
charge != 0). Question: are these fixable deterministically (a compound has a
wrong charge) OR are they redox reactions missing electrons (compounds are fine)?

Tests, read-only:
  1. compound charge vs structure-derived formal charge (RDKit on stored SMILES):
     if they disagree -> genuine compound defect (charge-reassignment fixable).
  2. redox signal: reaction contains a known electron carrier / O2 -> the charge
     gap is transferred electrons, not a compound defect.
  3. does |Δq| match a small electron count consistent with the redox partners?
"""
from collections import Counter

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

import msbio

# Common electron carriers / redox couples (ModelSEED cpd ids).
REDOX_CPDS = {
    "cpd00007": "O2", "cpd00003": "NAD", "cpd00004": "NADH",
    "cpd00005": "NADPH", "cpd00006": "NADP", "cpd00015": "FAD",
    "cpd00982": "FADH2", "cpd00016": "PLP", "cpd00042": "GSH",
    "cpd00023": "Glu", "cpd01270": "ferredoxin_ox", "cpd11621": "ubiquinone",
    "cpd11620": "ubiquinol", "cpd00109": "cytc_ox", "cpd00110": "cytc_red",
    "cpd15499": "ferricytochrome", "cpd00013": "NH3",
    "cpd00025": "H2O2", "cpd00011": "CO2", "cpd00418": "NO", "cpd00075": "NO2",
}


def rdkit_charge(smiles):
    if not smiles or smiles in ("null", ""):
        return None
    m = Chem.MolFromSmiles(smiles, sanitize=False)
    if m is None:
        return None
    try:
        Chem.SanitizeMol(m, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^
                         Chem.SanitizeFlags.SANITIZE_KEKULIZE)
    except Exception:
        pass
    return Chem.GetFormalCharge(m)


def main():
    cpd = msbio.load_compounds()
    rxn = msbio.load_reactions(True)
    info = msbio.SpeciesInfo.from_compounds(cpd)
    smiles_of = dict(zip(cpd["id"], cpd["smiles"]))
    name = dict(zip(cpd["id"], cpd["name"]))

    # gather charge-only reactions
    charge_only = []
    for _, r in rxn.iterrows():
        sp = msbio.parse_stoich(r["stoichiometry"])
        res, fl = msbio.compute_residual(sp, info)
        if fl["no_formula"] or fl["unknown_charge"] or not res:
            continue
        if set(res) == {"charge"}:
            charge_only.append((r["id"], r["status"], res["charge"], sp))
    print(f"charge-only reactions: {len(charge_only)}")

    # cache structure-derived charges for involved compounds
    involved = {s["cpd"] for _, _, _, sp in charge_only for s in sp}
    struct_charge = {}
    for c in involved:
        struct_charge[c] = rdkit_charge(smiles_of.get(c))

    n_has_redox = 0
    n_compound_defect = 0        # some compound: stored charge != structure charge
    n_defect_fixes_it = 0        # ...and using structure charges zeroes Δq
    n_pure_redox_no_defect = 0
    dq_abs = Counter()
    defect_cpd_freq = Counter()

    for rid, status, dq, sp in charge_only:
        dq_abs[abs(int(dq))] += 1
        has_redox = any(s["cpd"] in REDOX_CPDS for s in sp)
        n_has_redox += has_redox

        # recompute charge using structure-derived charges where available
        defect = False
        new_dq = 0.0
        resolvable = True
        for s in sp:
            stored = info.charge(s["cpd"])
            sc = struct_charge.get(s["cpd"])
            if sc is None:
                resolvable = False
                use = stored
            else:
                use = sc
                if stored is not None and sc != stored:
                    defect = True
                    defect_cpd_freq[(s["cpd"], name.get(s["cpd"], ""))] += 1
            new_dq += s["coeff"] * (use if use is not None else 0)

        if defect:
            n_compound_defect += 1
            if resolvable and abs(new_dq) < 1e-6:
                n_defect_fixes_it += 1
        elif has_redox:
            n_pure_redox_no_defect += 1

    print("\n--- |Δq| distribution ---")
    for k, v in sorted(dq_abs.items()):
        print(f"  |Δq|={k:2d} : {v}")
    print(f"\ncontains a known redox carrier / O2 : {n_has_redox}  "
          f"({100*n_has_redox/len(charge_only):.1f}%)")
    print(f"has a compound whose stored charge != RDKit structure charge : {n_compound_defect}")
    print(f"   ...and swapping to structure charges ZEROES Δq (deterministic fix) : {n_defect_fixes_it}")
    print(f"no compound defect AND contains redox carrier (needs e-/agent) : {n_pure_redox_no_defect}")

    print("\n--- most frequent charge-mismatch compounds (candidate defects) ---")
    for (c, nm), v in defect_cpd_freq.most_common(15):
        print(f"  {c} {nm[:30]:30s} stored={info.charge(c)} struct={struct_charge.get(c)}  in {v} rxns")

    # Dump candidate compound-charge defects for review (fix the compound, not the rxn).
    import pandas as pd
    rows = [{"cpd": c, "name": nm, "stored_charge": info.charge(c),
             "structure_charge": struct_charge.get(c),
             "smiles": smiles_of.get(c), "n_charge_only_rxns": v}
            for (c, nm), v in defect_cpd_freq.most_common()]
    pd.DataFrame(rows).to_csv("charge_defect_candidates.tsv", sep="\t", index=False)
    print(f"\nwrote charge_defect_candidates.tsv ({len(rows)} candidates)")


if __name__ == "__main__":
    main()
