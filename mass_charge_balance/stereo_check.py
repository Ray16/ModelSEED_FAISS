#!/usr/bin/env python
"""
Targeted structure/stereo consistency check for GLYCOSYL corrections only.

Mass+charge balance is stereo-blind: it confirms "a hexose is missing" but not which
sugar, because glucose/mannose/galactose share formula C6H10O5. This adds a
structure-level check on the reaction's own substrate/product pair (not on the added
cofactor) to grade how verifiable the sugar call is:

  scaffold+hexose : the product = substrate scaffold + a hexose-sized group
                    (substrate is a substructure of product) -> it IS a glycosylation
                    of that scaffold. Sub-grade by whether product sugar stereo is
                    fully defined:
                      structure-stereo-defined  (epimer identity is checkable)
                      structure-stereo-undefined (glucose/mannose call NOT checkable
                                                   from structure -> rests on the name)
  unverifiable     : no clean C6H10O5 substrate/product pair, or structures missing.

This never blocks a correction; it only tags confidence for review.
"""
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

HEXOSE = {"C": 6, "H": 10, "O": 5}
# cofactors to exclude when locating the organic substrate/product pair
COFACTORS = {"cpd00001", "cpd00067", "cpd00002", "cpd00008", "cpd00018",
             "cpd00026", "cpd00014", "cpd00083", "cpd00031", "cpd00012", "cpd00009"}


def _mol(smiles):
    if not smiles or smiles in ("null", ""):
        return None
    m = Chem.MolFromSmiles(smiles, sanitize=False)
    if m is None:
        return None
    try:
        Chem.SanitizeMol(m, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^
                         Chem.SanitizeFlags.SANITIZE_KEKULIZE)
    except Exception:
        return None
    return m


def _formula_diff(fa, fb):
    keys = set(fa) | set(fb)
    return {k: fa.get(k, 0) - fb.get(k, 0) for k in keys}


def glycosyl_confidence(orig_species, info, smiles_of):
    """Return (tier, detail). orig_species is the ORIGINAL (pre-patch) species list."""
    organic = [s for s in orig_species
               if s["cpd"] not in COFACTORS and (info.formula(s["cpd"]) or {}).get("C", 0) > 0]
    reactants = [s for s in organic if s["coeff"] < 0]
    products = [s for s in organic if s["coeff"] > 0]

    # find a substrate/product pair differing by exactly one hexose
    for small_side, big_side in ((reactants, products), (products, reactants)):
        for s in small_side:
            fs = info.formula(s["cpd"])
            if not fs:
                continue
            for p in big_side:
                fp = info.formula(p["cpd"])
                if not fp:
                    continue
                diff = _formula_diff(fp, fs)
                if all(diff.get(k, 0) == v for k, v in HEXOSE.items()) and \
                   all(diff.get(k, 0) == 0 for k in diff if k not in HEXOSE):
                    # big = small + hexose. structural check:
                    ms, mp = _mol(smiles_of.get(s["cpd"])), _mol(smiles_of.get(p["cpd"]))
                    if ms is None or mp is None:
                        return "unverifiable", f"{p['cpd']}=+hexose over {s['cpd']} but structure missing"
                    if not mp.HasSubstructMatch(ms, useChirality=False):
                        return "unverifiable", f"{s['cpd']} not a substructure of {p['cpd']}"
                    # scaffold+hexose confirmed; is the product sugar stereo defined?
                    stereo = Chem.FindMolChiralCenters(mp, includeUnassigned=True, useLegacyImplementation=False)
                    unassigned = [c for c, lab in stereo if lab == "?"]
                    if unassigned:
                        return "structure-stereo-undefined", \
                            f"{p['cpd']} scaffold+hexose ok; {len(unassigned)} undefined stereocentre(s) -> sugar id rests on name"
                    return "structure-stereo-defined", \
                        f"{p['cpd']} scaffold+hexose ok; stereo fully defined"
    return "unverifiable", "no clean substrate/product pair differing by one hexose"
