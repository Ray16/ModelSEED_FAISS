#!/usr/bin/env python
"""
Regression + unit tests for the mass/charge balance corrector.

Runnable two ways:
    pytest tests/                 # if pytest is installed
    python tests/test_regression.py   # plain-stdlib fallback runner

Guards against silent regressions in a *validated* deliverable:
  * the delivered corrected_reactions.tsv still holds the exact golden 516 reactions,
  * every one still balances to mass+charge = 0 and is non-degenerate,
  * no indefinite-polymer or added-phosphate-mechanism-conflict correction slips back in,
  * the closure/guard primitives behave as specified.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from mcb import io as msbio
from mcb import closures as cl
from mcb.paths import wp

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, "golden_corrected_reactions.tsv")
REFUSED = {"rxn05757", "rxn15388", "rxn15000", "rxn49477", "rxn03028"}

_info = None


def info():
    global _info
    if _info is None:
        _info = msbio.SpeciesInfo.from_compounds(msbio.load_compounds())
    return _info


# --------------------------------------------------------------------------- #
# Unit tests: closure arithmetic + guards (no DB needed)
# --------------------------------------------------------------------------- #
def test_proton_water_close():
    # a whole water missing (H2O = 2H + O): residual O:-1, H:-2 -> add 1 water
    assert cl.proton_water_close({"O": -1, "H": -2}) == (1, 0)
    # a whole proton missing (H+ = H, +1): residual charge:-1, H:-1 -> add 1 proton
    assert cl.proton_water_close({"charge": -1, "H": -1}) == (0, 1)
    # a bare O deficit CANNOT close: water would inject 2 unbalanced H
    assert cl.proton_water_close({"O": -1}) is None
    # a bare charge deficit CANNOT close: proton would inject 1 unbalanced H
    assert cl.proton_water_close({"charge": -1}) is None
    assert cl.proton_water_close({"C": 1}) is None             # heavy element -> not closeable


def test_generic_species_guard():
    assert cl.is_generic_species("(Phosphate)n")
    assert cl.is_generic_species("Polyphosphate")
    assert cl.is_generic_species("Oligophosphate")
    assert cl.is_generic_species("(Glucose)n")                # generalizes to any (X)n
    # a DEFINED structure that merely contains "oligosaccharide" must NOT be refused
    assert not cl.is_generic_species("core oligosaccharide-lipid A (E. coli K-12 core type)")
    assert not cl.is_generic_species("D-Glucose")


def test_phosphate_mechanism_guard():
    atp_adp = {"cpd00002", "cpd00008"}       # ATP -> ADP releases Pi, not PPi
    atp_amp = {"cpd00002", "cpd00018"}       # ATP -> AMP releases PPi, not Pi
    assert cl.phosphate_cosubstrate_conflict(atp_adp, cl.CPD_PPI)      # adding PPi is wrong
    assert not cl.phosphate_cosubstrate_conflict(atp_adp, cl.CPD_PI)   # adding Pi is fine
    assert cl.phosphate_cosubstrate_conflict(atp_amp, cl.CPD_PI)       # adding Pi is wrong
    assert not cl.phosphate_cosubstrate_conflict(atp_amp, cl.CPD_PPI)  # adding PPi is fine
    assert not cl.phosphate_cosubstrate_conflict(set(), cl.CPD_PPI)    # no couple -> no rule


def test_ec_prefilter():
    assert cl.ec_compatible("null", "methyl (SAM/SAH)")            # null EC always allowed
    assert cl.ec_compatible("2.1.1.1", "methyl (SAM/SAH)")         # transferase ok
    assert not cl.ec_compatible("1.14.11.31", "methyl (SAM/SAH)")  # oxidoreductase blocked


# --------------------------------------------------------------------------- #
# Regression tests: the delivered corrections file
# --------------------------------------------------------------------------- #
def test_matches_golden():
    cur = pd.read_csv(wp("corrected_reactions.tsv"), sep="\t", dtype=str, keep_default_na=False)
    gold = pd.read_csv(GOLDEN, sep="\t", dtype=str, keep_default_na=False)
    assert len(cur) == len(gold) == 516, (len(cur), len(gold))
    key = ["reaction_id", "stoichiometry_corrected"]
    a = cur[key].sort_values(key).reset_index(drop=True)
    b = gold[key].sort_values(key).reset_index(drop=True)
    assert a.equals(b), "delivered corrections drifted from golden"


def test_all_balanced_and_nondegenerate():
    df = pd.read_csv(wp("corrected_reactions.tsv"), sep="\t", dtype=str, keep_default_na=False)
    bad = []
    for _, r in df.iterrows():
        sp = msbio.parse_stoich(r["stoichiometry_corrected"])
        res, fl = msbio.compute_residual(sp, info())
        if res or fl["no_formula"] or fl["unknown_charge"] or not msbio.is_valid_reaction(sp):
            bad.append(r["reaction_id"])
    assert not bad, f"{len(bad)} corrections fail balance/degeneracy: {bad[:10]}"


def test_no_guard_violations():
    df = pd.read_csv(wp("corrected_reactions.tsv"), sep="\t", dtype=str, keep_default_na=False)
    assert not (REFUSED & set(df["reaction_id"])), "a refused reaction reappeared"
    for _, r in df.iterrows():
        after = msbio.parse_stoich(r["stoichiometry_corrected"])
        before = msbio.parse_stoich(r["stoichiometry_before"])
        assert not cl.is_refused_reaction(after), r["reaction_id"]
        added = {s["cpd"] for s in after} - {s["cpd"] for s in before}
        acpds = {s["cpd"] for s in after}
        for pc in (cl.CPD_PI, cl.CPD_PPI):
            if pc in added:
                assert not cl.phosphate_cosubstrate_conflict(acpds - {pc}, pc), r["reaction_id"]


def _main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
