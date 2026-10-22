#!/usr/bin/env python
"""Curated positive/negative controls for the mechanism screens.

Run after any change to _acylation.py or _normalize.py:
    python scripts/test_screens.py

Every case here came from a concrete failure, so each one documents a bug that
is now guarded. Cases are written as explicit SMILES wherever possible so the
suite does not depend on a rebuilt corpus.
"""
from __future__ import annotations

import sys
from pathlib import Path

from rdkit import Chem, RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from amide_synthesis.acylation import acylated_acceptors, count_free_amines  # noqa: E402
from amide_synthesis.normalize import normalize_smiles  # noqa: E402

RDLogger.DisableLog('rdApp.*')

# ---------------------------------------------------------------- structure repair
# The 2026-10-06 refresh writes amides as iminols, in BOTH an anionic and a
# neutral form. Repairing only the anionic form left 665 compounds broken,
# including N-acetylputrescine - which silently removed the single most relevant
# positive control (acetyl-CoA:putrescine N-acetyltransferase) from the results.
REPAIR_CASES = [
    ('anionic iminolate (CoA-like)', 'CC([O-])=NCCS', 'CC(=O)NCCS'),
    ('neutral iminol (N-acetylputrescine)', 'CC(O)=NCCCC[NH3+]', 'CC(=O)NCCCC[NH3+]'),
    ('neutral iminol (nicotinamide)', 'OC(=N)c1cccnc1', 'NC(=O)c1cccnc1'),
    ('already correct amide is untouched', 'CC(=O)NCCS', 'CC(=O)NCCS'),
]

# ---------------------------------------------------------------- amine classifier
# "Two nitrogens" wrongly called all of these diamines.
AMINE_CASES = [
    ('hexamethylenediamine', '[NH3+]CCCCCC[NH3+]', 2),
    ('putrescine', '[NH3+]CCCC[NH3+]', 2),
    ('spermidine', '[NH3+]CCCC[NH2+]CCC[NH3+]', 3),
    ('serotonin (aromatic ring N is not an amine)', 'NCCc1c[nH]c2ccc(O)cc12', 1),
    ('adenine (aromatic amine cannot attack)', 'Nc1ncnc2[nH]cnc12', 0),
    ('arginine (guanidino N excluded)', 'N[C@@H](CCCNC(=N)N)C(=O)[O-]', 1),
    ('ethylimidazole', 'CCn1ccnc1', 0),
    ('beta-aminopropiononitrile (nitrile N excluded)', 'NCCC#N', 1),
]

# ---------------------------------------------------------------- acylation screen
# POSITIVES must be retrieved; NEGATIVES previously passed a weaker screen.
ACYLATION_CASES = [
    ('acetyl-CoA + putrescine -> N-acetylputrescine',
     'CC(=O)SCCNC(C)=O.[NH3+]CCCC[NH3+]>>CC(=O)NCCCC[NH3+].SCCNC(C)=O',
     True, 2),
    ('citrate + spermidine (AsbA-like)',
     'OC(=O)CC(O)(CC(=O)[O-])C(=O)[O-].[NH3+]CCCC[NH2+]CCC[NH3+]'
     '>>OC(=O)CC(O)(CC(=O)[O-])C(=O)NCCCC[NH2+]CCC[NH3+].O',
     True, 3),
    ('alanine dehydrogenase: amine lost, unrelated amide present',
     'C[C@H]([NH3+])C(=O)[O-].O>>CC(=O)C(=O)[O-].[NH4+]',
     False, 0),
    ('asparagine synthetase: N arrives already amide-bound',
     'NC(=O)CC[C@H]([NH3+])C(=O)[O-].[O-]C(=O)C[C@H]([NH3+])C(=O)[O-]'
     '>>[O-]C(=O)CC[C@H]([NH3+])C(=O)[O-].NC(=O)C[C@H]([NH3+])C(=O)[O-]',
     False, 0),
    ('lysine donor acylates ethylamine: acceptor is the MONOamine',
     '[NH3+]CCCC[C@H]([NH3+])C(=O)[O-].CC[NH3+]'
     '>>[NH3+]CCCC[C@H]([NH3+])C(=O)NCC.O',
     True, 1),
    ('BOTH ends of HMDA acetylated (needs a 2-bond cut)',
     'CC(=O)SC.CC(=O)SC.[NH3+]CCCCCC[NH3+]'
     '>>CC(=O)NCCCCCCNC(C)=O.CS.CS',
     True, 2),
    ('guanidino N acylated while an aliphatic amine sits elsewhere',
     'CC(=O)SC.N[C@@H](CCCNC(=N)N)C(=O)[O-]'
     '>>CC(=O)NC(=N)NCCC[C@H](N)C(=O)[O-].CS',
     False, 0),
    ('hydrolysis of an amide is not formation',
     'CC(=O)NCCCC[NH3+].O>>CC(=O)[O-].[NH3+]CCCC[NH3+]',
     False, 0),
]


# --- retrieval-layer controls (added with the 2026-10-09 refactor) ---
# These guard the reaction -> enzyme mapping, not the chemistry screens. Each
# one is a bug that was live in the shipped output before being fixed.
PRODUCT_NAME_CASES = [
    # (name, ModelSEED definition, expected product_names())
    ('pre- prefix is also tried stripped: UniProt names the MATURE product, so '
     'without this rxn21311 (alcaligin) returns nothing',
     '(1) ATP[0] + (2) foo[0] <=> (1) PPi[0] + (1) AMP[0] + (1) pre-alcaligin[0]',
     ['pre-alcaligin', 'alcaligin']),
    ('cofactors are dropped: searching UniProt for "ATP" is meaningless and '
     'every reaction would otherwise emit it',
     '(1) ATP[0] + (1) foo[0] <=> (1) ADP[0] + (1) Phosphate[0] + (1) acinetoferrin[0]',
     ['acinetoferrin']),
    ('short names are dropped as insufficiently distinctive',
     '(1) ATP[0] <=> (1) urea[0]',
     []),
    ('only the PRODUCT side is read; a distinctive reactant must not leak in',
     '(1) bisucaberin[0] + (1) H2O[0] <=> (1) Phosphate[0]',
     []),
]


def check_product_names():
    """Route 3 feeds these strings straight to UniProt; a bad one is a bad query."""
    failures = []
    print('\nproduct-name extraction (enzyme-retrieval route 3)')
    for name, definition, expected in PRODUCT_NAME_CASES:
        got = product_names(definition)
        ok = got == expected
        print(f'  {"PASS" if ok else "FAIL"}  {name}')
        if not ok:
            failures.append(f'{name}: {got} != {expected}')
    return failures


from amide_synthesis.enzyme_resolution import product_names


def main():
    failures = []

    print('structure repair')
    for name, broken, expected in REPAIR_CASES:
        got = normalize_smiles(broken)
        want = Chem.CanonSmiles(expected)
        ok = got == want
        print(f'  {"PASS" if ok else "FAIL"}  {name}')
        if not ok:
            failures.append(f'{name}: {got} != {want}')

    print('\nfree-amine classifier')
    for name, smiles, expected in AMINE_CASES:
        got = count_free_amines(Chem.MolFromSmiles(smiles))
        ok = got == expected
        print(f'  {"PASS" if ok else "FAIL"}  {name} (expected {expected}, got {got})')
        if not ok:
            failures.append(f'{name}: {got} != {expected}')

    print('\nacylation screen')
    for name, rxn, should_pass, expected_amines in ACYLATION_CASES:
        acceptors = acylated_acceptors(rxn)
        passed = bool(acceptors)
        amines = max((n for _, n in acceptors), default=0)
        ok = (passed == should_pass) and (amines == expected_amines)
        print(f'  {"PASS" if ok else "FAIL"}  {name} '
              f'(acylation={passed}, acceptor amines={amines})')
        if not ok:
            failures.append(f'{name}: pass={passed}/{should_pass} '
                            f'amines={amines}/{expected_amines}')

    failures += check_product_names()

    total = (len(REPAIR_CASES) + len(AMINE_CASES) + len(ACYLATION_CASES)
             + len(PRODUCT_NAME_CASES))
    print(f'\n{total - len(failures)}/{total} passed')
    if failures:
        for failure in failures:
            print(f'  FAILED: {failure}')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
