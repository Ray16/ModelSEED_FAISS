"""Repair two systematic structure defects in the current ModelSEEDDatabase master.

The 2026-10-06 structure refresh introduced, for ~14% of compounds (including NAD,
NADH, NADP, CoA and every acyl-CoA), two coupled errors:

  1. carboxamides written as IMINOLS, both deprotonated (C([O-])=N, ~5,100
     compounds) and neutral (C(O)=N, ~665 including nicotinamide and
     N1-acetylspermidine)
  2. adenine N1 protonated at pH 7:                    n        ->  [nH+]

The two cancel in formal charge, which is presumably why an automated protonation
step accepted them, but the tautomer is wrong: at pH 7 an amide is an amide, and
adenine N1 (pKa ~3.5) is neutral. Undoing both reproduces the previous, correct
structures exactly (verified against commit 581ba04 for NAD and CoA).

This matters here beyond cosmetics: an amide-detection SMARTS does not match an
iminolate, so leaving it in place would make every CoA-dependent reaction invisible
to the amide filter this analysis depends on.
"""
from __future__ import annotations

from rdkit import Chem

IMINOLATE = Chem.MolFromSmarts('[CX3]([OX1-])=[NX2]')
# The same tautomer error also appears NEUTRAL (C(O)=N, 665 compounds including
# nicotinamide, formamide and N1-acetylspermidine). Repairing only the anionic
# form leaves those broken, which silently hides real amide chemistry.
IMINOL = Chem.MolFromSmarts('[CX3]([OX2H1])=[NX2]')
PROTONATED_AROMATIC_N = Chem.MolFromSmarts('[nH+]')


def normalize(mol: Chem.Mol) -> Chem.Mol:
    """Return mol with iminolates restored to amides and aromatic [nH+] neutralised."""
    if mol is None:
        return None
    edit = Chem.RWMol(mol)
    changed = False

    for carbon, oxygen, nitrogen in mol.GetSubstructMatches(IMINOLATE):
        edit.GetBondBetweenAtoms(carbon, oxygen).SetBondType(Chem.BondType.DOUBLE)
        edit.GetBondBetweenAtoms(carbon, nitrogen).SetBondType(Chem.BondType.SINGLE)
        edit.GetAtomWithIdx(oxygen).SetFormalCharge(0)
        n_atom = edit.GetAtomWithIdx(nitrogen)
        n_atom.SetNumExplicitHs(n_atom.GetTotalNumHs() + 1)
        n_atom.SetNoImplicit(False)
        changed = True

    for carbon, oxygen, nitrogen in mol.GetSubstructMatches(IMINOL):
        edit.GetBondBetweenAtoms(carbon, oxygen).SetBondType(Chem.BondType.DOUBLE)
        edit.GetBondBetweenAtoms(carbon, nitrogen).SetBondType(Chem.BondType.SINGLE)
        o_atom = edit.GetAtomWithIdx(oxygen)
        o_atom.SetNumExplicitHs(0)
        o_atom.SetNoImplicit(False)
        n_atom = edit.GetAtomWithIdx(nitrogen)
        n_atom.SetNumExplicitHs(n_atom.GetTotalNumHs() + 1)
        n_atom.SetNoImplicit(False)
        changed = True

    for (idx,) in mol.GetSubstructMatches(PROTONATED_AROMATIC_N):
        atom = edit.GetAtomWithIdx(idx)
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(0)
        changed = True

    if not changed:
        return mol
    out = edit.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        return mol          # never let a repair attempt corrupt a usable structure
    return out


def normalize_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(normalize(mol))
