#!/usr/bin/env python
"""
Step 1 — Extract reaction centers from atom-mapped reaction SMILES.

For each mapped reaction, identifies atoms whose local environment changes
between reactants and products (bond connectivity, bond order, formal charge,
or hydrogen count). Reports the reaction center as a list of atom fragments
from the reactant side (or product side for atoms that appear only in products).

Reads:  mapped_rxns.csv
Writes: mapped_rxns_with_rxn_centers.csv

Usage:
    conda activate rxnfp
    python 1_extract_reaction_center.py
"""

import csv
import os
import sys
from multiprocessing import Pool, cpu_count

from config import BASE_DIR
from rdkit import Chem, RDLogger

# Suppress RDKit warnings (e.g. "not removing hydrogen atom without neighbors")
RDLogger.logger().setLevel(RDLogger.ERROR)

MAPPED_RXNS_CSV = os.path.join(BASE_DIR, "mapped_rxns.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "mapped_rxns_with_rxn_centers.csv")


def _atom_env(mol):
    """Build a dict mapping atom-map number -> local environment for every
    mapped atom in *mol*.

    The environment captures:
      - symbol, formal charge, total number of Hs
      - dict of {neighbour_atom_map: bond_order}
    """
    info = {}
    for atom in mol.GetAtoms():
        am = atom.GetAtomMapNum()
        if am == 0:
            continue
        bonds = {}
        for nbr in atom.GetNeighbors():
            nbr_map = nbr.GetAtomMapNum()
            bond = mol.GetBondBetweenAtoms(atom.GetIdx(), nbr.GetIdx())
            bonds[nbr_map] = (bond.GetBondTypeAsDouble(), str(bond.GetStereo()))
        info[am] = {
            "symbol": atom.GetSymbol(),
            "charge": atom.GetFormalCharge(),
            "numHs": atom.GetTotalNumHs(),
            "chiral": str(atom.GetChiralTag()),
            "bonds": bonds,
        }
    return info


def _is_proton_or_water(env):
    """Return True if this atom is a standalone [H+] or [OH2] (water)."""
    if env["symbol"] == "H" and env["charge"] == 1 and not env["bonds"]:
        return True  # [H+]
    if env["symbol"] == "O" and env["numHs"] == 2 and not env["bonds"]:
        return True  # [OH2] (standalone water)
    return False


def _atom_label(symbol, charge, numHs):
    """Return a compact SMILES-like label such as 'OH', 'O-', 'NH2', 'H+', 'C'."""
    h_str = f"H{numHs}" if numHs > 1 else ("H" if numHs == 1 else "")
    if charge > 0:
        chg_str = "+" if charge == 1 else f"+{charge}"
    elif charge < 0:
        chg_str = "-" if charge == -1 else f"-{abs(charge)}"
    else:
        chg_str = ""
    return f"{symbol}{h_str}{chg_str}"


def extract_reaction_center(mapped_rxn):
    """Return a list of atom-fragment strings that form the reaction center.

    An atom is part of the reaction center if any of the following differ
    between reactants and products:
      - formal charge
      - total hydrogen count
      - set of bonded neighbours (by atom-map number) or bond orders

    Atoms present only on one side (e.g. explicit [H+] produced) are also
    included.
    """
    if not mapped_rxn or ">>" not in mapped_rxn:
        return []

    r_smi, p_smi = mapped_rxn.split(">>")
    r_mol = Chem.MolFromSmiles(r_smi)
    p_mol = Chem.MolFromSmiles(p_smi)

    if r_mol is None or p_mol is None:
        return []

    r_env = _atom_env(r_mol)
    p_env = _atom_env(p_mol)

    changed_labels = []
    all_maps = sorted(set(r_env.keys()) | set(p_env.keys()))

    for am in all_maps:
        if am not in r_env:
            # Atom only in products — skip standalone [H+] or [OH2]
            e = p_env[am]
            if not _is_proton_or_water(e):
                changed_labels.append(_atom_label(e["symbol"], e["charge"], e["numHs"]))
        elif am not in p_env:
            # Atom only in reactants — skip standalone [H+] or [OH2]
            e = r_env[am]
            if not _is_proton_or_water(e):
                changed_labels.append(_atom_label(e["symbol"], e["charge"], e["numHs"]))
        else:
            re, pe = r_env[am], p_env[am]
            if (
                re["charge"] != pe["charge"]
                or re["numHs"] != pe["numHs"]
                or re["chiral"] != pe["chiral"]
                or re["bonds"] != pe["bonds"]
            ):
                # Use reactant-side label
                changed_labels.append(
                    _atom_label(re["symbol"], re["charge"], re["numHs"])
                )

    return sorted(changed_labels)


def main():
    print(f"Reading {MAPPED_RXNS_CSV} ...")
    rows = []
    with open(MAPPED_RXNS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        in_fields = reader.fieldnames
        for row in reader:
            rows.append(row)
    print(f"Loaded {len(rows)} reactions.\n")

    out_fields = list(in_fields) + ["reaction_center"]

    mapped_rxns = [row.get("mapped_rxn", "") for row in rows]

    n_workers = cpu_count()
    print(f"Extracting reaction centers using {n_workers} workers ...")
    with Pool(n_workers) as pool:
        centers = pool.map(extract_reaction_center, mapped_rxns, chunksize=256)

    n_with_center = 0
    for row, center in zip(rows, centers):
        row["reaction_center"] = str(center) if center else "[]"
        if center:
            n_with_center += 1

    print(f"\nReactions with identified center: {n_with_center} / {len(rows)}")

    print(f"Writing {OUTPUT_CSV} ...")
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")
    print("Done.")


if __name__ == "__main__":
    main()
