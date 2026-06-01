#!/usr/bin/env python
"""
Extract reaction centers from atom-mapped reaction SMILES.

For each mapped reaction, identifies atoms whose local environment changes
between reactants and products (bond connectivity, bond order, formal charge,
or hydrogen count). Reports the reaction center as a list of atom fragments
from the reactant side (or product side for atoms that appear only in products).

Reads:  data/mapped_rxns.csv
Writes: data/mapped_rxns_with_rxn_centers.csv

Usage:
    conda activate rxnfp
    python -m src.reaction_center
"""

import csv
from multiprocessing import Pool, cpu_count

from rdkit import Chem, RDLogger
from src.config import MAPPED_RXNS_CSV, MAPPED_RXNS_WITH_RXN_CENTERS_CSV

RDLogger.logger().setLevel(RDLogger.ERROR)


def _atom_env(mol):
    """Build a dict mapping atom-map number -> local environment for every
    mapped atom in *mol*."""
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
        return True
    if env["symbol"] == "O" and env["numHs"] == 2 and not env["bonds"]:
        return True
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
    """Return a list of atom-fragment strings that form the reaction center."""
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
            e = p_env[am]
            if not _is_proton_or_water(e):
                changed_labels.append(_atom_label(e["symbol"], e["charge"], e["numHs"]))
        elif am not in p_env:
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

    print(f"Writing {MAPPED_RXNS_WITH_RXN_CENTERS_CSV} ...")
    with open(MAPPED_RXNS_WITH_RXN_CENTERS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {MAPPED_RXNS_WITH_RXN_CENTERS_CSV}")
    print("Done.")


if __name__ == "__main__":
    main()
