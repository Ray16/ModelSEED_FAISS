#!/usr/bin/env python
"""
Generate reaction SMILES and RXNFP fingerprints for all ModelSEED reactions.

Inputs:  ModelSEEDDatabase/
Outputs: data/rxn_data.csv, data/rxn_fingerprints.npy

Usage:
    conda activate rxnfp
    python -m src.data_prep
"""

import csv
import math
import sys
import warnings
from collections import OrderedDict

import numpy as np
from src.config import (
    FINGERPRINT_BATCH_SIZE,
    MODELSEED_PYTHON_LIB,
    RXN_DATA_CSV,
    RXN_FINGERPRINTS_NPY,
)

warnings.simplefilter("ignore", SyntaxWarning)


def _build_rxn_smiles(rxn_obj, compounds_dict):
    """Return a reaction SMILES string, or ``None`` if any compound lacks SMILES."""
    rgt_smiles, pdt_smiles = [], []

    for rgt in rxn_obj["stoichiometry"]:
        cpd_id = rgt["compound"]
        if cpd_id not in compounds_dict:
            return None

        cpd_smiles = compounds_dict[cpd_id]["smiles"]
        if cpd_smiles == "":
            return None

        count = math.ceil(abs(rgt["coefficient"]))
        if rgt["coefficient"] < 0:
            rgt_smiles.extend([cpd_smiles] * count)
        elif rgt["coefficient"] > 0:
            pdt_smiles.extend([cpd_smiles] * count)

    return ">>".join([".".join(rgt_smiles), ".".join(pdt_smiles)])


def generate_rxn_data(compounds_dict, reactions_dict):
    """Build a list of reaction data dicts and a parallel list of SMILES strings."""
    rxn_data_list = []
    rxn_smiles_list = []

    for rxn_obj in reactions_dict.values():
        if rxn_obj["is_obsolete"] != 0:
            continue

        rxn_smiles_str = _build_rxn_smiles(rxn_obj, compounds_dict)
        if rxn_smiles_str is None:
            continue

        rxn_smiles_list.append(rxn_smiles_str)

        rxn_data = OrderedDict()
        rxn_data["id"] = rxn_obj.get("id", "")
        rxn_data["name"] = rxn_obj.get("name", "")
        rxn_data["abbreviation"] = rxn_obj.get("abbreviation", "")
        ec = rxn_obj.get("ec_numbers", [])
        rxn_data["ec_numbers"] = str(ec) if isinstance(ec, list) else "[]"
        rxn_data["reversibility"] = rxn_obj.get("reversibility", "")
        rxn_data["deltag"] = rxn_obj.get("deltag", "")
        rxn_data["deltagerr"] = rxn_obj.get("deltagerr", "")
        rxn_data["definition"] = rxn_obj.get("definition", "")
        rxn_data["is_transport"] = rxn_obj.get("is_transport", "")
        rxn_data["rxn_smiles"] = rxn_smiles_str
        rxn_data_list.append(rxn_data)

    return rxn_data_list, rxn_smiles_list


def write_rxn_csv(rxn_data_list):
    """Write *rxn_data_list* to the configured CSV path."""
    if not rxn_data_list:
        print("No data to write to CSV.")
        return

    fieldnames = list(rxn_data_list[0].keys())
    print(f"Writing data to {RXN_DATA_CSV}...")

    with open(RXN_DATA_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=",")
        writer.writeheader()
        writer.writerows(rxn_data_list)

    print(
        f"Successfully created {RXN_DATA_CSV} with {len(rxn_data_list)} reaction entries."
    )


def generate_fingerprints(rxn_smiles_list):
    """Generate RXNFP fingerprints in batches and save to disk."""
    from rxnfp.transformer_fingerprints import (
        RXNBERTFingerprintGenerator,
        get_default_model_and_tokenizer,
    )

    model, tokenizer = get_default_model_and_tokenizer()
    rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)

    rxn_fps = []
    for i in range(0, len(rxn_smiles_list), FINGERPRINT_BATCH_SIZE):
        print(f"Chunk: {i}")
        chunk = rxn_smiles_list[i : i + FINGERPRINT_BATCH_SIZE]
        rxn_fps.extend(rxnfp_generator.convert_batch(chunk))
        print(f"{len(rxn_fps)} fingerprints generated (dim={len(rxn_fps[0])})")

    rxn_fps_array = np.array(rxn_fps, dtype=np.float32)
    np.save(RXN_FINGERPRINTS_NPY, rxn_fps_array)
    print("Generated the fingerprints for all reactions.")


if __name__ == "__main__":
    sys.path.append(MODELSEED_PYTHON_LIB)
    from BiochemPy import Compounds, Reactions

    compounds_dict = Compounds().loadCompounds()
    reactions_dict = Reactions().loadReactions()

    rxn_data_list, rxn_smiles_list = generate_rxn_data(compounds_dict, reactions_dict)
    print(f"\nGenerated SMILES string for {len(rxn_smiles_list)} reactions.")

    write_rxn_csv(rxn_data_list)
    generate_fingerprints(rxn_smiles_list)
