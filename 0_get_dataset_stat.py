#!/usr/bin/env python
"""
Step 0 — Print comprehensive dataset statistics.

Usage:
    conda activate rxnfp
    python 0_get_dataset_stat.py
"""

import ast
import csv
from collections import Counter

import numpy as np
from config import (
    BASE_DIR,
    MAPPED_RXNS_CSV,
    MAPPED_RXNS_WITH_RXN_CENTERS_CSV,
    RXN_DATA_CSV,
    RXN_FINGERPRINTS_NPY,
)
from utils import parse_ec_numbers


def main():
    # ── Original dataset ─────────────────────────────────────────────────
    with open(RXN_DATA_CSV) as f:
        rxn_data = list(csv.DictReader(f))

    total = len(rxn_data)
    has_ec = sum(
        1
        for r in rxn_data
        if r.get("ec_numbers", "").strip() and r.get("ec_numbers", "").strip() != "[]"
    )
    no_ec = total - has_ec

    single_ec = 0
    multi_ec = 0
    ec_counts = Counter()
    all_ecs = set()
    for r in rxn_data:
        ecs = parse_ec_numbers(r.get("ec_numbers", ""))
        if ecs:
            if len(ecs) == 1:
                single_ec += 1
            else:
                multi_ec += 1
            ec_counts[len(ecs)] += 1
            all_ecs.update(ecs)

    ec1_counts = Counter()
    for ec in all_ecs:
        ec1_counts[ec.split(".")[0]] += 1

    print("=" * 60)
    print("ORIGINAL DATASET (rxn_data.csv)")
    print("=" * 60)
    print(f"  Total reactions:       {total}")
    print(f"  With EC number:        {has_ec}")
    print(f"  Without EC number:     {no_ec}")
    print(f"  Single EC:             {single_ec}")
    print(f"  Multiple ECs:          {multi_ec}")
    print(f"  Unique EC numbers:     {len(all_ecs)}")
    print(f"  EC count distribution: {dict(sorted(ec_counts.items()))}")
    print(f"  EC main classes:       {dict(sorted(ec1_counts.items()))}")
    print()

    # ── Mass balance issues ──────────────────────────────────────────────
    hp_prod_only = 0
    hp_react_only = 0
    water_imbalanced = 0
    for r in rxn_data:
        smi = r.get("rxn_smiles", "")
        if ">>" not in smi:
            continue
        react, prod = smi.split(">>")
        rp = react.split(".")
        pp = prod.split(".")
        r_hp = sum(1 for x in rp if x.strip() == "[H+]")
        p_hp = sum(1 for x in pp if x.strip() == "[H+]")
        r_w = sum(1 for x in rp if x.strip() == "O")
        p_w = sum(1 for x in pp if x.strip() == "O")
        if p_hp > r_hp:
            hp_prod_only += 1
        if r_hp > p_hp:
            hp_react_only += 1
        if r_w != p_w:
            water_imbalanced += 1

    print("=" * 60)
    print("MASS BALANCE ISSUES (original rxn_data.csv)")
    print("=" * 60)
    print(f"  [H+] only in products:  {hp_prod_only}")
    print(f"  [H+] only in reactants: {hp_react_only}")
    print(f"  Water imbalanced:       {water_imbalanced}")
    print()

    # ── Atom mapping ─────────────────────────────────────────────────────
    with open(MAPPED_RXNS_CSV) as f:
        mapped = list(csv.DictReader(f))
    mapped_success = sum(1 for r in mapped if r.get("mapped_rxn", "").strip())
    mapped_fail = total - mapped_success

    print("=" * 60)
    print("ATOM MAPPING (mapped_rxns.csv)")
    print("=" * 60)
    print(f"  Successfully mapped:   {mapped_success}")
    print(f"  Failed to map:         {mapped_fail}")
    print()

    # ── Reaction centers ─────────────────────────────────────────────────
    with open(MAPPED_RXNS_WITH_RXN_CENTERS_CSV) as f:
        centers = list(csv.DictReader(f))
    has_center = sum(1 for r in centers if r.get("reaction_center", "[]") != "[]")
    no_center = total - has_center

    sizes = []
    center_strs = []
    for r in centers:
        try:
            labels = ast.literal_eval(r.get("reaction_center", "[]"))
            if labels:
                sizes.append(len(labels))
                center_strs.append(r["reaction_center"])
        except (ValueError, SyntaxError):
            pass

    unique_centers = len(set(center_strs))
    center_freq = Counter(center_strs)
    singletons = sum(1 for n in center_freq.values() if n == 1)

    has_hp = sum(1 for s in center_strs if "'H+'" in s)
    has_oh2 = sum(1 for s in center_strs if "'OH2'" in s)

    print("=" * 60)
    print("REACTION CENTERS (mapped_rxns_with_rxn_centers.csv)")
    print("=" * 60)
    print(f"  With reaction center:  {has_center}")
    print(f"  Without reaction center: {no_center}")
    print(f"    - Failed mapping:      {mapped_fail}")
    print(f"    - Identity (R==P):     {no_center - mapped_fail}")
    print(
        f"  Center size: mean={np.mean(sizes):.1f}, "
        f"median={np.median(sizes):.0f}, "
        f"min={min(sizes)}, max={max(sizes)}"
    )
    print(f"  Unique centers:        {unique_centers}")
    print(f"  Singleton centers:     {singletons}")
    print(f"  Centers with H+:       {has_hp}")
    print(f"  Centers with OH2:      {has_oh2}")
    print()

    # ── RXNFP fingerprints ───────────────────────────────────────────────
    fps = np.load(RXN_FINGERPRINTS_NPY)

    print("=" * 60)
    print("RXNFP FINGERPRINTS")
    print("=" * 60)
    print(f"  Shape: {fps.shape} ({fps.shape[1]}-dimensional)")
    print(f"  Dtype: {fps.dtype}")
    print()

    # ── Evaluation subsets ───────────────────────────────────────────────
    # EC info is in rxn_data.csv, centers in mapped_rxns_with_rxn_centers.csv
    ec_by_id = {}
    for r in rxn_data:
        ecs = parse_ec_numbers(r.get("ec_numbers", ""))
        ec_by_id[r["id"]] = ecs

    annotated_with_center = 0
    unannotated_with_center = 0
    for r in centers:
        if r.get("reaction_center", "[]") == "[]":
            continue
        ecs = ec_by_id.get(r["id"], [])
        if ecs:
            annotated_with_center += 1
        else:
            unannotated_with_center += 1

    print("=" * 60)
    print("EVALUATION SUBSETS")
    print("=" * 60)
    print(f"  Annotated + center (leave-one-out): {annotated_with_center}")
    print(f"  Unannotated + center (prediction):  {unannotated_with_center}")
    print(f"  Annotated without center:           {has_ec - annotated_with_center}")
    print()


if __name__ == "__main__":
    main()
