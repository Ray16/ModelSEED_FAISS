#!/usr/bin/env python
"""
Compare cosine similarity of EC X.Y.Z.* reactions (intra-group)
vs their similarity to all other reactions (cross-group).

Usage:
    python -m src.ec_analysis
    python -m src.ec_analysis --ec_prefix 2.7.1.
"""

import argparse

import numpy as np
import pandas as pd
from src.config import PAIR_COS_SIM_NPY, RXN_DATA_CSV
from src.utils import parse_ec_numbers


def analyze_ec_group(ec_prefix: str = "3.5.1.") -> None:
    """Print intra- and cross-group similarity statistics for *ec_prefix*."""
    cos_sim_matrix = np.load(PAIR_COS_SIM_NPY)
    rxn_data = pd.read_csv(RXN_DATA_CSV)

    rxn_data["_ec_list"] = rxn_data["ec_numbers"].apply(parse_ec_numbers)
    mask = rxn_data["_ec_list"].apply(
        lambda ec_list: any(ec.startswith(ec_prefix) for ec in ec_list)
    )
    selected_indices = list(rxn_data[mask].index)
    N = len(selected_indices)

    subset_matrix = cos_sim_matrix[np.ix_(selected_indices, selected_indices)]
    row_idx, col_idx = np.triu_indices(N, k=1)
    intra_pairs = subset_matrix[row_idx, col_idx]

    print(f"Number of EC {ec_prefix}x reactions found: {N}")
    print(f"Number of unique reaction pairs analyzed: {len(intra_pairs)}")
    print(f"Average similarity ({ec_prefix}x vs {ec_prefix}x): {intra_pairs.mean():.6f}")
    print(f"Std dev          ({ec_prefix}x vs {ec_prefix}x): {intra_pairs.std():.6f}")

    non_selected_indices = sorted(set(rxn_data.index) - set(selected_indices))
    B = len(non_selected_indices)
    print(f"\nNumber of NON-{ec_prefix}x reactions: {B}")

    cross_matrix = cos_sim_matrix[np.ix_(selected_indices, non_selected_indices)]
    cross_pairs = cross_matrix.flatten()

    print(f"Number of cross-pairs analyzed: {len(cross_pairs)}")
    print(f"Average similarity ({ec_prefix}x vs NON-{ec_prefix}x): {cross_pairs.mean():.6f}")
    print(f"Std dev          ({ec_prefix}x vs NON-{ec_prefix}x): {cross_pairs.std():.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EC group cosine similarity analysis")
    parser.add_argument("--ec_prefix", type=str, default="3.5.1.", help="EC prefix to analyze (default: 3.5.1.)")
    args = parser.parse_args()
    analyze_ec_group(args.ec_prefix)
