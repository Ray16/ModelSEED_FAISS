#!/usr/bin/env python
"""
Step 2b — Single-reaction similarity search with reaction-center prefiltering.

Only reactions that share the same reaction center as the query are considered.
Within that subset the top-k most similar (by RXNFP cosine similarity) are
returned.

Usage:
    python 2_perform_similarity_search_single_top30_rxn_center_prefilter.py \
        --rxn_name <rxn_name>
"""

import argparse

import numpy as np
import pandas as pd
from config import (
    FAISS_INDEX_FILE,
    MAPPED_RXNS_WITH_RXN_CENTERS_CSV,
    QUERY_RESULT_PREFILTER_CSV,
    RXN_DATA_CSV,
    RXN_FINGERPRINTS_NPY,
)
from utils import l2_normalize_vectors, load_faiss_index, parse_ec_numbers


def search(rxn_name: str, top_k: int = 30) -> pd.DataFrame:
    """Search for the *top_k* most similar reactions that share the same
    reaction center as *rxn_name*."""

    # Load data
    df = pd.read_csv(RXN_DATA_CSV)
    df_centers = pd.read_csv(MAPPED_RXNS_WITH_RXN_CENTERS_CSV)
    fps = np.load(RXN_FINGERPRINTS_NPY).astype(np.float32)

    # Identify query
    query_idx = df[df.id == rxn_name].index.to_numpy()[0]
    query_ec = parse_ec_numbers(df.iloc[query_idx, 3])
    query_center = df_centers.iloc[query_idx]["reaction_center"]

    print(f"Query reaction name: {df.iloc[query_idx, 0]}")
    print(f"Query reaction EC number: {query_ec}")
    print(f"Query reaction center: {query_center}")

    # Find all reactions with the same reaction center
    mask = df_centers["reaction_center"] == query_center
    candidate_indices = np.where(mask.values)[0]
    print(f"Candidates with same reaction center: {len(candidate_indices)}")

    if len(candidate_indices) == 0:
        print("No candidates found with the same reaction center.")
        return pd.DataFrame()

    # Compute cosine similarity between query and all candidates
    query_vec = fps[query_idx].reshape(1, -1).copy()
    l2_normalize_vectors(query_vec)

    candidate_vecs = fps[candidate_indices].copy()
    l2_normalize_vectors(candidate_vecs)

    # Cosine similarity = dot product of L2-normalized vectors
    similarities = (candidate_vecs @ query_vec.T).flatten()

    # Sort by similarity descending, take top_k
    top_k_actual = min(top_k, len(similarities))
    top_local = np.argsort(-similarities)[:top_k_actual]

    result_rows = []
    for rank, local_idx in enumerate(top_local, start=1):
        global_idx = candidate_indices[local_idx]
        result_rows.append(
            {
                "similarity_ranking": rank,
                "rxn_name": df.iloc[global_idx]["id"],
                "ec_number": parse_ec_numbers(df.iloc[global_idx, 3]),
                "cosine_similarity": similarities[local_idx],
                "reaction_center": df_centers.iloc[global_idx]["reaction_center"],
                "rxn_smiles": df.iloc[global_idx]["rxn_smiles"],
            }
        )

    result_df = pd.DataFrame(result_rows)
    result_df.to_csv(QUERY_RESULT_PREFILTER_CSV, index=False)
    return result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Single-reaction FAISS similarity search with "
        "reaction-center prefiltering"
    )
    parser.add_argument(
        "--rxn_name", type=str, required=True, help="Reaction ID (e.g. rxn00044)"
    )
    args = parser.parse_args()

    result_df = search(args.rxn_name)
    print(f"\nTop {len(result_df)} most similar reactions (same reaction center):")
    print(result_df.to_string())
    print(f"\nQuery results saved to {QUERY_RESULT_PREFILTER_CSV}")
