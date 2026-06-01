#!/usr/bin/env python
"""
Similarity search against the FAISS index.

Provides two search modes:
  - vanilla: search all reactions by RXNFP cosine similarity
  - prefilter: restrict candidates to those sharing the same reaction center

Usage:
    python -m src.search --rxn_name rxn00044
    python -m src.search --rxn_name rxn00044 --mode prefilter
    python -m src.search --rxn_name rxn00044 --top_k 5
"""

import argparse

import numpy as np
import pandas as pd
from src.config import (
    FAISS_INDEX_FILE,
    MAPPED_RXNS_WITH_RXN_CENTERS_CSV,
    QUERY_RESULT_CSV,
    QUERY_RESULT_PREFILTER_CSV,
    RXN_DATA_CSV,
    RXN_FINGERPRINTS_NPY,
)
from src.utils import l2_normalize_vectors, load_faiss_index, parse_ec_numbers


def search_vanilla(rxn_name: str, top_k: int = 30) -> pd.DataFrame:
    """Search for the *top_k* most similar reactions to *rxn_name*."""
    index = load_faiss_index(FAISS_INDEX_FILE)
    df = pd.read_csv(RXN_DATA_CSV)

    query_idx = df[df.id == rxn_name].index.to_numpy()[0]
    query_ec = parse_ec_numbers(df.iloc[query_idx, 3])
    print(f"Query reaction name: {df.iloc[query_idx, 0]}")
    print(f"Query reaction EC number: {query_ec}")

    query_vector = (
        np.load(RXN_FINGERPRINTS_NPY)[query_idx].astype(np.float32).reshape(1, -1)
    )
    query_vector = l2_normalize_vectors(query_vector)

    D, I = index.search(query_vector, top_k)
    distances = D[0]
    indices = I[0]

    result_df = pd.DataFrame(
        {
            "similarity_ranking": list(range(1, len(distances) + 1)),
            "rxn_name": [df.iloc[idx, 0] for idx in indices],
            "ec_number": [parse_ec_numbers(df.iloc[idx, 3]) for idx in indices],
            "distance": distances,
            "rxn_smiles": [df.iloc[idx]["rxn_smiles"] for idx in indices],
        }
    )

    result_df.to_csv(QUERY_RESULT_CSV, index=False)
    return result_df


def search_prefilter(rxn_name: str, top_k: int = 30) -> pd.DataFrame:
    """Search for the *top_k* most similar reactions that share the same
    reaction center as *rxn_name*."""

    df = pd.read_csv(RXN_DATA_CSV)
    df_centers = pd.read_csv(MAPPED_RXNS_WITH_RXN_CENTERS_CSV)
    fps = np.load(RXN_FINGERPRINTS_NPY).astype(np.float32)

    query_idx = df[df.id == rxn_name].index.to_numpy()[0]
    query_ec = parse_ec_numbers(df.iloc[query_idx, 3])
    query_center = df_centers.iloc[query_idx]["reaction_center"]

    print(f"Query reaction name: {df.iloc[query_idx, 0]}")
    print(f"Query reaction EC number: {query_ec}")
    print(f"Query reaction center: {query_center}")

    mask = df_centers["reaction_center"] == query_center
    candidate_indices = np.where(mask.values)[0]
    print(f"Candidates with same reaction center: {len(candidate_indices)}")

    if len(candidate_indices) == 0:
        print("No candidates found with the same reaction center.")
        return pd.DataFrame()

    query_vec = fps[query_idx].reshape(1, -1).copy()
    l2_normalize_vectors(query_vec)

    candidate_vecs = fps[candidate_indices].copy()
    l2_normalize_vectors(candidate_vecs)

    similarities = (candidate_vecs @ query_vec.T).flatten()

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
        description="Reaction similarity search (vanilla or prefiltered)"
    )
    parser.add_argument(
        "--rxn_name", type=str, required=True, help="Reaction ID (e.g. rxn00044)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="prefilter",
        choices=["vanilla", "prefilter"],
        help="Search mode (default: prefilter)",
    )
    parser.add_argument(
        "--top_k", type=int, default=30, help="Number of results (default: 30)"
    )
    args = parser.parse_args()

    if args.mode == "vanilla":
        result_df = search_vanilla(args.rxn_name, args.top_k)
        print(f"\nTop {len(result_df)} most similar reactions (vanilla):")
        print(result_df.to_string())
        print(f"\nQuery results saved to {QUERY_RESULT_CSV}")
    else:
        result_df = search_prefilter(args.rxn_name, args.top_k)
        print(f"\nTop {len(result_df)} most similar reactions (prefiltered):")
        print(result_df.to_string())
        print(f"\nQuery results saved to {QUERY_RESULT_PREFILTER_CSV}")
