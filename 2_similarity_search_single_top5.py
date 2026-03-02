#!/usr/bin/env python
"""
Step 2 — Perform a single-reaction similarity search against the FAISS index.

Usage:
    python 2_perform_similarity_search_single.py --rxn_name <rxn_name>
"""

import argparse

import numpy as np
import pandas as pd
from config import (
    FAISS_INDEX_FILE,
    QUERY_RESULT_CSV,
    RXN_DATA_CSV,
    RXN_FINGERPRINTS_NPY,
)
from utils import l2_normalize_vectors, load_faiss_index, parse_ec_numbers


def search(rxn_name: str, top_k: int = 5) -> pd.DataFrame:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Single-reaction FAISS similarity search"
    )
    parser.add_argument(
        "--rxn_name", type=str, required=True, help="Reaction ID (e.g. rxn00001)"
    )
    args = parser.parse_args()

    result_df = search(args.rxn_name)
    print(f"Top {len(result_df)} most similar reactions:")
    print(result_df.to_string())
    print(f"Query results saved to {QUERY_RESULT_CSV}")
