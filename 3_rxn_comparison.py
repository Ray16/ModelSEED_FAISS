#!/usr/bin/env python
"""
Step 3 — Compare search results from the vanilla FAISS search vs.
the reaction-center-prefiltered search.

Runs both searches for a given query reaction and prints a side-by-side
comparison including:
  - Overlap between the two result sets
  - EC number agreement with the query
  - Similarity score distributions
  - Detailed result tables

Usage:
    python 3_rxn_comparison.py --rxn_name rxn00044
"""

import argparse
import importlib
import sys

import pandas as pd
from config import MAPPED_RXNS_WITH_RXN_CENTERS_CSV, RXN_DATA_CSV
from utils import parse_ec_numbers

# Import the two search modules
vanilla_mod = importlib.import_module("2_perform_similarity_search_single_top30")
prefilter_mod = importlib.import_module(
    "2_perform_similarity_search_single_top30_rxn_center_prefilter"
)


def ec_overlap(query_ec, result_ec):
    """Return the deepest EC prefix shared between two EC number lists.

    For example if query has '3.5.1.54' and result has '3.5.1.4', the shared
    prefix is '3.5.1' (depth 3).  Returns 0 if no overlap at all.
    """
    best = 0
    for qe in query_ec:
        q_parts = qe.split(".")
        for re_ in result_ec:
            r_parts = re_.split(".")
            depth = 0
            for a, b in zip(q_parts, r_parts):
                if a == b:
                    depth += 1
                else:
                    break
            best = max(best, depth)
    return best


def compare(rxn_name: str, top_k: int = 30):
    """Run both searches and print a comparison."""

    # Load query info
    df_rxn = pd.read_csv(RXN_DATA_CSV)
    df_centers = pd.read_csv(MAPPED_RXNS_WITH_RXN_CENTERS_CSV)
    query_idx = df_rxn[df_rxn.id == rxn_name].index.to_numpy()[0]
    query_ec = parse_ec_numbers(df_rxn.iloc[query_idx, 3])
    query_center = df_centers.iloc[query_idx]["reaction_center"]

    print("=" * 80)
    print(f"Query: {rxn_name}")
    print(f"EC:    {query_ec}")
    print(f"Center: {query_center}")
    print(f"SMILES: {df_rxn.iloc[query_idx]['rxn_smiles']}")
    print("=" * 80)

    # --- Run vanilla search ---
    print("\n>>> Running vanilla FAISS search ...")
    df_vanilla = vanilla_mod.search(rxn_name, top_k)
    # The vanilla script stores inner-product as "distance" which equals
    # cosine similarity on L2-normalised vectors.
    df_vanilla = df_vanilla.rename(columns={"distance": "cosine_similarity"})

    # --- Run prefiltered search ---
    print("\n>>> Running reaction-center-prefiltered search ...")
    df_prefilter = prefilter_mod.search(rxn_name, top_k)

    # ------------------------------------------------------------------
    # Overlap
    # ------------------------------------------------------------------
    vanilla_set = set(df_vanilla["rxn_name"])
    prefilter_set = set(df_prefilter["rxn_name"])
    overlap = vanilla_set & prefilter_set
    only_vanilla = vanilla_set - prefilter_set
    only_prefilter = prefilter_set - vanilla_set

    print("\n" + "=" * 80)
    print("OVERLAP ANALYSIS")
    print("=" * 80)
    print(f"Vanilla results:     {len(vanilla_set)}")
    print(f"Prefilter results:   {len(prefilter_set)}")
    print(f"Shared:              {len(overlap)}")
    print(f"Only in vanilla:     {len(only_vanilla)}")
    print(f"Only in prefilter:   {len(only_prefilter)}")

    # ------------------------------------------------------------------
    # EC agreement
    # ------------------------------------------------------------------
    def ec_stats(df, label):
        depths = []
        for _, row in df.iterrows():
            ec = (
                row["ec_number"]
                if isinstance(row["ec_number"], list)
                else parse_ec_numbers(str(row["ec_number"]))
            )
            depths.append(ec_overlap(query_ec, ec))
        df = df.copy()
        df["ec_depth"] = depths
        exact = sum(1 for d in depths if d == 4)
        partial = sum(1 for d in depths if 1 <= d < 4)
        none_ = sum(1 for d in depths if d == 0)
        print(f"\n{label}:")
        print(f"  Exact EC match (depth=4): {exact}/{len(depths)}")
        print(f"  Partial EC match (1-3):   {partial}/{len(depths)}")
        print(f"  No EC overlap (depth=0):  {none_}/{len(depths)}")
        print(f"  Mean EC depth:            {sum(depths) / len(depths):.2f}")
        return df

    print("\n" + "=" * 80)
    print("EC NUMBER AGREEMENT")
    print("=" * 80)
    df_vanilla = ec_stats(df_vanilla, "Vanilla FAISS")
    df_prefilter = ec_stats(df_prefilter, "Prefilter (reaction center)")

    # ------------------------------------------------------------------
    # Similarity score distributions
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("COSINE SIMILARITY DISTRIBUTION")
    print("=" * 80)
    for label, df in [("Vanilla", df_vanilla), ("Prefilter", df_prefilter)]:
        sims = df["cosine_similarity"].astype(float)
        print(f"\n{label}:")
        print(f"  Mean:   {sims.mean():.4f}")
        print(f"  Median: {sims.median():.4f}")
        print(f"  Min:    {sims.min():.4f}")
        print(f"  Max:    {sims.max():.4f}")

    # ------------------------------------------------------------------
    # Add reaction center info to vanilla results for the table
    # ------------------------------------------------------------------
    center_map = dict(zip(df_centers["id"], df_centers["reaction_center"]))
    df_vanilla["reaction_center"] = df_vanilla["rxn_name"].map(center_map)

    # ------------------------------------------------------------------
    # Print result tables
    # ------------------------------------------------------------------
    display_cols = [
        "similarity_ranking",
        "rxn_name",
        "ec_number",
        "cosine_similarity",
        "ec_depth",
        "reaction_center",
    ]

    print("\n" + "=" * 80)
    print("VANILLA FAISS — TOP 30")
    print("=" * 80)
    print(df_vanilla[display_cols].to_string(index=False))

    print("\n" + "=" * 80)
    print("PREFILTER (REACTION CENTER) — TOP 30")
    print("=" * 80)
    print(df_prefilter[display_cols].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare vanilla vs reaction-center-prefiltered search"
    )
    parser.add_argument(
        "--rxn_name",
        type=str,
        default="rxn00044",
        help="Reaction ID to query (default: rxn00044)",
    )
    args = parser.parse_args()
    compare(args.rxn_name)
