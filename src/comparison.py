#!/usr/bin/env python
"""
Compare search results from vanilla FAISS search vs.
reaction-center-prefiltered search.

Usage:
    python -m src.comparison --rxn_name rxn00044
"""

import argparse

import pandas as pd
from src.config import MAPPED_RXNS_WITH_RXN_CENTERS_CSV, RXN_DATA_CSV
from src.search import search_prefilter, search_vanilla
from src.utils import parse_ec_numbers


def ec_overlap(query_ec, result_ec):
    """Return the deepest EC prefix shared between two EC number lists."""
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

    print("\n>>> Running vanilla FAISS search ...")
    df_vanilla = search_vanilla(rxn_name, top_k)
    df_vanilla = df_vanilla.rename(columns={"distance": "cosine_similarity"})

    print("\n>>> Running reaction-center-prefiltered search ...")
    df_prefilter = search_prefilter(rxn_name, top_k)

    # Overlap
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

    # EC agreement
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

    # Similarity score distributions
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

    # Add reaction center info to vanilla results
    center_map = dict(zip(df_centers["id"], df_centers["reaction_center"]))
    df_vanilla["reaction_center"] = df_vanilla["rxn_name"].map(center_map)

    # Print result tables
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
