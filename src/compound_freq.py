#!/usr/bin/env python
"""
Count compound frequency across all non-obsolete ModelSEED reactions.

Outputs: data/cpd_freq.csv

Usage:
    python -m src.compound_freq
"""

import sys
import warnings
from collections import Counter

import pandas as pd
from src.config import CPD_FREQ_CSV, MODELSEED_PYTHON_LIB

warnings.simplefilter("ignore", SyntaxWarning)


if __name__ == "__main__":
    sys.path.append(MODELSEED_PYTHON_LIB)
    from BiochemPy import Compounds, Reactions

    compounds_dict = Compounds().loadCompounds()
    reactions_dict = Reactions().loadReactions()

    cpd_counter = Counter()
    for rxn_id, rxn_obj in reactions_dict.items():
        if rxn_obj["is_obsolete"] == 0:
            for rgt in rxn_obj["stoichiometry"]:
                cpd_counter[rgt["compound"]] += 1

    df = pd.DataFrame(cpd_counter.most_common(), columns=["compound_id", "frequency"])
    df["name"] = df["compound_id"].map(
        lambda cid: compounds_dict[cid]["name"] if cid in compounds_dict else ""
    )
    df.to_csv(CPD_FREQ_CSV, index=False)

    print(f"Counted {len(cpd_counter)} unique compounds across non-obsolete reactions.")
    print(f"Saved to {CPD_FREQ_CSV}")
    print(df.head(20))
