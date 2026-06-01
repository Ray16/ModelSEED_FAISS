#!/usr/bin/env python
"""
Compute the full pairwise cosine-similarity matrix.

Inputs:  data/rxn_fingerprints.npy
Outputs: data/pair_cos_sim.npy

Usage:
    python -m src.cosine_matrix
"""

import numpy as np
from src.config import PAIR_COS_SIM_NPY, RXN_FINGERPRINTS_NPY
from src.utils import l2_normalize_vectors


if __name__ == "__main__":
    fp = np.load(RXN_FINGERPRINTS_NPY)
    fp = l2_normalize_vectors(fp)

    print("Computing cosine similarity matrix...")
    cos_matrix = fp @ fp.T
    np.save(PAIR_COS_SIM_NPY, cos_matrix)
    print(f"Saved cosine similarity matrix to {PAIR_COS_SIM_NPY}")
