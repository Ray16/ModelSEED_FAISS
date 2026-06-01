#!/usr/bin/env python
"""
Create a FAISS index from pre-computed RXNFP fingerprints.

Inputs:  data/rxn_fingerprints.npy
Outputs: data/reaction_embeddings.faiss

Usage:
    conda activate rxnfp
    python -m src.faiss_index
"""

import time

import faiss
import numpy as np
from src.config import FAISS_INDEX_FILE, RXN_FINGERPRINTS_NPY
from src.utils import l2_normalize_vectors, save_faiss_index


def create_index(indexed_vectors: np.ndarray, index_file: str) -> None:
    """Create a FAISS inner-product index from *indexed_vectors* and save it."""
    print("Normalizing indexed vectors (L2) for cosine similarity...")
    indexed_vectors = l2_normalize_vectors(indexed_vectors)

    d = indexed_vectors.shape[1]

    index = faiss.IndexFlatIP(d)
    print(f"FAISS index created (Type: {type(index).__name__}, Dimension: {d})")

    index.add(indexed_vectors)
    print(f"Added {index.ntotal} vectors to the index.")

    save_faiss_index(index, index_file)


if __name__ == "__main__":
    start_time = time.time()

    fingerprints = np.load(RXN_FINGERPRINTS_NPY).astype("float32")
    create_index(fingerprints, FAISS_INDEX_FILE)

    elapsed = time.time() - start_time
    print(f"Time taken to create index: {elapsed:.2f} seconds")
