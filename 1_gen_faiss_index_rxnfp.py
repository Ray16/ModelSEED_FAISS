"""
Step 1 — Create a FAISS index from pre-computed RXNFP fingerprints.

Inputs:  rxn_fingerprints.npy
Outputs: reaction_embeddings.faiss
"""

import time

import faiss
import numpy as np
from config import FAISS_INDEX_FILE, RXN_FINGERPRINTS_NPY
from utils import l2_normalize_vectors, save_faiss_index


def create_index(indexed_vectors: np.ndarray, index_file: str) -> None:
    """Create a FAISS inner-product index from *indexed_vectors* and save it."""
    print("Normalizing indexed vectors (L2) for cosine similarity...")
    indexed_vectors = l2_normalize_vectors(indexed_vectors)

    d = indexed_vectors.shape[1]

    # IndexFlatIP computes inner product; on L2-normalized vectors this
    # is equivalent to cosine similarity.
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
