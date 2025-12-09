#!/usr/bin/env python

# input: rxn_fingerprints.npy
# output: reaction_embeddings.faiss

import time
import pickle

import numpy as np
import faiss
import os
from typing import Union

FAISS_INDEX_FILE = 'reaction_embeddings.faiss'

def create_index(indexed_vectors: np.ndarray, index_file: str) -> None:
    """
    Creates a FAISS index from the indexed vectors and performs a search.
    """
  
    # === CRITICAL: Normalizing Indexed Vectors ===
    print("Normalizing indexed vectors (L2) for Cosine Similarity...")
    indexed_vectors = l2_normalize_vectors(indexed_vectors)
    
    # d is taken from the normalized vectors
    d = indexed_vectors.shape[1] 
    
    # 1. Initialize the Index
    # index = faiss.IndexFlatL2(d)
    # FlatIP is for Inner Product and works best with cosine similarity
    index = faiss.IndexFlatIP(d)
    print(f"FAISS index created (Type: {type(index).__name__}, Dimension: {d})")
    
    # 2. Add vectors to the Index
    index.add(indexed_vectors)
    print(f"Added {index.ntotal} vectors to the index.")

    save_faiss_index(index, index_file)
    
def perform_search_on_loaded_index(index: faiss.Index) -> None:
    """
    Defines new query vectors and performs a search on the provided index.
    """
    d = index.d
    
    # 1. Define New Query Vectors 
    np.random.seed(10)
    new_query_vector = np.random.rand(1, d).astype('float32')

    # 2. Perform the Search
    k = 3  # Find the 3 most similar hits
    print(f"\nSearching for the {k} nearest neighbors to the new query...")
    
    D, I = index.search(new_query_vector, k)
    
    # 3. Output Results
    print("-" * 50)
    print("Local Similarity Search Results (Using Loaded Index):")
    for i in range(k):
        distance = D[0][i]
        index_id = I[0][i]
        print(f"Rank {i+1}: Original Index ID: {index_id}, Distance (L2): {distance:.6f}")
    print("-" * 50)

def save_faiss_index(index: faiss.Index, filename: str) -> None:
    """Saves a FAISS index to a specified file path."""
    try:
        faiss.write_index(index, filename)
        print(f"Successfully saved FAISS index to: {filename}")
    except Exception as e:
        print(f"Error saving FAISS index: {e}")

def load_faiss_index(filename: str) -> Union[faiss.Index, None]:
    """Loads a FAISS index from a specified file path."""
    if os.path.exists(filename):
        try:
            index = faiss.read_index(filename)
            print(f"Successfully loaded existing FAISS index from: {filename}")
            return index
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            return None
    return None

def l2_normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """L2 normalizes a set of vectors (making their magnitude 1.0)."""
    # faiss.normalize_L2 is a highly optimized, in-place operation
    faiss.normalize_L2(vectors)
    return vectors

if __name__ == "__main__":
    start_time = time.time()
    fingerprints_file = "rxn_fingerprints.npy"
    fingerprints = np.load(fingerprints_file).astype('float32')
    create_index(fingerprints, FAISS_INDEX_FILE)
    end_time = time.time()
    print(f"Time taken to create index: {end_time - start_time:.2f} seconds")
    