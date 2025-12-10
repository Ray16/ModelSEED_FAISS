import os
import faiss
from typing import Union
import numpy as np
import pandas as pd

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

index = load_faiss_index('reaction_embeddings.faiss')

rxn_data = pd.read_csv('rxn_data.csv')

all_idx_i = []
all_idx_j = []
cos_sim = []
fp = np.load('rxn_fingerprints.npy')
fp = l2_normalize_vectors(fp) 

print(f'Computing cosine similarity matrix...')
cos_matrix = fp @ fp.T
np.save('pair_cos_sim.npy', cos_matrix)