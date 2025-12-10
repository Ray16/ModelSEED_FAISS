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
'''
for idx_i in range(len(rxn_data)):
    fp_i = fp[idx_i].astype(np.float32).reshape(1, -1)
    for idx_j in range(idx_i + 1, len(rxn_data)):
        fp_j = fp[idx_j].astype(np.float32).reshape(1, -1)
        all_idx_i.append(idx_i)
        all_idx_j.append(idx_j)
        cos_sim.append()
'''
fp = l2_normalize_vectors(fp) 
cos_matrix = fp @ fp.T
N = cos_matrix.shape[0]
idx_i, idx_j = np.triu_indices(N, k=1)
pair_cos_sim = cos_matrix[idx_i, idx_j]
pairs_df = pd.DataFrame({
    "idx_i": idx_i,
    "idx_j": idx_j,
    "cosine_similarity": pair_cos_sim
})
print(f'cosine similarity matrix computed: {cos_matrix}')
pairs_df.to_csv('cos_sim_matrix.csv',index=False)