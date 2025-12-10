import faiss
import numpy as np

def l2_normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """L2 normalizes a set of vectors (making their magnitude 1.0)."""
    # faiss.normalize_L2 is a highly optimized, in-place operation
    faiss.normalize_L2(vectors)
    return vectors

fp = np.load('rxn_fingerprints.npy')
fp = l2_normalize_vectors(fp) 

print(f'Computing cosine similarity matrix...')
cos_matrix = fp @ fp.T
np.save('pair_cos_sim.npy', cos_matrix)