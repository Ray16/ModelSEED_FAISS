import os
import argparse
import faiss
import pandas as pd
import numpy as np
from typing import Union

parser = argparse.ArgumentParser()
parser.add_argument('--rxn_name',type=str)
args = parser.parse_args()

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

# choose query idx (for specific raction type)
df = pd.read_csv('rxn_data.csv')
query_idx = df[df.id==args.rxn_name].index.to_numpy()[0]

query_rxn_name = df.iloc[query_idx,0]
query_ec_number = df.iloc[query_idx,3]

print(f'Query reaction name: {query_rxn_name}')
print(f'Query reaction EC number: {query_ec_number}')

# index query vector
query_vectors = np.load('rxn_fingerprints.npy')[query_idx].astype(np.float32).reshape(1, -1)
query_vectors = l2_normalize_vectors(query_vectors)

# search for top-k candidates
k = 30

D, I = index.search(query_vectors, k)
distances = D[0]
indices = I[0]

# return the queried reactions
result_reaction_names = []
result_ec_number = []
for idx in indices:
    result_reaction_names.append(df.iloc[idx,0])
    result_ec_number.append(df.iloc[idx,3])

df = pd.DataFrame({'similarity_ranking':list(range(1,len(distances)+1)), 'rxn_name':result_reaction_names,'ec_number':result_ec_number,'distance':distances})
df.to_csv('query_result.csv',index=False)

print('Top 30 most similar reactions:')
print(df)
print('Query results saved to query_result.csv')