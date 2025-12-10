from time import time
import numpy as np
import pandas as pd

cos_sim_matrix = np.load('pair_cos_sim.npy')

# get the indices of all EC 3.5.1.x reactions
rxn_data = pd.read_csv('rxn_data.csv')
selected_rxn_indices = list(rxn_data[rxn_data['ec_numbers'].str.startswith('3.5.1.')].index)
N = len(selected_rxn_indices)

subset_matrix = cos_sim_matrix[np.ix_(selected_rxn_indices, selected_rxn_indices)]
row_indices, col_indices = np.triu_indices(N, k=1)
unique_pairs = subset_matrix[row_indices, col_indices]

average_similarity = unique_pairs.mean()
std_similarity = unique_pairs.std()

print(f"Number of EC 3.5.1.x reactions found: {N}")
print(f"Number of unique reaction pairs analyzed: {len(unique_pairs)}")
print(f"Average reaction similarity (3.5.1.x vs 3.5.1.x): {average_similarity:.6f}")
print(f"Standard deviation of reaction similarity (3.5.1.x vs 3.5.1.x): {std_similarity:.6f}")

# all reaction indices
all_indices = set(rxn_data.index)

# non-3.5.1.x reactions
non_selected_indices = sorted(list(all_indices - set(selected_rxn_indices)))

A = len(selected_rxn_indices)
B = len(non_selected_indices)

print(f"\nNumber of NON-3.5.1.x reactions: {B}")

# extract cross-similarity matrix:
# rows = 3.5.1.x reactions
# cols = non-3.5.1.x reactions
cross_matrix = cos_sim_matrix[np.ix_(selected_rxn_indices, non_selected_indices)]

# flatten all cross-pairs
cross_pairs = cross_matrix.flatten()

# summary statistics
cross_avg = cross_pairs.mean()
cross_std = cross_pairs.std()

print(f"Number of cross-pairs analyzed: {len(cross_pairs)}")
print(f"Average similarity (3.5.1.x vs NON-3.5.1.x): {cross_avg:.6f}")
print(f"Standard deviation (3.5.1.x vs NON-3.5.1.x): {cross_std:.6f}")