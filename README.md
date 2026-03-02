# ModelSEED_FAISS
Fast indexing of similar reactions based on RXNFP fingerprints using FAISS

## Project Structure

| File | Purpose |
|---|---|
| `config.py` | Centralized file paths and constants |
| `utils.py` | Shared utilities (FAISS I/O, L2 normalization) |
| `0_generate_reaction_fp.py` | Generate RXNFP fingerprints for all ModelSEED reactions |
| `1_create_faiss_index.py` | Build a FAISS index from the fingerprints |
| `2_perform_similarity_search_single.py` | Single-reaction similarity search |
| `3_compute_cosine_similarity_matrix.py` | Compute full pairwise cosine-similarity matrix |
| `4_exam_cos_sim_for_EC3.5.1.x_reaction_vs_rest.py` | Analyze EC 3.5.1.x intra- vs cross-group similarity |

## Usage
Make sure that [ModelSEEDDatabase](https://github.com/ModelSEED/ModelSEEDDatabase.git) is cloned to current directory, then run the following scripts consecutively:
- `python 0_generate_reaction_fp.py`: generate RXNFP fingerprints for all ModelSEED reactions
- `python 1_create_faiss_index.py`: generate FAISS indices for all reactions
- `python 2_perform_similarity_search_single.py --rxn_name <RXN_NAME>`: perform similarity search using cosine similarity based on the generated RXNFP fingerprints. Here `<RXN_NAME>` is the name of a reaction, for example, `rxn00001`
- `python 3_compute_cosine_similarity_matrix.py`: compute a full pairwise cosine similarity matrix
- `python 4_exam_cos_sim_for_EC3.5.1.x_reaction_vs_rest.py`: compare EC 3.5.1.x intra-group similarity to cross-group similarity
