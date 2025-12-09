# ModelSEED_FAISS
Fast indexing of similar reactions based on RXNFP fingerprints using FAISS

## Usage
Make sure that [ModelSEEDDatabase](https://github.com/ModelSEED/ModelSEEDDatabase.git) is cloned to current directory, then run the following scripts consecutively:
- `python 0_generate_reaction_fp.py`: generate RXNFP fingerprints for all ModelSEED reactions
- `python 1_create_faiss_index.py`: generate FAISS indices for all reactions
- `python 2_perform_similarity_search.py --rxn_name <RXN_NAME>`: perform similarity search using cosine similarity based on the generated RXNFP fingerprints. Here <RXN_NAME> is the name of a reaction, for example, rxn_00001
