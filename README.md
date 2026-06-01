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

## Cleanup — Structure Validation & Correction

The `Cleanup/` directory contains a pipeline for validating and correcting compound structures in the ModelSEED database against PubChem, with protonation normalization using ChemAxon pKa values.

### Pipeline overview

Run from the `Cleanup/` directory:

```bash
# Full validation (builds cache on first run)
python pubchem_validate.py

# Apply corrections and generate output files
python pubchem_validate.py --apply

# Skip protonation normalization
python pubchem_validate.py --apply --skip-ph7
```

### What it does

1. **Phase 1–2**: Resolve compound names/IDs against PubChem via cross-references and name lookups
2. **Phase 3**: Fetch PubChem structures for mismatched compounds and classify differences (STEREO_DIFF, PROTONATION_DIFF, MISMATCH)
3. **Phase 4 — Protonation validation**: Compare stored SMILES charges against ChemAxon's `db_charge` from the ModelSEED database (`compound_*.tsv`). Correct protonation using RDKit SMARTS-based deprotonation/protonation when charges disagree. Two cross-validation filters prevent incorrect corrections:
   - **pKa cross-validation**: When the compound has pKa data, skip the correction if the stored charge is closer to the pKa-predicted ionic charge than `db_charge` is
   - **Missing ionization data**: Skip correction if the compound has no pKa/pKb data but contains strongly acidic groups (phosphate, sulfonate) — indicates ChemAxon didn't compute ionization for this compound

### Key files

| File | Purpose |
|---|---|
| `pubchem_validate.py` | Main pipeline orchestrator |
| `protonation.py` | ChemAxon pKa-based protonation validation and SMILES adjustment |
| `corrections.py` | Fetch/apply PubChem corrections, normalize STEREO_DIFF protonation |
| `pka_comparison.py` | Load ChemAxon pKa/pKb/charge from `compound_*.tsv` |
| `structure_compare.py` | InChIKey comparison and structure classification |
| `reporting.py` | Generate validation reports and comparison images |
| `data_io.py` | Load structures, names, aliases from database files |

### Output files

| File | Content |
|---|---|
| `Unique_ModelSEED_Structures_modified.txt` | Corrected structures (Charged, unique format) |
| `pubchem_corrections_log.tsv` | Full changelog with timestamps, old/new values per field |
| `pubchem_protonation_corrections.tsv` | Protonation corrections with before/after SMILES and charges |
| `pubchem_validation_report.tsv` | Per-compound validation results |
| `pubchem_mismatches.tsv` | Compounds with structural mismatches against PubChem |
| `pubchem_stereo_diffs.tsv` | Compounds with stereochemistry differences |
| `pubchem_protonation_diffs.tsv` | Compounds with protonation differences |

## FAISS — Reaction Similarity Search

Fast indexing of similar reactions based on RXNFP fingerprints using FAISS.

### Project structure

| File | Purpose |
|---|---|
| `config.py` | Centralized file paths and constants |
| `utils.py` | Shared utilities (FAISS I/O, L2 normalization) |
| `0_generate_reaction_fp.py` | Generate RXNFP fingerprints for all ModelSEED reactions |
| `1_create_faiss_index.py` | Build a FAISS index from the fingerprints |
| `2_perform_similarity_search_single.py` | Single-reaction similarity search |
| `3_compute_cosine_similarity_matrix.py` | Compute full pairwise cosine-similarity matrix |
| `4_exam_cos_sim_for_EC3.5.1.x_reaction_vs_rest.py` | Analyze EC 3.5.1.x intra- vs cross-group similarity |

### Usage
Make sure that [ModelSEEDDatabase](https://github.com/ModelSEED/ModelSEEDDatabase.git) is cloned to current directory, then run the following scripts consecutively:
- `python 0_generate_reaction_fp.py`: generate RXNFP fingerprints for all ModelSEED reactions
- `python 1_create_faiss_index.py`: generate FAISS indices for all reactions
- `python 2_perform_similarity_search_single.py --rxn_name <RXN_NAME>`: perform similarity search using cosine similarity based on the generated RXNFP fingerprints. Here `<RXN_NAME>` is the name of a reaction, for example, `rxn00001`
- `python 3_compute_cosine_similarity_matrix.py`: compute a full pairwise cosine similarity matrix
- `python 4_exam_cos_sim_for_EC3.5.1.x_reaction_vs_rest.py`: compare EC 3.5.1.x intra-group similarity to cross-group similarity
