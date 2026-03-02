"""
Centralized configuration for all file paths and constants used across the pipeline.
"""

import os

# ---------------------------------------------------------------------------
# Directory paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELSEED_DB_DIR = os.path.join(BASE_DIR, "ModelSEEDDatabase")
MODELSEED_PYTHON_LIB = os.path.join(MODELSEED_DB_DIR, "Libs", "Python")

# ---------------------------------------------------------------------------
# Data file paths
# ---------------------------------------------------------------------------
RXN_DATA_CSV = os.path.join(BASE_DIR, "rxn_data.csv")
RXN_FINGERPRINTS_NPY = os.path.join(BASE_DIR, "rxn_fingerprints.npy")
FAISS_INDEX_FILE = os.path.join(BASE_DIR, "reaction_embeddings.faiss")
PAIR_COS_SIM_NPY = os.path.join(BASE_DIR, "pair_cos_sim.npy")
QUERY_RESULT_CSV = os.path.join(BASE_DIR, "query_result.csv")
MAPPED_RXNS_CSV = os.path.join(BASE_DIR, "mapped_rxns.csv")
MAPPED_RXNS_WITH_RXN_CENTERS_CSV = os.path.join(
    BASE_DIR, "mapped_rxns_with_rxn_centers.csv"
)
QUERY_RESULT_PREFILTER_CSV = os.path.join(BASE_DIR, "query_result_prefilter.csv")

# ---------------------------------------------------------------------------
# Processing constants
# ---------------------------------------------------------------------------
FINGERPRINT_BATCH_SIZE = 1000
DEFAULT_TOP_K = 30
