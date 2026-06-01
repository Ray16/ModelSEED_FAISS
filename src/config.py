"""
Centralized configuration for all file paths and constants used across the pipeline.
"""

import os

# ---------------------------------------------------------------------------
# Directory paths
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")
MODELSEED_DB_DIR = os.path.join(PROJECT_DIR, "ModelSEEDDatabase")
MODELSEED_PYTHON_LIB = os.path.join(MODELSEED_DB_DIR, "Libs", "Python")

# ---------------------------------------------------------------------------
# Data file paths
# ---------------------------------------------------------------------------
RXN_DATA_CSV = os.path.join(DATA_DIR, "rxn_data.csv")
RXN_FINGERPRINTS_NPY = os.path.join(DATA_DIR, "rxn_fingerprints.npy")
FAISS_INDEX_FILE = os.path.join(DATA_DIR, "reaction_embeddings.faiss")
PAIR_COS_SIM_NPY = os.path.join(DATA_DIR, "pair_cos_sim.npy")
QUERY_RESULT_CSV = os.path.join(DATA_DIR, "query_result.csv")
MAPPED_RXNS_CSV = os.path.join(DATA_DIR, "mapped_rxns.csv")
MAPPED_RXNS_WITH_RXN_CENTERS_CSV = os.path.join(
    DATA_DIR, "mapped_rxns_with_rxn_centers.csv"
)
QUERY_RESULT_PREFILTER_CSV = os.path.join(DATA_DIR, "query_result_prefilter.csv")
CPD_FREQ_CSV = os.path.join(DATA_DIR, "cpd_freq.csv")

# Evaluation outputs
EVAL_RESULTS_CSV = os.path.join(DATA_DIR, "evaluation_results.csv")
EC_PREDICTIONS_CSV = os.path.join(DATA_DIR, "ec_predictions_unannotated.csv")
MISANNOTATION_CSV = os.path.join(DATA_DIR, "misannotation_candidates.csv")

# Figure outputs
EVAL_PLOT_PNG = os.path.join(FIGURES_DIR, "evaluation_comparison.png")
DATASET_PIE_PNG = os.path.join(FIGURES_DIR, "dataset_pie_charts.png")
DATASET_TABLE_TEX = os.path.join(DATA_DIR, "dataset_summary_table.tex")
UNANNOTATED_PREDICTIONS_PNG = os.path.join(FIGURES_DIR, "unannotated_predictions.png")
MISANNOTATION_ANALYSIS_PNG = os.path.join(FIGURES_DIR, "misannotation_analysis.png")
ANNOTATION_SUMMARY_PNG = os.path.join(FIGURES_DIR, "annotation_summary.png")
MISANNOTATION_CORRECTION_PNG = os.path.join(
    FIGURES_DIR, "misannotation_correction_summary.png"
)

# Phylo outputs
PHYLO_SEQS_RAW = os.path.join(DATA_DIR, "sim_sequences_raw.fasta")
PHYLO_SEQS_CDHIT = os.path.join(DATA_DIR, "sim_sequences_cdhit.fasta")
PHYLO_ALIGNED = os.path.join(DATA_DIR, "sim_aligned.fasta")
PHYLO_TREE_NWK = os.path.join(DATA_DIR, "sim_phylo_tree.nwk")
PHYLO_TREE_PNG = os.path.join(FIGURES_DIR, "sim_phylo_tree.png")
PHYLO_METADATA_CSV = os.path.join(DATA_DIR, "sim_phylo_metadata.csv")
PHYLO_STATS_TXT = os.path.join(DATA_DIR, "sim_phylo_stats.txt")

# ---------------------------------------------------------------------------
# Processing constants
# ---------------------------------------------------------------------------
FINGERPRINT_BATCH_SIZE = 1000
DEFAULT_TOP_K = 30
