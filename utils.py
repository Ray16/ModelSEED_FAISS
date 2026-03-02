"""
Shared utility functions for FAISS index management and vector normalization.
"""

import ast
import os
from typing import List, Optional

import faiss
import numpy as np


def _clean_ec(ec: str) -> str:
    """Normalize a single EC number string.

    Handles leading/trailing whitespace and 'EC-' prefixes found in
    the original ModelSEED data (e.g. ' 1.14.11.13' -> '1.14.11.13',
    'EC-2.3.1.122' -> '2.3.1.122').
    """
    ec = ec.strip()
    if ec.upper().startswith("EC-"):
        ec = ec[3:]
    return ec


def parse_ec_numbers(val) -> List[str]:
    """Parse an ec_numbers value from the CSV back into a list of strings."""
    if not isinstance(val, str) or val.strip() == "":
        return []
    try:
        result = ast.literal_eval(val)
        if not isinstance(result, list):
            return []
        cleaned = []
        for ec in result:
            c = _clean_ec(str(ec))
            if c and c not in cleaned:
                cleaned.append(c)
        return cleaned
    except (ValueError, SyntaxError):
        return []


def l2_normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize a set of vectors in-place (making their magnitude 1.0).

    After normalization, inner-product search is equivalent to cosine similarity.
    """
    faiss.normalize_L2(vectors)
    return vectors


def save_faiss_index(index: faiss.Index, filename: str) -> None:
    """Save a FAISS index to *filename*."""
    try:
        faiss.write_index(index, filename)
        print(f"Successfully saved FAISS index to: {filename}")
    except Exception as e:
        print(f"Error saving FAISS index: {e}")


def load_faiss_index(filename: str) -> Optional[faiss.Index]:
    """Load a FAISS index from *filename*, returning ``None`` on failure."""
    if not os.path.exists(filename):
        print(f"FAISS index file not found: {filename}")
        return None
    try:
        index = faiss.read_index(filename)
        print(f"Successfully loaded existing FAISS index from: {filename}")
        return index
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return None
