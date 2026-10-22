"""Shared input/output locations and corpus configuration."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT.parent  # rxnfp_clustering, including the shared figure style
HERE = Path(os.environ.get('AMIDE_SYNTHESIS_WORKDIR', PROJECT)).expanduser().resolve()
DB = Path(os.environ.get('MODELSEED_BIOCHEMISTRY',
                         ROOT.parent / 'ModelSEEDDatabase/Biochemistry')).expanduser().resolve()
RES = HERE / 'results'
CACHE = HERE / 'cache'
FIG = HERE / 'figures'
LOGS = HERE / 'logs'
DATA = HERE / 'data'


def ensure_directories():
    """Create stage outputs when running, rather than as an import side effect."""
    for directory in (RES, CACHE, FIG, LOGS, DATA, HERE / 'embeddings'):
        directory.mkdir(parents=True, exist_ok=True)
    # ModelSEED structure fields exceed the standard CSV parser's limit.
    csv.field_size_limit(10 ** 8)


def ood_floor() -> float:
    """The 5th percentile of the reference set's own 1-NN cosine (stage search)."""
    return json.loads((RES / 'ood.json').read_text())[
        'ood_threshold_5th_pct_of_reference_1nn']
