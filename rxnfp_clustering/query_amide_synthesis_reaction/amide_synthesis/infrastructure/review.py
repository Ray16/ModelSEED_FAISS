"""Paths for a review output tree, independent of executable file locations."""
from pathlib import Path
import json
import os
from .paths import PROJECT

REVIEW = Path(os.environ.get('AMIDE_REVIEW_DIR', PROJECT/'results/colleague_review_20261017')).resolve()
RUN = REVIEW

def artifact(name):
    index = REVIEW/'artifacts.json'
    paths = json.loads(index.read_text())['files'] if index.exists() else {}
    return REVIEW/paths.get(str(name), str(name))

def relative(name):
    return str(artifact(name).relative_to(REVIEW))

def require_unfrozen_review():
    if (REVIEW/'cofold/manifest.json').exists():
        raise RuntimeError('This review has a frozen modeling manifest. Set AMIDE_REVIEW_DIR to a new prepared review tree to generate a new panel.')
