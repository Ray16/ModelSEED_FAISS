#!/usr/bin/env python
"""Thin wrapper: the family evaluation now lives in the pipeline.

It became a stage (`run_all.py --only famval`) because `shortlist` consumes its
verdict to exclude members of rejected families, and a step the pipeline
depends on should not live in `tools/`. Kept here so the documented command
still works.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from amide_synthesis.family_evaluation import main  # noqa: E402

if __name__ == '__main__':
    main(as_csv='--csv' in sys.argv)
