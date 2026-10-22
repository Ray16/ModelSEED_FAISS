#!/usr/bin/env python
"""Compatibility entry point for amide_synthesis.candidate_table."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from amide_synthesis.candidate_table import *  # noqa: F401,F403,E402

if __name__ == '__main__':
    raise SystemExit(main())
