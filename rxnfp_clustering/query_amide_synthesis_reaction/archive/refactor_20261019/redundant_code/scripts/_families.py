#!/usr/bin/env python
"""Compatibility entry point for amide_synthesis.families."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from amide_synthesis.families import *  # noqa: F401,F403,E402
