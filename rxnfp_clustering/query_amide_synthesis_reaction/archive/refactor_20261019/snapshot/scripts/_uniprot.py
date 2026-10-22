#!/usr/bin/env python
"""Compatibility entry point for amide_synthesis.uniprot."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from amide_synthesis.uniprot import *  # noqa: F401,F403,E402
