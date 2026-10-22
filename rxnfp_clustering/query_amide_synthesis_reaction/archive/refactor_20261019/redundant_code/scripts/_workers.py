#!/usr/bin/env python
"""Compatibility entry point for amide_synthesis.workers."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from amide_synthesis.workers import *  # noqa: F401,F403,E402
