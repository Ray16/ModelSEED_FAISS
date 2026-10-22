#!/usr/bin/env python
"""Compatibility entry point for amide_synthesis.amide_enzymes."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from amide_synthesis.amide_enzymes import *  # noqa: F401,F403,E402

if __name__ == '__main__':
    raise SystemExit(main())
