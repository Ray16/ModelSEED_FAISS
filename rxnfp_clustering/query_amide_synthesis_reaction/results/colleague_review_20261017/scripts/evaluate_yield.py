"""Compatibility entry point; implementation lives in the scientific package."""
from pathlib import Path
import sys
PROJECT = next(parent for parent in Path(__file__).resolve().parents if (parent/'amide_synthesis').is_dir())
sys.path.insert(0, str(PROJECT))
from amide_synthesis.evaluation.yield_validation import main

if __name__ == '__main__':
    raise SystemExit(main())
