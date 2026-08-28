"""Single source of truth for where the pipeline reads and writes.

All correction proposals and reports live in WORKDIR (the mass_charge_balance/ root),
NOT inside the package. Override with $MCB_WORKDIR to run against a scratch directory.
The ModelSEED database path lives in mcb.io (read-only).
"""
import os
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent           # .../mass_charge_balance/mcb
WORKDIR = Path(os.environ.get("MCB_WORKDIR", str(PKG_DIR.parent)))  # .../mass_charge_balance


def wp(name: str) -> str:
    """Absolute path to an output/intermediate file in the work directory."""
    return str(WORKDIR / name)
