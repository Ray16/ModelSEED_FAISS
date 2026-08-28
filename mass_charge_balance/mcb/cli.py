#!/usr/bin/env python
"""
Command-line entrypoint for the ModelSEED mass/charge balance corrector.

    python -m mcb run       # full pipeline: stages 1-8 -> corrected_reactions.tsv + report
    python -m mcb verify     # per-stage balance re-check (exit != 0 on any failure)
    python -m mcb audit      # deliverable-level balance + chemistry-guard audit
    python -m mcb <stage>    # a single stage by name (see STAGES below)

All outputs go to $MCB_WORKDIR (default: the mass_charge_balance/ root). The database is
read strictly read-only. Stages 4-5 (apply_couples, apply_phospho) consume the on-disk
agent_decisions_*.tsv produced during the original agent phase; the pipeline is otherwise
fully deterministic and reproduces corrected_reactions.tsv byte-for-byte.
"""
import sys

from mcb.stages import (proton_water, cosubstrate, couple, apply_couples,
                        apply_phospho, twoog, build_log, package, predicted_ec)
from mcb import verify, audit

# ordered pipeline: name -> module with main()
STAGES = [
    ("proton_water", proton_water),
    ("cosubstrate",  cosubstrate),
    ("couple",       couple),
    ("apply_couples", apply_couples),
    ("apply_phospho", apply_phospho),
    ("twoog",        twoog),
    ("build_log",    build_log),
    ("package",      package),
]
STAGE_MAP = dict(STAGES)


def run_pipeline():
    for name, mod in STAGES:
        print(f"\n{'='*70}\n### stage: {name}\n{'='*70}")
        mod.main()


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    cmd = argv[0] if argv else "run"

    if cmd == "run":
        run_pipeline()
        print(f"\n{'='*70}\n### verify\n{'='*70}")
        rc = verify.run()
        return 1 if rc else 0
    if cmd == "verify":
        return 1 if verify.run() else 0
    if cmd == "audit":
        return 1 if audit.run() else 0
    if cmd == "predict_ec":         # opt-in, review-only; separate from the curated 516
        predicted_ec.main()
        return 0
    if cmd in STAGE_MAP:
        STAGE_MAP[cmd].main()
        return 0

    print(__doc__)
    print(f"unknown command: {cmd!r}. Valid: run, verify, audit, {', '.join(STAGE_MAP)}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
