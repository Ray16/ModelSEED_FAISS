"""ModelSEED mass/charge balance corrector.

A staged pipeline that proposes mass- and charge-balancing corrections for ModelSEED
reactions, reading the database strictly read-only and writing proposals into a work
directory. See mcb.cli / `python -m mcb`.
"""
