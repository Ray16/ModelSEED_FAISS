# Separate-interpreter workers

`native_similarity_rxnfp.py` is executed directly by the pinned Python 3.6 RXNFP
interpreter. It must avoid importing `amide_synthesis`, which uses the modern
scientific environment. Keep this directory free of modules named after standard
library or RXNFP dependency modules: direct script execution puts this directory
first on `sys.path`. In particular, placing this worker beside the project's
`infrastructure/http.py` breaks `urllib.request` in the legacy interpreter.

The ranking and native-reaction comparison modules supply JSON input/output
paths and explicitly bound CPU threads. No GPU is used by this worker.
