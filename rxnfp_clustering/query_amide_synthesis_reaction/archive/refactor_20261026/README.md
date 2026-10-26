# Pipeline cleanup — 26 October 2026

Nine maintained top-level implementations moved into their scientific phase
packages. Their callers, imports, subprocess paths and new-run provenance paths
now use the canonical locations. The Python 3.6 RXNFP worker has an isolated
directory to avoid shadowing its standard-library dependencies.

The redundant five-reaction score-context plot, one-off sampled legacy mapping
audit and an empty shell-created file are archived here. Previously moved
reports, slide versions and pre-OOD-gate candidate snapshots are now committed
at their archived destinations. The tracked version of the old mapping report
is preserved separately from its later locally archived revision.

- [manifest.json](manifest.json) records 33 former paths, destinations and
  SHA-256 hashes before and after relocation. Archived originals are byte-identical.
- [verification.json](verification.json) records tests, replay comparisons,
  scientific-source hashes and the 19 current artifact stages.
- `redundant_code/` contains historical implementations, not maintained commands.
- `historical_docs/` preserves the previously tracked mapping report.
- `clutter/` contains the accidental empty file.

Validation passed with **661 tests on the staged sources** and **665 tests in
the working tree**, plus 35 subtests in each. The difference is pre-existing
local candidate-evaluation work deliberately excluded from this cleanup commit.
The public utility help commands, flag forwarding, legacy RXNFP interpreter and
balanced polymerization drawing are exercised by the new CLI regression tests.

An isolated offline replay recomputed the 480-reaction screen, top 100,
chemically selected ten, 1,064 candidate identifiers, 12 supported reaction
links, 2,526 rejected search hits and 2,152 cluster-membership records. All
scientific comparisons and the complete equation audit passed. The staged
scientific implementations match the replayed sources byte for byte. A final
CLI help-forwarding adjustment was checked by both final test runs.

The full replay and diagnostic logs remain locally in this directory and are
excluded from Git. Frozen policies, database responses, model configurations,
predictions and scientific reference tables retain their original locations.
See [CODE_LAYOUT.md](../../docs/CODE_LAYOUT.md) for maintained commands and
[the archive index](../README.md) for the other historical snapshots.
