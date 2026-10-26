# Code layout and commands

Run commands from `query_amide_synthesis_reaction/` in the scientific Python
environment described in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
`python -m amide_synthesis --help` lists the public commands.

| Location | Responsibility |
|---|---|
| `amide_synthesis/reaction_search/` | Corpus, target encoding, reaction retrieval and chemical selection |
| `amide_synthesis/enzyme_mapping/` | Reaction-scoped evidence, protein retrieval and annotation |
| `amide_synthesis/candidate_selection/` | Candidate merging, sequence redundancy and shortlists |
| `amide_synthesis/structure_modeling/` | Frozen model inputs, preparation, explicit launch and inspection |
| `amide_synthesis/evaluation/` | Audits, activity comparisons, geometry, figures and reports |
| `amide_synthesis/infrastructure/` | Paths, HTTP/cache clients, provenance, freshness and workers |
| `config/`, `data/` | Versioned policies/locks and local input snapshots |
| `tests/` | Regression and scientific contract checks |
| `results/`, `figures/` | Scientific outputs, raw evidence and frozen model campaigns |
| `cache/`, `embeddings/`, `logs/` | Runtime data and diagnostics |
| `archive/` | Superseded implementations, reports and before-change snapshots |

`stages.py` is the single declaration of discovery stage order, dependencies and
artifact contracts. `pipeline.py` executes those stages; `cli.py` routes public
commands. Bare discovery flags still work, and `run_all.py` remains a launcher.
The lazy aliases in `_compat.py` support older flat **package** imports without
keeping duplicate implementations.

## Former top-level scripts

The former scripts have moved into the package. Update imports to the canonical
modules below and replace direct script invocation with the public command.
Arguments are forwarded to the utility; use `--help` for its options.

| Former script | Command after `python -m amide_synthesis` | Canonical module below `amide_synthesis` |
|---|---|---|
| `rank_modelseed_nylon.py` | `rank` | `reaction_search.rank_modelseed_nylon` |
| `audit_native_reactions.py` | `evaluate native-audit` | `evaluation.audit_native_reactions` |
| `compare_native_reactions.py` | `evaluate native-similarity` | `evaluation.compare_native_reactions` |
| `draw_native_reactions.py` | `evaluate native-draw` | `evaluation.draw_native_reactions` |
| `draw_top5_native_reactions.py` | `evaluate selected-reactions` | `evaluation.draw_top5_native_reactions` |
| `draw_polymerization_target.py` | `evaluate polymerization-target` | `evaluation.draw_polymerization_target` |
| `fetch_native_enzyme_go.py` | `native-go` | `enzyme_mapping.fetch_native_enzyme_go` |

`nylon_chemical_review.py` is now the library
`amide_synthesis.reaction_search.nylon_chemical_review`. The legacy Python 3.6
RXNFP worker is directly executed at
`amide_synthesis/infrastructure/legacy_workers/native_similarity_rxnfp.py` by the separate
RXNFP interpreter. It intentionally avoids importing the modern main package.

For example:

```bash
python -m amide_synthesis discover --dry-run --phase reaction-search
python -m amide_synthesis rank --help
python -m amide_synthesis reproduce --check-only
python -m amide_synthesis evaluate native-similarity --help
```

`plot_drfp_score_context.py` hard-coded the historical five-reaction panel and
duplicates context already exported by ranking. Its original implementation is
in `archive/refactor_20261026/redundant_code/`.
The one-off sampled `enzyme_mapping/audit_legacy_mapping.py` is archived there
as well. Use `evaluate audit-mapping` for the maintained snapshot-consistency
audit and `evaluate top5-mapping` for evidence-scoped panel mapping.

## Evidence and archive boundaries

Historical data is still a dependency when a lock, mapping audit or reused
structural ensemble references it. Those records stay at their established
paths. Frozen YAMLs, sequences, predictions and scientific baselines are not
rewritten by code organization. Saved manifests retain their original hashes;
the relocation manifest in `archive/refactor_20261026/` maps former source paths
to maintained modules. New runs record the new source paths and hashes.

Use `python -m amide_synthesis report` to inspect artifact freshness and
`python -m pytest -q` to run the maintained regression suite. Archives are
excluded from test discovery. See [the archive index](../archive/README.md) for
retention details and [PIPELINE.md](../PIPELINE.md) for the scientific method.
