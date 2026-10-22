# Artifact maintenance

The user requires affected artifacts to stay up to date when source data, mappings,
selection rules, or code change. Regenerate dependent tables, summaries, reports,
FASTA files and figures, and check `results/artifact_freshness.json` before treating
pipeline results as current. Do not update only the source file and leave derived
results presented as current.

Preserve prior versions with provenance. A frozen scientific run remains tied to
its recorded inputs; update its status and interpretation without silently changing
its sequences, configurations or predictions. Distinguish consistency with declared
database snapshots from a new live database retrieval, and distinguish annotation
support from experimental validation.

# Visual comparisons

The user wants reaction and similarity comparisons visualized by default.
Keep figures compact and clean, with minimal text. Label each embedding and
similarity metric explicitly; separate fingerprint similarity, hand-weighted
heuristics and experimental activity. Preserve missing or inapplicable scores
as missing. Use the retained compact native-reaction layout as the reference.
For similarity comparisons retain cofactors, use a single compact heatmap,
and prefer a blue-to-red palette (blue low, red high). Avoid repeated enzyme
rows across alternative target panels; label any per-enzyme target selection.
For nylon reaction discovery use DRFP–Tanimoto to retrieve candidates, then
select the five highlighted reactions by explicit reacting-group chemistry,
spacers and retained reactive ends. Keep retrieval ranking separate from the
qualitative shortlist; other embeddings provide supporting evidence.

# Similarity output location

Keep the current artifacts directly in `results/native_nylon_similarity/`,
without timestamped or other subfolders. The user explicitly requested removing
obsolete runs here; this overrides the general prior-version retention rule for
this output directory. Reruns update the flat current outputs.
