# UniProt–ModelSEED Mapping Generation Report

## Overview

This report documents how the project generates mappings between UniProt proteins and ModelSEED reactions. The primary mapping is stored in:

```text
UniProt/uniprot-msd-links.tsv
```

The pipeline does not directly infer a reaction from a protein sequence. Instead, it combines curated UniProt reaction annotations with the Rhea-to-ModelSEED reaction aliases maintained by the local ModelSEED database. A second, indirect route uses UniProt Gene Ontology (GO) annotations and a local Rhea-to-GO table. The two routes are merged into the final UniProt-to-ModelSEED link file.

## Mapping concepts

- **UniProt accession**: the protein identifier, for example `A4D2B8`.
- **Rhea reaction**: a chemically explicit reaction identifier, usually represented as `RHEA:<number>` in UniProt data.
- **ModelSEED reaction**: a ModelSEED biochemical reaction identifier, for example `rxn00062`.
- **Rhea–ModelSEED alias**: the relationship loaded from ModelSEED's reaction aliases. One Rhea reaction can map to multiple ModelSEED reactions, and one UniProt protein can therefore map to multiple ModelSEED reactions.

The resulting relationship is therefore best interpreted as an evidence-based annotation link, not necessarily a one-to-one enzyme assignment.

## End-to-end pipeline

```text
Fetch_UniProt_Identifiers.py
        |
        v
uniprot_rhea_results.json
        |
        v
parse_results.py
        |
        v
uniprot-rhea.tsv
        |
        +------------------------------+
        |                              |
        v                              v
uniprot_msd_mappings.py       UniProt_GO_Query.py
        |                              |
        v                              v
uniprot-rhea-msd.tsv          uniprot_geneontology_results.json
                                       |
                                       v
                               pair_uniprot_go_tsv.py
                                       |
                                       v
                               uniprot-go-rhea-msd.tsv
        +------------------------------+
        |
        v
merge-up-ms-links.py
        |
        v
uniprot-msd-links.tsv
```

## 1. Download UniProt entries with Rhea annotations

Script:

```text
UniProt/Fetch_UniProt_Identifiers.py
```

The script queries the UniProt REST API at:

```text
https://rest.uniprot.org/uniprotkb/search
```

Its query requests reviewed UniProtKB entries containing Rhea annotations:

```text
(rhea:*) AND (reviewed:true)
```

The requested fields are:

- UniProt accession
- Rhea/reaction annotation data
- Protein sequence

The script follows UniProt pagination links using the cursor returned in the HTTP `Link` header. The collected entries are saved as:

```text
UniProt/uniprot_rhea_results.json
```

This file is the source for the direct UniProt-to-Rhea mapping route.

## 2. Parse UniProt reaction annotations

Script:

```text
UniProt/parse_results.py
```

For each JSON entry, the script extracts:

1. `primaryAccession` as the UniProt identifier.
2. Reaction cross-references found in `comments[*].reaction.reactionCrossReferences`.
3. PubMed evidence identifiers associated with reaction annotations.
4. The UniProt protein sequence.

The reaction cross-references are retained as a comma-separated list. The output is:

```text
UniProt/uniprot-rhea.tsv
```

The effective columns are:

| Column | Meaning |
|---|---|
| 1 | UniProt accession |
| 2 | Comma-separated reaction cross-references, including Rhea and other databases such as ChEBI |
| 3 | PubMed evidence identifiers |
| 4 | Protein sequence |

Only entries whose reaction list contains a Rhea identifier can contribute to the direct Rhea-to-ModelSEED mapping. The parser preserves non-Rhea cross-references in the intermediate file, but they are not converted by `uniprot_msd_mappings.py`.

## 3. Convert Rhea identifiers to ModelSEED reactions

Script:

```text
UniProt/uniprot_msd_mappings.py
```

This is the step that uses the local ModelSEED biochemical database. It adds the ModelSEED Python library to `sys.path` and loads:

```python
from BiochemPy import Reactions
reactions_helper = Reactions()
reactions_dict = reactions_helper.loadReactions()
reactions_aliases = reactions_helper.loadMSAliases()
```

The script builds a reverse lookup table from ModelSEED reaction aliases:

```text
Rhea reaction ID -> one or more ModelSEED reaction IDs
```

Obsolete ModelSEED reactions are skipped. For every row in `uniprot-rhea.tsv`, the script examines the comma-separated reaction references and selects references containing `RHEA`. The `RHEA:` prefix is removed before lookup. Every matching ModelSEED reaction is emitted along with:

- UniProt accession
- numeric Rhea identifier
- ModelSEED reaction identifier
- protein sequence

The output is:

```text
UniProt/uniprot-rhea-msd.tsv
```

Its columns are:

| Column | Meaning |
|---|---|
| 1 | UniProt accession |
| 2 | Rhea identifier without the `RHEA:` prefix |
| 3 | ModelSEED reaction ID |
| 4 | Protein sequence |

### Important behavior

The conversion is alias-based. It does not perform sequence alignment, BLAST, embedding similarity, EC-number prediction, or reaction-chemistry matching during this step. If a Rhea alias corresponds to several valid ModelSEED reactions, all of them are retained.

## 4. Generate the GO-assisted mapping route

### 4.1 Obtain UniProt entries with relevant GO annotations

Script:

```text
UniProt/UniProt_GO_Query.py
```

Before querying UniProt, the script reads:

```text
UniProt/rhea2go.tsv
```

It collects the GO IDs in the final column and queries reviewed UniProt entries with GO annotations. The query requests accession, GO, GO ID, and sequence data. UniProt results are then filtered locally: an entry is retained when it contains at least one GO cross-reference present in `rhea2go.tsv`.

The output is:

```text
UniProt/uniprot_geneontology_results.json
```

### 4.2 Join GO IDs to Rhea and ModelSEED reactions

Script:

```text
UniProt/pair_uniprot_go_tsv.py
```

This script performs three joins:

```text
GO ID -> Rhea ID -> ModelSEED reaction ID
```

The first join is built from `rhea2go.tsv`. The second is rebuilt from ModelSEED's reaction aliases using the same `BiochemPy.Reactions` methods as the direct mapping script. Obsolete ModelSEED reactions are excluded.

For every qualifying UniProt entry, the script collects its relevant GO IDs, maps those GO IDs to Rhea IDs, maps the Rhea IDs to ModelSEED reactions, removes duplicates, and writes the combinations to:

```text
UniProt/uniprot-go-rhea-msd.tsv
```

Its columns are:

| Column | Meaning |
|---|---|
| 1 | UniProt accession |
| 2 | GO ID |
| 3 | Rhea identifier without the `RHEA:` prefix |
| 4 | ModelSEED reaction ID |
| 5 | Protein sequence |

This route is indirect: the UniProt entry is linked to a ModelSEED reaction through a shared GO-to-Rhea association rather than through a direct UniProt reaction cross-reference.

## 5. Merge direct and GO-assisted mappings

Script:

```text
UniProt/merge-up-ms-links.py
```

The script reads both intermediate mapping files:

```text
uniprot-go-rhea-msd.tsv
uniprot-rhea-msd.tsv
```

It builds:

```text
UniProt accession -> unique ModelSEED reaction IDs
ModelSEED reaction ID -> UniProt accessions
UniProt accession -> protein sequence
```

Duplicate UniProt–ModelSEED pairs are removed while preserving the first sequence observed for each UniProt accession. The final output is:

```text
UniProt/uniprot-msd-links.tsv
```

The final file has three tab-separated columns:

| Column | Meaning |
|---|---|
| 1 | UniProt accession |
| 2 | Comma-separated ModelSEED reaction IDs |
| 3 | UniProt protein sequence |

Example structure:

```text
UniProt_ID    rxn00062,rxn49384    amino-acid-sequence
```

There is no header row in the generated file. Consumers should treat the second column as a comma-separated list of ModelSEED reaction IDs.

## 6. Model/template reaction summaries

The final mapping is used to summarize UniProt coverage of ModelSEED reactions.

### Model reaction summary

Script:

```text
Gapfilling/find-model-rxns-in-uniprot.py
```

This script:

1. Loads ModelSEED model JSON files in `Gapfilling/ecoli-files/`.
2. Extracts reaction IDs from model reactions.
3. Removes compartment suffixes such as `_c0`.
4. Looks up each base reaction ID in `../UniProt/uniprot-msd-links.tsv`.
5. Writes the reaction and associated UniProt IDs to:

```text
Gapfilling/reaction_model_uniprot_summary.tsv
```

Reactions without a mapping are written with `None`.

### Template reaction summary

Script:

```text
UniProt/find-template-rxns-in-uniprot.py
```

This script loads ModelSEED template JSON files, extracts template reactions, removes compartment suffixes, and joins the resulting reaction IDs to `uniprot-msd-links.tsv`. It writes:

```text
UniProt/reaction_template_uniprot_summary.tsv
```

The script also records which templates contain each reaction and the reaction types. Its current writer emits four values per data row while the declared header contains three labels; downstream users should inspect this formatting before relying on the header positionally.

## 7. Model gene-to-UniProt mappings

Script:

```text
Gapfilling/model-gene-uniprot-mappings.py
```

This is a separate, downstream sequence-similarity mapping. It does **not** generate the UniProt–ModelSEED reaction links in `uniprot-msd-links.tsv`.

It combines:

- Model JSON files in `Gapfilling/ecoli-files/`
- Protein embedding result files ending in `-protein.faa.results.json`

For each embedding query, it takes the reported `best_hit` UniProt ID and score, then associates that hit with genes extracted from each model reaction's gene–protein–reaction rule. Reaction compartment suffixes are stripped before output. The result is:

```text
Gapfilling/reaction_gene_uniprot_mapping.tsv
```

Its columns are:

```text
Genome_ID    Reaction_ID    Gene_ID    UniProt_ID    UniProt_Score
```

This file answers a different question: which UniProt sequence was the best embedding hit for a model gene associated with a reaction? It should not be confused with the curated/alias-based reaction mapping in `uniprot-msd-links.tsv`.

## Current generated-file inventory

The repository currently contains the following key outputs:

| File | Role |
|---|---|
| `UniProt/uniprot_rhea_results.json` | Raw UniProt API results for reviewed Rhea-annotated entries |
| `UniProt/uniprot-rhea.tsv` | Parsed UniProt-to-reaction cross-references and sequences |
| `UniProt/uniprot-rhea-msd.tsv` | Direct UniProt-to-Rhea-to-ModelSEED mappings |
| `UniProt/uniprot_geneontology_results.json` | UniProt entries selected using relevant GO terms |
| `UniProt/uniprot-go-rhea-msd.tsv` | GO-assisted UniProt-to-Rhea-to-ModelSEED mappings |
| `UniProt/uniprot-msd-links.tsv` | Merged final UniProt-to-ModelSEED mapping |
| `Gapfilling/reaction_model_uniprot_summary.tsv` | Model reaction coverage by UniProt IDs |
| `Gapfilling/reaction_gene_uniprot_mapping.tsv` | Model gene/reaction associations to embedding-derived UniProt hits |

At the time of writing, the key files contain approximately:

| File | Lines |
|---|---:|
| `uniprot_rhea_results.json` | 18,114,057 |
| `uniprot-rhea.tsv` | 234,984 |
| `uniprot-rhea-msd.tsv` | 451,534 |
| `uniprot-go-rhea-msd.tsv` | 712,111 |
| `uniprot-msd-links.tsv` | 262,380 |
| `reaction_model_uniprot_summary.tsv` | 1,699 |
| `reaction_gene_uniprot_mapping.tsv` | 128,188 |

These counts are file-line counts, not necessarily counts of unique proteins, reactions, or UniProt–ModelSEED pairs. The intermediate files can contain multiple rows per protein because a protein may have multiple GO, Rhea, or ModelSEED associations. The final file consolidates ModelSEED reaction IDs into a comma-separated list per UniProt accession.

## Reproducibility and caveats

1. **External data are time-dependent.** UniProt API results and the local ModelSEED database can change. Re-running the scripts may produce different mappings.
2. **The ModelSEED library path is machine-specific.** The mapping scripts expect:

   ```text
   /Users/seaver/Seaver_Lab/Git_Repos/ModelSEEDDatabase/Libs/Python/
   ```

   This path must be changed on another machine.

3. **The template path is also machine-specific.** `find-template-rxns-in-uniprot.py` expects ModelSEED templates under:

   ```text
   /Users/seaver/Seaver_Lab/Git_Repos/ModelSEEDTemplates/templates/
   ```

4. **Relative working directories matter.** Several scripts open files using paths such as `uniprot-rhea.tsv`, `uniprot-msd-links.tsv`, or `ecoli-files`. They should be run from their containing directory unless paths are updated.
5. **Rhea aliases may be non-unique.** A single Rhea identifier can produce multiple ModelSEED reaction IDs. The pipeline retains these alternatives rather than selecting one.
6. **GO mappings are indirect.** A GO-associated mapping is not equivalent in strength or specificity to a direct UniProt reaction annotation.
7. **The final file has no header.** The expected schema is `UniProt accession`, `ModelSEED reaction list`, and `protein sequence`.
8. **The scripts do not preserve provenance in the final file.** After merging, `uniprot-msd-links.tsv` does not indicate whether each pair came from the direct Rhea route, the GO route, or both. Provenance remains available in the two intermediate files.
9. **A credential is present in `fetch-uniprot-via-proxy.py`.** That script contains a BERDL API token in source code and should be treated as sensitive. It is a rescue/augmentation workflow for missing reactions, not part of the original direct mapping chain described above.

## Summary

The primary UniProt–ModelSEED mapping is generated by joining curated UniProt reaction annotations to ModelSEED reactions through Rhea aliases. A complementary GO-based route expands coverage by joining UniProt GO annotations to Rhea through `rhea2go.tsv`, then applying the same ModelSEED alias table. `merge-up-ms-links.py` combines both routes into `UniProt/uniprot-msd-links.tsv`, which is then used to assess reaction coverage in models and templates. Sequence-embedding scripts provide a separate model-gene-to-UniProt mapping and should be considered downstream evidence rather than part of the core reaction alias mapping.
