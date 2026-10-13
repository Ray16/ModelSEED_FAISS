# `uniprot-msd-links.tsv` — suggested improvements

Reviewed 2026-10-12 against `UNIPROT_MODELSEED_MAPPING_REPORT.md` and the
262,380-row link table (485,495 links over 11,207 ModelSEED reactions).

The table is a genuine asset and is now ingested by this project as a
retrieval route (`amide_synthesis/precomputed_links.py`). Everything below is
a suggested improvement, ordered by value. Every claim is measured; the
measurement is given so each can be rechecked or disputed.

---

## 1. Add the MetaNetX `reac_xref` pivot — 1,777 missing reactions

The pipeline pivots exclusively through **ModelSEED's own** Rhea alias table.
MetaNetX reconciles reaction identifiers across databases, and pivoting
ModelSEED → MNXR → Rhea reaches reactions ModelSEED does not cross-reference
itself.

| measured | |
|---|---:|
| ModelSEED reactions with a Rhea alias in ModelSEED | 17,487 |
| reactions given a Rhea ID **only** via MetaNetX | **1,777** |
| of those 1,777, present in the link table | **0** |

This is systematic and entirely fixable. It is also not marginal: two of the
highest-value amide synthetases in this project sit in the gap.

| reaction | enzyme | in link table |
|---|---|---|
| `rxn21314` | BibC, bisucaberin synthase | absent |
| `rxn21308` | PubC, putrebactin synthase | absent |

Both have no Rhea and no EC in ModelSEED. MetaNetX resolves them to RHEA:79295
and RHEA:79283 — exactly the reactions cited in those proteins' own UniProt
records.

**Change:** add MetaNetX `reac_xref.tsv` (80 MB, download once) as a second
source of ModelSEED↔Rhea aliases.

---

## 2. Emit a `source` column — the two routes are currently indistinguishable

The report describes two routes merged into one file: a precise one (curated
UniProt Rhea annotation → ModelSEED alias) and a looser one (UniProt GO → a
Rhea↔GO table → ModelSEED). The output TSV has three columns — accession,
reaction list, sequence. There is no route label and no confidence, so the two
kinds of evidence are byte-identical to any consumer.

**Verified example.** The table's first row maps `A4D2B8` → `rxn00062`
("ATP phosphohydrolase"). Checked against UniProt:

- `A4D2B8` is *"Putative postmeiotic segregation increased 2-like protein 1"* —
  a **DNA mismatch-repair protein**.
- Its `Catalytic activity` field is **empty**: no curated Rhea annotation at all.
- It carries **GO:0016887, "ATP hydrolysis activity"**.

So this link can only have come from the GO route, and no consumer of the file
can tell. The same inference fires for every helicase, ABC transporter and
chaperone in the proteome.

**Change:** one extra column, `source` ∈ {`rhea_direct`, `go_inferred`}. This
is the cheapest high-value fix in the list and it also defuses item 3.

---

## 3. Flag over-annotated reactions

Generic cofactor chemistry accumulates enormous protein sets, which are
chemically true and functionally uninformative.

| reaction | proteins | name |
|---|---:|---|
| `rxn00062` | **17,801** | ATP phosphohydrolase |
| `rxn00241` | 9,338 | GTP phosphohydrolase |
| `rxn02410` | 6,070 | …:NADP+ oxidoreductase |
| `rxn02105` | 4,407 | Pentanamide amidohydrolase |

The **top 20 reactions hold 19.6% of all 485,495 links**, against a median of
**4 proteins per reaction**. A consumer that joins on reaction without
noticing this will have its result set dominated by ATP hydrolysis.

**Change:** emit `n_proteins_on_reaction`, or a boolean flag above a
threshold. Deleting the links is not necessary and probably not desirable.

*Scope note:* this is low-impact for the present project specifically, because
links are only accepted for reactions that an independent acylation screen has
proven form an amide on a free amine, and no cofactor-hydrolysis reaction can
pass it. Only 2 of 93 resolved reactions tripped the flag. It matters for
consumers without such a filter.

---

## 4. Make `reviewed:true` a column rather than a filter

The fetch query is `(rhea:*) AND (reviewed:true)`. Swiss-Prot is roughly 0.5%
of UniProtKB, so every TrEMBL enzyme is excluded by construction. The report
does not list this as a limitation and it is invisible in the output.

Measured on `rxn22343` (MurE, UDP-MurNAc-tripeptide synthetase): the link
table returns **1** protein; an unfiltered live UniProt query returns **8**.

Reviewed-only is a defensible default — curation quality is real — but it
should be a documented, reversible choice rather than a silent property of the
artefact.

**Change:** drop the filter, add a `reviewed` column.

---

## 5. Expose alias fan-out so degenerate links can be down-weighted

One Rhea reaction can alias to several ModelSEED reactions, and the report
treats the resulting links as equivalent. Measured fan-out is **mean 1.85
reactions per protein, median 1, max 62**.

A link that is 1-of-62 and a link that is 1-of-1 currently look identical,
though they are not equally informative — the first is one annotation
multiplied by alias degeneracy.

*Caveat:* I did not check whether the max-62 protein is genuinely promiscuous
or an artefact, so treat this as a request for the data rather than a
demonstrated error.

**Change:** emit the fan-out per link and let consumers decide.

---

## 6. Add fallback routes for reactions with neither Rhea nor GO

Rhea and GO are the only routes, so a reaction carrying neither is
unresolvable. Two cheap fallbacks, both weaker and both worth labelling as
such:

- **EC number** — broader than Rhea but widely populated.
- **Constrained product-name search** — low precision alone, but it is the
  only route that reaches the desferrioxamine / bisucaberin / putrebactin
  oligomerisations, which carry *only* MetaCyc and MetaNetX identifiers. The
  constraint matters: unconstrained, a name search for these returns
  siderophore *transporters*.

---

## 7. Write a provenance sidecar

The artefact records no UniProt release, no ModelSEED commit, no Rhea version
and no generation date. Both UniProt and ModelSEED change monthly, so a link
table without a snapshot identifier cannot be reproduced or invalidated.

**Change:** a small JSON sidecar with the four identifiers.

---

## Reproducing these measurements

From this directory, with `uniprot-msd-links.tsv` present:

```bash
P=/homes/rzhu/miniforge3/envs/palm/bin/python
$P -m amide_synthesis.precomputed_links    # ingest + over-annotation report
```

The MetaNetX gap (item 1) is `seed_to_rhea()` in `amide_synthesis/metanetx.py`
differenced against the link table's reaction set; the A4D2B8 check (item 2)
is a single UniProt query for `accession:A4D2B8` with the
`cc_catalytic_activity` and `go_f` fields.
