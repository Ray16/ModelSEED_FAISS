# Package refactor — 2026-10-09

The implementation now lives in `amide_synthesis/`. The numbered stage scripts
and old helper import paths remain compatibility shims. Query definitions,
structure normalization, fingerprint encoding, mechanism admission rules,
priority weights, and per-reaction protein quotas retain their existing behavior.
Unused weaker screen heuristics were removed so there is one active acylation
implementation.

`enzyme_resolution.py` separates product-name extraction, OOD selection, evidence
routes, ranking, de-duplication, and taxonomy slot filling from API and file I/O.
The resolver accepts a search callable, returns lookup failure/truncation metadata,
and copies search rows before adding evidence provenance. Candidate scoring is
also exposed as `build_candidate_table()`.

`pipeline.py` keeps dependency order, uses the calling interpreter (with a
`--python` / `PIPELINE_PYTHON` override), validates `--only`, streams unbuffered
stage output to flushed logs, and offers `--dry-run`. Output directories are
created when a stage runs, rather than when helpers are imported. Work and
Biochemistry paths are configurable; see the README.

Intentional fixes:

- `--offline` now disables all HTTP requests, including uncached InterPro names.
  Native-substrate analysis can still run from cached enzyme annotations.
- Failed family-name requests no longer poison the cache with empty names.
- UniProt pagination finds the next Link relation even when another relation
  precedes it, handles header casing, and preserves multiline TSV fields.
- Empty enzyme exports overwrite previous files and retain their column schemas.
  Empty candidate/native reports work; a complete lookup outage exits with failure.
- The candidate table applies the current OOD gate when reusing enzyme outputs.
- Missing Pfam annotations do not become an artificial `nan` family.
- Unresolved reactions retain a `lookup_status` column.

Validation uses no live UniProt refresh. The 24 existing curated controls pass;
34 unittest cases cover those controls plus offline retrieval, ranking, pagination,
outages, empty exports, stage ordering, cache behavior, and import side effects.

A baseline run and refactored run used the existing ModelSEED corpus and cached
protein annotations in separate temporary output directories (four local workers).
Queries, neighbours, EC votes, OOD report, screen summary, screened reactions,
per-EC candidates, ranked candidates, and top FASTA are byte-identical. The
shortlist remains 118 proteins. Native reaction records agree after treating
missing strings consistently. Family output removes one spurious `nan` group
containing four unannotated proteins (18 rows -> 17). All six offline stages,
including the three figures, completed; the cached t-SNE labels all 10 candidates.
Existing project results and figures were preserved during verification.

Remaining limitations are the scientific/retrieval ones from the earlier pass:
truncated broad EC pools, unmeasured name-route recall, cosine dominated by
cofactors, and no experimental validation on the nylon monomers. This refactor
does not change those conclusions.

---

# Refactor notes — 2026-10-09

Written for whoever refactors this next. Covers what moved, what changed in
behaviour, and what was deliberately left alone.

## New shared modules

| Module | Owns | Replaced |
|---|---|---|
| `scripts/_paths.py` | `HERE`/`ROOT`/`DB`/`RES`/`CACHE`/`FIG`/`LOGS`, directory creation, `csv.field_size_limit`, `ood_floor()` | the same 4–6 line preamble re-derived in 11 files |
| `scripts/_http.py` | User-Agent, retry count, backoff, and the failed-vs-empty distinction | three separate `urlopen` call sites, two with **no retry at all** |
| `scripts/_uniprot.py` | the UniProtKB search client: pagination, saturation reporting, deterministic ordering | `fetch()` inlined in stage 04 |

Two latent bugs surfaced while consolidating, both now fixed:

- `csv.field_size_limit` was `10**7` in stage 04 and `10**8` everywhere else, so
  the widest ModelSEED rows would have failed to parse in exactly one stage.
- `06_native_substrates.py` imported `urlopen`/`Request` and never used them.

## Behaviour changes (outputs differ; this is intended)

1. **OOD gate.** Stage 04 refuses reactions scoring below the floor stage 01
   computed (0.5041). 17 reactions dropped. They resolved to real proteins, so
   in the final table they were indistinguishable from genuine candidates —
   the one way this output actively misled.
2. **Name route scoped to `protein_name`.** It was an unfielded, i.e. full-text,
   search over the whole entry, so `"capsaicin"` returned COX-2 and glutamine
   synthetase. All 9 surviving name-route hits now carry the expected
   IucA/IucC (PF04183) domain — verified independently of the name that found them.
3. **Taxonomy-stratified slot fill** (`MAX_EUKARYOTE = 3`). Eukaryotes were
   taking 56% of rows (37 human, 31 Drosophila, 25 mouse) ahead of the bacterial
   enzymes the project is actually about. The cap only applies when prokaryotes
   are available to fill the slots.
4. **Protein-existence ranking key.** An experimentally observed protein now
   outranks a genome prediction. This matters most on the name route, where
   every hit is unreviewed so the reviewed-status key cannot discriminate: of
   twelve alcaligin synthetases only AlcC (`Q7W557`) is "Evidence at protein
   level". 144 of 208 rows are now protein-level evidence.
5. **Deterministic ordering.** `_uniprot.search` returns accession-sorted rows
   and the stage-04 sort ends in a total key. Previously, entries tied on every
   ranking key kept UniProt's relevance order, which is not stable across runs.
6. **Saturation is reported, not silent.** Page budget raised 3 → 5, and a query
   whose pool still exceeds it prints `[TRUNCATED]`. 9 reactions currently hit
   it; for those, which proteins you see is decided by UniProt's relevance
   ranking inside a truncation.

Net: 285 rows / 152 accessions → **208 rows / 118 accessions / 55 reactions**.
All seven README headline enzymes survive (BibC, PubC, AsbA, AsbB, SfnaD, PuuA, AlcC).

## Logs

`logs/<stage>.log`, written by `run_all.py`, streamed line-by-line so a long
stage doesn't look like a hang. The four hand-named `logs_*.txt` files that were
sitting in the project root moved to `logs/` as `*.txt`; they are from earlier
runs and can be deleted.

## Tests

`scripts/test_screens.py` — 24 controls, all passing. The 20 chemistry controls
were already there; 4 were added for `product_names()`, which feeds strings
straight to UniProt, so a bad one is a bad query. Run it after touching
`_acylation.py`, `_normalize.py`, or the name route.

## Gaps identified in the earlier pass (addressed below)

- **Stage files keep their numeric names** (`04_fetch_enzymes.py`). Leading
  digits make them non-importable, which is why the tests load stage 04 through
  `importlib.util.spec_from_file_location`. Renaming to `stage04_*.py` would
  make them importable and properly testable, but touches `run_all.py`, the
  README, and every path reference. Left to keep this diff small.
- **No offline fixtures for stage 04.** Its logic can only be exercised with
  network. Fixtures for `search()` would let the OOD gate, the stratified fill
  and the existence ladder be unit-tested.
- **The 9 truncated reactions are reported, not solved.** Fixing properly means
  either narrowing those EC queries or accepting the pool and ranking within it.
- **Recall of the name route is unmeasured.** It queries `size=25` and keeps 8.
- **Cosine ranks the candidates, and it is known to be weak** — for the CoA
  query the top EC-labelled neighbours are thioester *hydrolases*. The screens,
  not cosine, do the discrimination; `ec_vote.tsv` is diagnostic only.
