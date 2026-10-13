# Querying ModelSEED for nylon-6,6 amide synthesis

**Question.** Is there any reaction — and therefore any enzyme — in ModelSEED
similar to the nylon-6,6 diad-forming step, the amide bond between adipate (C6
diacid) and hexamethylenediamine (HMDA, C6 diamine)?

**Answer.** No exact analogue exists, but the search converges on one family
with striking consistency: **NIS (NRPS-independent siderophore) synthetases**.
They form ATP-dependent amide bonds between carboxylic acids and **free
polyamines**, and — the part that matters most for nylon — several members
**iteratively oligomerise bifunctional amine-acid monomers** into dimers and
trimers. That is enzymatic polyamide chemistry.

Run on ModelSEEDDatabase `master` @ `b539adb6` (2026-10-06).
`results/candidates_ranked.tsv` is the deliverable: **118 candidate enzymes**
with evidence columns exposed for independent scoring. Whole pipeline: **~5 min**.

---

## Headline candidates

| Enzyme | UniProt | Organism | Cosine | Why |
|---|---|---|---:|---|
| **BibC** bisucaberin synthase (EC 6.3.2.64) | `B6EP94` | *A. salmonicida* | **0.911** | **Dimerises two identical amine-acid monomers** into a macrocyclic diamide — closest thing to a polycondensation step |
| **PubC** putrebactin synthase (EC 6.3.2.63) | `P0DXD6` | *Shewanella* sp. | 0.898 | Same iterative dimerisation, putrescine-derived monomer |
| **AsbA** spermidine–citrate ligase | `A0A0F7RJ52` | *B. anthracis* | 0.851 | ATP-dependent amide, **free polyamine + polyacid** |
| **AsbB** chain extension | `Q81RQ8` | *B. anthracis* | 0.772 | Adds a **second** polyamine to the same acid core |
| **SfnaD** D-ornithine–citrate ligase | `Q2FW70` | *S. aureus* | 0.753 | Independent NIS synthetase, same chemistry |
| **PuuA** γ-glutamylputrescine synthetase | `P78061` | *E. coli* K-12 | 0.749 | ATP-dependent amide onto a **free C4 diamine**; easiest to express and assay |
| **AlcC** alcaligin synthetase | `Q7W557` | *B. parapertussis* | 0.738 | Macrocyclic dimerising NIS synthetase |
| **SbnC / SbnF** staphyloferrin B | `Q2G1N1` / `Q2G1M8` | *S. aureus* | 0.677 | NIS synthetases |

The desferrioxamine/bisucaberin/putrebactin reactions are the most nylon-like
chemistry in the database: a **single bifunctional monomer carrying both a free
amine and a carboxylate**, joined head-to-tail by ATP-dependent amide bonds,
**iteratively** (dimer → trimer). Nylon-6,6 is the AA/BB analogue of the same
polycondensation.

**Suggested first experiment.** Offer **adipate + HMDA + ATP** to BibC/PubC (the
oligomerisers) and to PuuA (easiest expression). AsbA/AsbB cover the
two-different-monomers case directly.

**The honest gap:** these natively prefer C4–C5 hydroxamate monomers or
spermidine/putrescine, not HMDA's C6 spacing, and AsbA's native acid is citrate.
Activity on the nylon monomers is a hypothesis this analysis *generates*, not
one it validates.

---

## Retrieval architecture (current)

ModelSEED ships **no reaction -> enzyme mapping**, so breadth comes from having
several independent bridges rather than one good one. Six routes feed one
table, `results/master_candidates.tsv`, and every row records which route
produced it.

| route | what it claims | substrate distance |
|---|---|---|
| `reaction` | UniProt annotated to a ModelSEED-mappable reaction the screen PROVED acylates a free amine (Rhea / MetaNetX pivot / EC / product name) | **measured** |
| `precomputed_link` | present in a global UniProt->ModelSEED link table, filtered through the screen | measured |
| `domain` | member of a Pfam family whose chemistry is *characteristic* (enrichment test) | inherited, discounted |
| `clade` | member of a validated protein's own UniRef50 cluster, where the Pfam is too broad to mean anything | inherited, discounted |
| `assayed` | **measured** nylon-6,6 L1 titre | n/a - it is a measurement |
| `proxied` | literature-curated, carries an independent geometric score | n/a |

Measured and inherited distances are kept in separate columns and are not
interchangeable.

### Cosine is a recall route, never a gate

The encoding is still ECFP4-diff || DRFP, but a whole-reaction difference
fingerprint is dominated by the **cofactor**, not the amide bond. Ranking on it
is actively misleading - AsbA has the third-highest cosine in the shortlist and
ranks 32 of 36 on an independent geometric proxy, while PylC at cosine 0.554
ranks 11.

Worse, cosine was *gating* retrieval in three separate places. Each one was
removed:

1. `TOPN=30` per query, applied to cosine, before enzyme lookup.
2. `priority` in the candidate table, where cosine broke every tie.
3. The OOD floor in `select_reactions`, a cosine threshold that refused
   reactions the screen had already proven.

The third was the worst: it discarded **every folylpolyglutamate synthetase**
and the **meso-diaminopimelate Mur ligases** - the highest-scoring reactions in
the whole screen, all carrying a usable EC that would have resolved
immediately. Reactions are now admitted on cosine **or** on substrate
chemistry, and cosine contributes nothing to ranking.

### `substrate_distance`: scoring the chemistry, not the embedding

`amide_synthesis/substrate_distance.py` scores the acyl donor and the amine
acceptor that the screen has already proven, using no embedding:

1. **Can it polycondense at all?** The boolean screen cannot separate a
   terminal amide (glutamine synthetase: one bond, then stop) from an AA/BB
   pair (a diacid acylating a diamine - nylon-6,6) from a self-condensing
   bifunctional monomer (BibC/PubC: the product is again a substrate, so the
   same enzyme runs again, monomer -> dimer -> trimer). Only the last two
   extend a chain.
2. **Does the product stay reactive?**
3. **Are the spacers the right length?** Adipate puts 4 carbons between its
   carboxyls, HMDA puts 6 between its nitrogens. This quantifies the known gap
   to the NIS enzymes rather than asserting it in prose.
4. **Is the substrate mostly motif, or mostly scaffold?** Forming the right
   bond is not accepting the right molecule. FPGS genuinely runs iterative
   polyamide extension and scored 0.883 on bond chemistry alone - but its
   substrate is tetrahydrofolyl-polyglutamate and its pocket grips a pterin.
   Parsimony charges for the **parent** donor (32 heavy atoms there, against
   adipate's 10), which the cut acyl fragment cannot reveal.
5. **Is the backbone a plain aliphatic chain?**

Weights are flat, documented and exposed per-component, in the spirit of
`priority`: a stated chemical opinion, not a fit to data.

### Family expansion: enrichment, clades, and passengers

A family whose chemistry is validated makes every member a candidate. But
"family" means different things at different scales, and expanding every Pfam
that touches a validated protein is wrong in two opposite ways:

* **Co-occurrence.** Crediting every domain on a multi-domain protein promoted
  a peptidoglycan amidase domain (CHAP, a *hydrolase*) and a calcium-channel
  domain (RyR/IP3R homology) to "validated amide-forming families". Domains
  seen on a single seed are now dropped as passengers.
* **Scale.** Enrichment - validated seeds per 1000 members - separates a family
  whose chemistry is characteristic (DdaG 47/1k, IucA/IucC 3.5/1k) from one
  where it is incidental (GNAT 0.03/1k). But low enrichment is **not**
  irrelevance: Mur ligase (0.07) and AMP-binding (0.01) are ancient and
  universal, so only a clade is relevant and those expand by the seed's own
  UniRef50 cluster instead.

Enrichment is not sufficient either. PF06276 is shared by the NIS synthetases
and by FhuF, a ferric siderophore *reductase*, so homologs whose own annotation
contradicts the chemistry are flagged (`annotation_contradicts_chemistry`) and
demoted rather than deleted.

### Where recall stops — probed after every change, not assumed

Of 259 screened reactions, 168 resolve to at least one protein. The residual
was re-probed after each new route (the composition shifts every time one
lands, so a stale breakdown is worthless):

| | count | meaning |
|---|---:|---|
| identifier present, UniProt returns **zero** | 18 | genuinely undeposited enzymology |
| identifier present, UniProt returns hits | **0** | nothing is being lost |
| no identifier; a name was tried, UniProt had nothing | 60 | no such protein is named in UniProt |
| no identifier and no usable name | 13 | nothing left to query |

The important line is the third: **0 reactions remain where an identifier
would have worked.** Every reaction that can be resolved through Rhea, EC, a
sibling-identifier pivot, a product name or the reaction name now is. The
remaining 91 are limited by what has been deposited and named, not by
retrieval.

`rxn19370` is the clean undeposited case: the xref pivot *does* supply its
Rhea IDs (35623-35626) and UniProt *still* returns nothing for them.
D-glutamyltransferase has no deposited protein. The identifier gap and the
deposition gap are separate problems; only the first was ours.

27 of the 91 score >= 0.6 on substrate distance, so the loss is not uniformly
uninteresting - but it is not addressable from UniProt.

### The sibling-identifier pivot

"MetaCyc-only means product-name-only" turned out to be wrong as well. The
MetaNetX pivot only looked up `seed.reaction:<id>`, but MetaNetX often knows a
reaction under a *sibling* identifier while not registering it under its SEED
id at all. Pivoting from those recovers Rhea IDs the SEED pivot cannot:

```
MetaCyc  6,073      KEGG  5,067      BiGG  9,896     (foreign id -> Rhea)
```

14 of the 77 unresolved MetaCyc-bearing reactions gain a Rhea this way.
Reactions resolved 156 -> 159; 277 enzyme rows arrive via `MetaNetX-xref`.

`rxn19370` is instructive: the pivot *does* supply its Rhea IDs, and UniProt
*still* returns nothing for them. The identifier gap and the deposition gap
are separate problems, and only the first is ours to fix.

## What is NOT validated

The ranking has never been fitted to, or validated against, measured activity.

The one independent check available - a geometric score computed in a separate
project, joined read-only - shows **no correlation** with substrate distance
(Spearman +0.14, p=0.60, n=17). This is expected rather than alarming: the two
score **different steps**. The geometric score measures the carboxylate O--P
distance and in-line angle to ATP, describing *adenylation*; substrate distance
describes the *amide bond*. A ligase can activate its acid perfectly and never
accept a C6 diamine. Neither validates the other, so `struct_S` is reported
alongside and never folded into the ranking.

A second limitation: `substrate_distance` scores a **reaction**, and many
proteins map to one reaction, so all inherit the same score. `glnA2` sits at
0.8225 purely because it shares `rxn05123` with PuuA - and its geometric score
is ~0.

See `MAPPING_REVIEW.md` for the review of the external link table.

## Method

Encoding is the benchmark winner throughout — **ECFP4-diff ‖ DRFP** (Morgan r=2,
2048, count, products − reactants; DRFP 2048 folded), cosine on L2-normalised
vectors. Queries and corpus share one geometry.

Free acid + free amine condensation is not how biology makes amides, so the
target is encoded in **six mechanistic framings**. Top-1 cosine against the
EC-labelled subset (n = 13,081; OOD floor 0.504):

| Query | Top-1 cosine | OOD? |
|---|---:|---|
| Direct condensation | 0.502 | **yes** (4.9th pct) |
| ATP → AMP + PPi | 0.751 | no |
| ATP → ADP + Pi | 0.749 | no |
| Acyl-CoA aminolysis | 0.890 | no |

**The nylon step is unprecedented only in its uncatalysed form.** Posed as
activate-then-aminolyse, it lands inside populated ModelSEED territory.

### Cosine alone is not evidence of chemistry

A whole-reaction difference fingerprint is dominated by the largest
compositional change — the cofactor, not the amide bond. The CoA query's
highest-scoring EC-labelled neighbours are thioester **hydrolases** (EC 3.1.2,
agreement 0.80, no abstain), which make no amide at all. `ec_vote.tsv` is
therefore marked **diagnostic only**; mechanistic calls come from the screens.

Candidates must pass three embedding-independent screens, applied **before** any
top-N cut:

1. **A free amine is acylated — proven by reconstruction.** Product carboxamide
   C–N bonds are cut and the parent amine reconstituted; if that amine is itself
   a reactant, the linkage is established and the **acceptor is named**. Weaker
   tests fail in both directions: net amide count misses asparagine synthetase;
   amide *environment* comparison misses acetyl-CoA:putrescine
   N-acetyltransferase (a new acetamide's radius-2 environment is identical to
   CoA's own pantetheine amides); and "an amine disappears and some product has
   an amide" wrongly admits alanine dehydrogenase, where NADH coincidentally
   supplies an unrelated amide. Two further refinements: the **specific**
   nitrogen is validated (otherwise acylating a guanidino N counts whenever an
   unrelated aliphatic amine sits elsewhere in the same reactant), and **cut
   sets of up to two bonds** are tried, so acylating *both* ends of a diamine is
   recognised — under any single cut the remaining fragment is mono-acylated and
   cannot match the free diamine.
2. **The nitrogen is a free aliphatic amine** — sp3, H-bearing, bonded to sp3
   carbon, not aromatic/amide/nitrile/nitro/guanidino. The diamine flag now
   describes the **acceptor that actually reacted**, not whichever reactant
   carries the most nitrogens: a lysine donor acylating ethylamine is correctly
   a monoamine acylation.
3. **The substrate is a small molecule, not a protein** — R-group formulas and
   name stems from the compound table, matched truncation-tolerantly because
   ModelSEED truncates names mid-word (`...-E2-protei`).

Reactions that are not heavy-atom balanced, or whose imbalance is not fixable by
protons alone, are **excluded** from ranking rather than merely flagged.

Of 33,246 corpus reactions, **726** acylate a free amine;
231 of those act on a macromolecule and 22 are unbalanced, leaving
**473 eligible** for ranking, of which 92 acylate a free diamine.

`scripts/test_screens.py` holds 24 curated controls (positives, negatives and
repair cases), each one a bug that is now guarded. Run it after touching
`amide_synthesis/acylation.py` or `amide_synthesis/normalize.py`.

### Enzyme retrieval

Three routes, kept separate and labelled by `evidence_strength`:

| Route | Meaning | Count |
|---|---|---:|
| `exact_reaction` | UniProt annotated to that **Rhea** reaction | 132 |
| `activity_ec` | matching **EC** number (broader) | 67 |
| `name_match` | distinctive **product name** + synthase/ligase | 9 |

Rhea IDs come from ModelSEED's alias table **and** from **MetaNetX**
(`_metanetx.py`), which reconciles reaction identifiers across databases:
pivoting ModelSEED → MNXR → Rhea covers reactions ModelSEED does not
cross-reference. ModelSEED alone gives a Rhea ID for 62 of 101 ranked
reactions; with MetaNetX, 68. More importantly it upgrades the two best hits —
`rxn21314` (BibC) and `rxn21308` (PubC) have **no Rhea and no EC in ModelSEED**,
and MetaNetX resolves them to RHEA:79295 and RHEA:79283, which are exactly the
reactions cited in those proteins' own UniProt records. They moved from
`name_match` to `exact_reaction`; 42 associations across 19 reactions came this
way. (MetaNetX reac_xref.tsv, 80 MB, downloaded once and cached; CC-BY 4.0.)

Stage 05 restricts the table to reactions present in the **current** screen, so
a partial rerun (`--offline`) cannot resurrect candidates the screen has since
rejected.

**The recall gap is reported, not hidden.** `reactions_without_enzymes.tsv`
lists the 29 ranked reactions that resolve to no protein, split into **5 with a
Rhea ID but no annotated enzyme** (genuine orphan reactions) and **24 with no
usable identifier** (lookup-limited). Without this file, absence of a candidate
is indistinguishable from absence of an enzyme.

The name route is weakest but necessary: the highest-scoring hits
(bisucaberin/putrebactin/desferrioxamine oligomerisations) carry **only MetaCyc
and MetaNetX identifiers** — no Rhea, no EC — so the structured routes return
nothing for them, and BibC at cosine 0.911 would have been missed entirely. The
synthase/ligase constraint is required; unconstrained, the name search returns
siderophore *transporters*. Rhea and EC are both collected rather than one
short-circuiting the other, results are de-duplicated **before** the per-reaction
limit, and pagination is followed.

### Native substrates and families

Reaction similarity says a candidate does the right *kind* of chemistry; it says
nothing about what the enzyme accepts. `06_native_substrates.py` reads the cached
curated catalytic-activity statement for each candidate, keeping the evidence
code and PubMed IDs, and groups the shortlist by Pfam family (names resolved
through InterPro). **24 of the top 60 candidates have experimentally evidenced
(ECO:0000269) native reactions.**

Every candidate row carries its `family`, because the shortlist is only
interpretable by family: **16 of the top 60 are one family** (IucA/IucC,
PF04183 — the NIS synthetases), which is a convergence, not 16 independent hits.

| Enzyme | Native reaction (experimental) | PubMed |
|---|---|---|
| BibC | 2 × succinyl-N-hydroxy-cadaverine + 2 ATP → bisucaberin | 18956041 |
| PubC | 2 × succinyl-N-hydroxy-putrescine + 2 ATP → putrebactin | 18630910 |
| AsbA | spermidine + citrate + ATP → N8-citryl-spermidine | 17579415, 19259597, 22408253 |
| AsbB | N8-citryl-spermidine + spermidine + ATP | 18758617, 22408253 |
| PuuA | putrescine + L-glutamate + ATP → γ-Glu-putrescine | 18495664 |

**The key observation.** BibC's and PubC's native substrate is a *diamine–diacid
half-amide* — free carboxylate at one end, free primary amine at the other —
which is the same structural class as the nylon-6,6 diad this project encodes as
a **product** (see `figures/fig_substrate_gap.png`). These enzymes natively
dimerise that motif. So the experiment is not only "make the diad" but **feed
the pre-formed diad to BibC/PubC and look for dimerisation** — the
chain-extension step of nylon-6,6. The gap is specific: native monomers are
N-hydroxylated and two carbons shorter per arm.

---

## Performance

Fully parallelised; ~5 min end-to-end on an idle-ish 80-core node.

| Stage | Time | Parallelism |
|---|---:|---|
| corpus (structures + 33k × ECFP4/DRFP) | ~75 s | process pool, ECFP4 and DRFP interleaved |
| search | ~9 s | BLAS matmul |
| screen (33k reactions) | ~16 s | process pool |
| enzymes (UniProt) | ~35 s | thread pool, **capped at 6** — polite to a public API |
| table | ~1 s | — |
| figure (t-SNE) | ~170 s cold, **~9 s warm** | content-keyed cache |

Worker count is **adaptive** (`scripts/_workers.py`): it measures the 1-minute
load average and claims 70% of genuinely idle cores, floor 4, **cap 48**. On a
busy node it backs off automatically; `N_WORKERS=48` overrides. Per-process BLAS
threads are pinned to 4 so a wide fan-out cannot oversubscribe a shared node.
UniProt concurrency is deliberately *not* scaled with cores.

---

## Data corrections found along the way

**1. Query construction bugs** (caught by balance assertions): the ATP and CoA
framings were charge/H-imbalanced, and a constructed adipyl-CoA had **lost its
sulfur**. Query species are now read from the database, and each query solves
for the H⁺ it needs and asserts balance before encoding.

**2. The 2026-10-06 structure refresh is chemically wrong for amides.** ~17% of
compounds — including NAD, NADH, NADP, CoA, every acyl-CoA, nicotinamide and
N-acetylputrescine — write carboxamides as **iminols**, in two forms: anionic
`C([O-])=N` (~5,100) and neutral `C(O)=N` (~665), with adenine N1 as `[nH+]`.
Both are wrong at pH 7, and an iminol is invisible to an amide SMARTS.
`_normalize.py` repairs all three defects (6,313 compounds) and reproduces the
previous snapshot's NAD and CoA exactly.

> The neutral form was found only because the single most relevant positive
> control — acetyl-CoA:putrescine N-acetyltransferase — silently failed the
> screen after the first repair pass.

⚠️ **The repair is not charge-neutral**: 5,807 compounds change formal charge
(ATP −3→−4, CoA −5→−4). ModelSEED's H⁺ coefficients were balanced against the
*original* structures. The corpus therefore audits **heavy-atom** balance
(repair-invariant) and separately records **proton-fixability** (H delta must
equal charge delta): 4,025 of 33,359 reactions are not heavy-balanced and 6,363
are not proton-fixable.

**3. Proton convention.** The corpus strips H⁺; queries balance *with* protons
then strip. Mixing conventions measurably changes scores (~0.97 cosine between
the two encodings of the same reaction).

**4. ModelSEED branch.** `origin/master` and `origin/dev` have byte-identical
`Biochemistry/` content, both last touched 2026-10-06.

---

## Reproducibility

The corpus manifest is written **last** and records git commit, counts, RDKit
version, and SHA-256 of both the ordered reaction table **and** the embedding
array; `_corpus.py` verifies both on load, so reordered-but-same-shape data
cannot slip through. Corpus and embeddings are published atomically. The t-SNE
cache is keyed by corpus hash plus query SMILES, not row count.

## Caveats

- Screen 1 proves the amine→amide linkage by reconstruction, but it is still a
  structural inference, not atom-mapped. It deliberately excludes amide
  *transfer* (asparagine synthetase): the nitrogen there arrives already
  amide-bound, a different mechanism from acylating HMDA.
- **t-SNE position is indicative only.** Coordinates are not comparable with
  `figures/tecrdb_shift/fig_tsne_ecfp4_drfp.png`, and t-SNE does not preserve
  cosine rank. **The cosine ranking is authoritative; the map is for orientation.**
- Figure cosines are against the full corpus; the query table is against the
  EC-labelled subset. Different reference sets, different numbers.
- Similarity is to *reactions*, not evidence of substrate promiscuity.
- `priority` is a transparent placeholder ordering (evidence strength, diamine
  acceptor, prokaryote, PDB, cosine) meant to be replaced by your metric.
- Metadata and aliases are still read live from the database rather than pinned
  to the manifest; UniProt pagination is capped at 5 pages and reports truncation.
- Simultaneous acylation is detected up to **two** bonds per product; a reaction
  acylating three or more sites at once would be missed.
- The figure labels the top 10 candidates **by cosine**, not by `priority`, so
  the name-route hits (BibC, PubC) remain visible. Labels are placed by a
  collision-avoidance search; one with no free position is dropped rather than
  overlapped, though all 10 currently place.

## Files

```
amide_synthesis/normalize.py       repair iminol / [nH+] structure defects
amide_synthesis/acylation.py       reconstruct the reacted amine acceptor
amide_synthesis/corpus.py          verify corpus and embedding against manifest
amide_synthesis/workers.py         load-aware worker sizing (cap 48)
amide_synthesis/paths.py           configurable input/output paths
amide_synthesis/pipeline.py        ordered stage runner and streamed logs
amide_synthesis/enzyme_resolution.py  evidence routes, OOD gate, protein selection
amide_synthesis/rebuild_corpus.py  reactions + ECFP4/DRFP from the checkout
amide_synthesis/encode_and_search.py  encode queries + cosine NN
amide_synthesis/amide_enzymes.py    mechanism screens + per-EC shortlist
amide_synthesis/fetch_enzymes.py    UniProt retrieval and enzyme exports
amide_synthesis/candidate_table.py  prioritised candidate table
amide_synthesis/native_substrates.py  native reactions + family grouping
amide_synthesis/deck_figures.py    family and substrate-gap figures
amide_synthesis/project_tsne.py    t-SNE map (runs last)
amide_synthesis/http.py            shared retries and offline guard
amide_synthesis/uniprot.py         paginated protein search
amide_synthesis/families.py        cached Pfam names via InterPro
amide_synthesis/metanetx.py        ModelSEED -> MNXR -> Rhea identifiers
scripts/00_*.py ... 07_*.py        compatibility entry points
scripts/_*.py                     compatibility imports for shared helpers
scripts/test_screens.py           24 curated chemistry / product-name controls
tests/                            offline regression suite (unittest)

results/candidates_ranked.tsv   <- THE DELIVERABLE: 118 enzymes + evidence
results/candidates_top.fasta    sequences for the top 25
results/enzyme_annotations.tsv  GO, cofactor, pathway, Pfam, sites, PDB
results/amide_forming_neighbours.tsv  screened reactions, ranked per query
results/screen_summary.json     how many reactions each screen admits
results/native_substrates.tsv   native reaction per enzyme, with PubMed evidence
results/enzyme_families.tsv     candidates collapsed to Pfam families
results/reactions_without_enzymes.tsv  the recall gap, orphan vs lookup-limited
results/corpus_manifest.json    git commit, counts, content hashes
figures/fig_tsne_nylon_query.png   map, candidate enzymes labelled
figures/fig_families.png           shortlist by protein family
figures/fig_substrate_gap.png      native monomer vs the nylon diad
slides/main.tex                    10-slide deck (seahorse, house style)
slides/main.pdf
```

Reproduce:

```bash
cd rxnfp_clustering/query_amide_synthesis_reaction
P=/homes/rzhu/miniforge3/envs/palm/bin/python     # env `palm` has rdkit + drfp
$P run_all.py                        # full pipeline, ~5 min
$P run_all.py --skip-corpus          # reuse corpus + embeddings
$P run_all.py --skip-corpus --offline  # reuse UniProt files; no HTTP requests
$P run_all.py --only screen table    # single stages
N_WORKERS=48 $P run_all.py           # override the adaptive worker count
$P scripts/test_screens.py           # original 24 controls
$P -m unittest discover -s tests -v   # offline regression suite
$P -m amide_synthesis --dry-run       # list the stages without running them
cd slides && pdflatex main.tex       # rebuild the deck (twice for refs)
```


## Refactored interfaces

The stage implementations are importable under `amide_synthesis`; the numeric
scripts are thin compatibility entry points. Run commands from this directory.
The pipeline now uses the Python interpreter that invoked it. `--python PATH`
or `PIPELINE_PYTHON=PATH` selects a different stage interpreter. Stage names in
`--only` are validated and execute in pipeline order; inputs from omitted stages
must already exist.

`--offline` skips live enzyme retrieval, reuses `enzymes_uniprot.tsv`,
`enzyme_annotations.tsv`, and `enzymes.fasta`, and still rebuilds the table,
native-substrate reports, and figures. Pfam names come from the existing cache;
unseen domains keep their accessions as labels and are not cached as failures.
The current screen and OOD threshold gate cached enzyme rows as well.

Set `AMIDE_SYNTHESIS_WORKDIR` to put data, embeddings, results, cache, figures,
and logs in a separate working directory. Set `MODELSEED_BIOCHEMISTRY` to use a
different Biochemistry checkout. Neither setting changes the query chemistry.
For example, populate a working directory with corpus/embedding inputs,
`results/corpus_manifest.json`, cached enzyme files, and optional caches before
running `--skip-corpus --offline`. A full run creates its own corpus inputs.

```python
from amide_synthesis.enzyme_resolution import rank_entries, resolve_reaction
from amide_synthesis.candidate_table import build_candidate_table

# Supply a UniProt-compatible callable returning (rows, failed, saturated).
resolution = resolve_reaction(hit, aliases, metanetx, search_client=my_search)
proteins = resolution.entries
```

An empty lookup overwrites enzyme exports with header-only tables and an empty
FASTA; reporting stages also handle an empty shortlist. A run with no resolved
proteins and failed lookups exits with failure. Unresolved reaction rows carry
`lookup_status` (`no_hits`, `no_identifier`, or `lookup_failed`) so a network
outage is distinguishable from an empty search. Missing Pfam values no longer
create a family named `nan`.
