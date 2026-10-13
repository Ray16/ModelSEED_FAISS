# Information sources: what was used, what was tried, what is blocked

Status as of 2026-10-13. The point of this file is that "resolve as much enzyme
information as possible" is only a meaningful claim if the sources that were
*not* used are named along with the reason — otherwise the next person repeats
the dead ends.

## In use

| source | what it gives | coverage reached |
|---|---|---|
| **UniProtKB** | sequence, organism, lineage, EC, Pfam/InterPro, catalytic activity, GO, keywords, PDB/AlphaFold xrefs, binding/active sites | sequence & organism 100%, family 99.8%, InterPro 99.7%, EC 81.8%, catalytic activity 76.5% |
| **UniParc** | exact-sequence identity for proteins absent from UniProtKB | 48/49 assayed panel members, which accession matching missed (10/48) |
| **MetaNetX** `reac_xref` | ModelSEED↔Rhea aliases, plus MetaCyc/KEGG/BiGG sibling pivots | +1,777 reactions the SEED pivot alone cannot reach |
| **RCSB PDB chemical components** | the IDENTITY of every bound ligand, classified by the substrate screen's own predicates | 147 components, 18 bifunctional, reaching 17 candidates |
| **RCSB PDB** | method, resolution, bound ligands/metals; downloaded mmCIF for geometry | 179 candidates, 825 entries, 0 lookup failures |
| **InterPro** | readable family names from Pfam accessions | 1,121 rows that had only `PF…` |
| **external link table** (`uniprot-msd-links.tsv`) | precomputed UniProt→ModelSEED links | +1,131 candidates; see `MAPPING_REVIEW.md` |
| **UniProtKB protein names** | proteins NAMED for the target function, which no reaction pivot reaches | 3,488 proteins, 3,469 of them new; fixed a measured 0-of-26 recall hole |
| **AlphaFold DB API** | model confidence (mean pLDDT + per-band fractions) for shortlisted candidates | 253 of 301; median pLDDT 94.1, none below 50 |
| **UniProt curated properties** (7 fields never requested) | curated family statement, domain architecture, subcellular location, kinetics, pH/temp optima | family statement 75.4%, domains 73.3%, location 37.3%, kinetics 0.5% (189 proteins) |
| **Rhea bulk SMILES + `rhea2ec`** | EC-class substrate ceiling: the best amide chemistry anyone in an EC class is curated to perform | 26,410 of 38,542 EC-bearing candidates; 13,063 at ≥0.6 |
| **assayed panel** (read-only, separate project) | **measured** nylon-6,6 L1 titre | 49 proteins, 3 strong producers |
| **geometric proxy** (read-only, separate project) | independent adenylation-geometry score | 36 candidates |

## Tried and unusable

**BRENDA, partially substituted.** UniProt curates kinetics itself, in a
field this pipeline never requested (`kinetics`; note that
`cc_biophysicochemical_properties`, the name the documentation suggests,
returns HTTP 400). It yields 189 proteins with a measured KM or kcat, each
with an ECO:0000269 publication, 42 of them on the shortlist, tightest KM
0.21 uM. That is real but it is 189 against the ~31,000 EC-bearing candidates
BRENDA would cover, and it is 0% for unreviewed entries, so BRENDA remains the
blocked high-value source rather than a solved one.

**BRENDA** — the one remaining source that would add genuinely new information
(Km, kcat, substrate specificity, optimal pH/temperature by EC, which 81.8% of
candidates have). The SOAP API requires a registered account:

```
BRENDA_EMAIL / BRENDA_PASSWORD
```

Neither is configured in this environment, and registering an account is tied
to a person's identity, so it is not something to do on the user's behalf.
**This is the single highest-value unblock available** — if credentials are
provided, kinetics and substrate-specificity annotation can be added for every
candidate carrying an EC number.

Four sources have now been tested for kinetics, so this is a measured
conclusion rather than an assumption: UniProt's own `kinetics` field works and
yields 194 proteins (in use, above); BRENDA needs credentials; SABIO-RK's REST
endpoint is dead; ChEMBL covers 1 of 253. There is no credential-free route to
kinetics at the scale of the ~31,000 EC-bearing candidates.

**ChEMBL** — the other credential-free bioactivity database, and the last
untested route to kinetics. Measured on the 253 shortlisted proteins carrying a
UniProt accession:

```
with a ChEMBL target record   14 of 253
with any Km activity           1        (P90518, 3 values)
```

ChEMBL indexes pharmacological bioactivity against drug targets. Our
candidates are bacterial biosynthetic enzymes, so the one protein with real
coverage is P07862 (D-alanine--D-alanine ligase), which is there because it is
a well-studied antibiotic target rather than because it is an amide synthetase.
Not usable.

**SABIO-RK** — the obvious credential-free alternative for kinetic data. Its
documented REST endpoint no longer serves data:

```
http://sabiork.h-its.org/sabioRestWebServices/kineticLaws?q=ECNumber:"6.3.2.17"
  -> 1182 bytes of HTML (a landing page), identical for every EC queried
```

Not a transient failure — the same response for four different ECs.

## The acceptor side, measured against UniProt directly

The nylon-6,6 acceptor is hexane-1,6-diamine. Asking UniProt how many enzymes
anywhere are curated to act on it:

```
cc_catalytic_activity:"hexane-1,6-diamine"   0 entries
cc_catalytic_activity:"1,6-diaminohexane"    0 entries
```

**Zero, in the whole knowledgebase.** That is the project's premise stated at
the source rather than inferred from our own table, and it is why no amount of
retrieval will find an enzyme that already does this reaction.

So the question becomes the nearest native acceptor, and the answer is sharp:

| acceptor | UniProt entries | of those, amide-FORMING (EC 2.3 / 6.3) |
|---|---:|---:|
| hexane-1,6-diamine (C6) | 0 | 0 |
| **cadaverine (C5)** | 1,139 | **3** |
| putrescine (C4) | 22,073 | — |
| spermidine | 24,607 | — |

All **three** of the amide-forming cadaverine enzymes in UniProt are already
in this table, and two are in the converged shortlist:

```
Q9RDS6  GlnA2  gamma-glutamylpolyamine synthetase   EC 6.3.1.-   converged, 4 lines
Q9KZC7  GlnA3  gamma-glutamylpolyamine synthetase   EC 6.3.1.-   converged
O41136  vPAT   viral polyamine acetyltransferase    EC 2.3.1.57
```

Cadaverine is one methylene shorter than HMDA. On the acceptor side, this
pipeline has found everything UniProt knows.

## Tried and rejected on precision

**Substrate-name search of the catalytic-activity field.** The largest
unresolved bucket is 121 reactions with no Rhea/EC whose *product* names
returned nothing. The name route reads only the product side by design, so
searching for the **reactant** names instead looked like the obvious remaining
route — a synthetase does name its substrate in its catalytic-activity
statement.

Tested on the 10 highest-scoring of those 121. All 10 "resolved" — and that is
the problem:

```
rxn21307  "Succinyl-CoA" -> 100 hits  Succinate--CoA ligase subunit beta
rxn40659  "Acetyl-CoA"   -> 100 hits  Acyl-CoA synthetase short-chain family
rxn21149  "L-Alanine"    -> 100 hits  Nonribosomal peptide synthetase echPS
rxn59943  "L-Glutamate"  -> 100 hits  Glutamate--methylamine ligase
rxn23369  "L-Lysine"     -> 100 hits  Hydroxylysine--glutamate ligase
```

Every query hit the 100-result page cap, because every substrate named is a
ubiquitous metabolite. The hits are every ligase that mentions acetyl-CoA or
L-glutamate anywhere — not the enzyme for that reaction.

This is the same failure mode as the GO route criticised in
`MAPPING_REVIEW.md`: chemically true, biologically meaningless.

**A sharper reason emerged later, and it is the real one.** The page-cap
argument above only explains why COMMON metabolites fail. Re-tested on
`cadaverine`, a *specific* metabolite returning a tractable 1,139 entries:
only **3** of those carry an amide-forming EC. The other 1,136 are
transaminases (EC 2.6.1), amine oxidases (EC 1.4) and decarboxylases
(EC 4.1) - enzymes that consume or transfer the amine rather than acylate it.
Naming a metabolite says nothing about the DIRECTION of the chemistry, which
is why this route fails even where the query is precise. A distinctive
**product** name (putrebactin, bisucaberin, alcaligin) identifies one enzyme;
a common **substrate** name identifies thousands. Building it would have added
roughly twelve thousand spurious associations and inflated the candidate count
while lowering its value.

Not built. This is why the 121 stay unresolved: not for want of a query, but
because no *specific* query exists for them.

**Protein-name queries that are superfamily labels or the wrong chemistry.**
The protein-name route (now in use, above) is curated, and these were measured
and left out rather than forgotten. Their totals are given to show that size
was *not* the deciding test — `cyanophycin synthetase` has more members than
`aminoacyltransferase` and is admissible on chemistry while the latter is not:

| name | entries | why not |
|---|---:|---|
| `glutamine synthetase` | 64,888 | forms a **terminal** amide from ammonia; the substrate screen scores that 0.0 because it cannot polycondense |
| `N-acyltransferase` | 59,383 | a superfamily label spanning every acyl donor, not a function |
| `peptide synthetase` | 18,124 | NRPS; the amide forms on a carrier-bound thioester, not between two free monomers |
| `cyanophycin synthetase` | 3,819 | on-chemistry, and the closest biological nylon analogue — but CphA is already captured by `activity_ec` and `family_homolog`, so this would add ~3,800 near-identical orthologs and not one new family |
| `aminoacyltransferase` | 930 | aminoacyl-tRNA and ribosomal machinery; the acceptor is not a free amine |

A name matching more than `ENZYME_NAME_CEILING` (10,000) entries is refused
loudly instead of ingested, so a superfamily label added here later fails
visibly rather than quietly inflating the count.

**Rhea, searched directly instead of pivoted into.** Every reaction route in
this pipeline starts from a ModelSEED reaction, so it can only ever reach
chemistry ModelSEED contains. Rhea does not have that limit, so the obvious
remaining move was to screen Rhea itself: download `rhea-reaction-smiles.tsv`
(36,552 reactions), run the project's OWN validated screen over all of it, and
query UniProt by `rhea:ID` for every reaction that proves amide formation on a
free amine. No new judgement, no new scoring - the existing screen decides.

Measured 2026-10-13:

| | |
|---|---:|
| Rhea reactions with SMILES | 36,552 |
| prove amide formation on a free amine | 532 |
| of those, NOT reachable from ModelSEED | **417** |
| novel and scoring >= 0.8 | 18 |
| best novel score (RHEA:21366) | **0.9191**, above anything we hold |

That looks like a large win and is not one. Querying UniProt for the 25
highest-scoring novel reactions returns **3 new proteins**, because 21 of the
25 have no UniProt protein annotated to them at all - the same "undeposited
enzymology" category as our own 21 unresolved reactions. The 3 that did return
are histone acetyltransferase KAT2A, a PROTEIN acetyltransferase that scores
0.88 only because Rhea writes the substrate lysine as a free small molecule, so
the macromolecular guard never fires.

**The data was right, the application was wrong.** Rhea is poor at naming
enzymes and excellent at describing reactions, so it is useless for retrieval
and valuable for ANNOTATION. The same scored corpus now supplies an
*EC-class substrate ceiling* — see `amide_synthesis/ec_substrate.py` — reaching
26,410 candidates that previously had only an inherited family distance, and
giving the protein-name route its first substrate-level number at all. The
retrieval verdict below stands; the dismissal of the dataset did not.

So the conclusion is the opposite of the premise: **ModelSEED is not the
bottleneck.** Rhea curates far more amide chemistry than ModelSEED does, and
almost none of it has an enzyme attached. Not adopted - it would add one real
protein and one false positive. This is the cleanest evidence yet that the
remaining 174 unresolved reactions are source-limited rather than
retrieval-limited.

## Deliberately not used

**KEGG REST** (free, works — tested `get/ec:6.3.2.17` and `link/ko/ec:…`). It
returns EC synonyms, reaction class and KO orthology identifiers. Every one of
those duplicates information already present: protein names from UniProt, and
family grouping at 99.8% from Pfam/InterPro. Adding it would raise the column
count without raising the information content, so it was left out. The one
non-duplicative thing it offers is KO-based orthology grouping, which could in
principle refine the shortlist's diversity caps (currently family / UniRef50
clade / genus) — a modest, speculative gain, recorded here rather than built.

**NetSolP / SoluProt / ESM-based solubility and expression predictors** — used
by the upstream project. These need GPU, which is blocked (below).

## Blocked on infrastructure

**Cofolding (Boltz-2 / AF3 / ESMFold2)** — all three kits run only on lambda13,
which has a driver/library mismatch: kernel module 565.57.01 against userspace
NVML 580.178.04, so `nvidia-smi` fails and `gpu_reserve` cannot enumerate the
node's GPUs. `gpu_reserve` is the mandatory gate on this cluster and was not
bypassed. Inputs are staged; see `results/cofold_validation/BLOCKED.md` for the
diagnosis, the reproduction and the one-line `pipefail` bug in `fold.sh` that
makes the failure silent. Needs root to fix.

## Blocked on data, not effort

**A validated candidate ranking.** Measured rather than asserted — see
`tools/validate_against_activity.py`. With 3 measured actives the bootstrap CI
on AUROC spans 0.23–0.85, so the panel cannot validate *any* score. More
measured actives are the only route; the 292-protein shortlist is built with
family/clade/genus diversity caps precisely so that a next assay round covers
as many distinct families as bench slots allow.
