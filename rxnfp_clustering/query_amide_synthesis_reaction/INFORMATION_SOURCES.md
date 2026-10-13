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
| **RCSB PDB** | method, resolution, bound ligands/metals; downloaded mmCIF for geometry | 179 candidates, 825 entries, 0 lookup failures |
| **InterPro** | readable family names from Pfam accessions | 1,121 rows that had only `PF…` |
| **external link table** (`uniprot-msd-links.tsv`) | precomputed UniProt→ModelSEED links | +1,131 candidates; see `MAPPING_REVIEW.md` |
| **assayed panel** (read-only, separate project) | **measured** nylon-6,6 L1 titre | 49 proteins, 3 strong producers |
| **geometric proxy** (read-only, separate project) | independent adenylation-geometry score | 36 candidates |

## Tried and unusable

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

**SABIO-RK** — the obvious credential-free alternative for kinetic data. Its
documented REST endpoint no longer serves data:

```
http://sabiork.h-its.org/sabioRestWebServices/kineticLaws?q=ECNumber:"6.3.2.17"
  -> 1182 bytes of HTML (a landing page), identical for every EC queried
```

Not a transient failure — the same response for four different ECs.

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
