# First assay round: eight proteins, two hypotheses

The two readings of this project's evidence produce disjoint lists — at most
8.3% sequence identity between them — so one is substantially wrong and
nothing here can say which. Four from each is the honest hedge.

## A. Chemistry hypothesis — four independent lines of evidence

Reached by substrate chemistry: a proven acylation of a free amine, a diacid
donor, a diamine acceptor. No measured activity on the diad.

| protein | organism | aa | why |
|---|---|---:|---|
| **Q9RDS6** glnA2 SCO2241 | Streptomyces coelicolor (strai | 453 | natively acylates **cadaverine** (C5, one methylene from HMDA); K_M 0.13 mM; pocket span 17 |
| **Q81RQ8** asbB GBAA_1982 | Bacillus anthracis | 612 | pocket span **23**; cofactor-bound structure; independent geometric score |
| **P9WJL3** murE Rv2158c MTCY270.1 | Mycobacterium tuberculosis (st | 535 | **meso-diaminopimelate bound in the crystal**; acid span 10, amine span 5 — predicted to struggle on the acceptor |
| **Q46E79** pylC Mbar_A0837 | Methanosarcina barkeri (strain | 363 | bifunctional ligand bound; cofactor structure; geometric score |

## B. Measured-data hypothesis — closest to a protein that works

The 49-member panel is the only measured activity that exists; AS9, AS17 and
AS23 produce 9.28, 1.91 and 1.45 µg/ml. These are the ANL candidates most
similar to them, restricted to comparable length (519 aa ± 25%) so that a
fragment matching one domain cannot outrank a whole enzyme.

| protein | organism | aa | % id | to |
|---|---|---:|---:|---|
| **A0A075UYU2** AJAP_12390 | Amycolatopsis japonica | 521 | 30.6 | AS9 |
| **A1YAM3** nan | Amycolatopsis orientalis (Noca | 521 | 30.6 | AS9 |
| **D2TMN3** entE ROD_06031 | Citrobacter rodentium (strain  | 535 | 30.2 | AS9 |
| **A0ABC8C1M7** B7C62_33055 | Kitasatospora albolonga | 514 | 29.8 | AS23 |

## Why both

Group A has mechanistic support and no measured support. Group B has the only
measured support and reaches just ~30% identity to it — far enough that
function is not guaranteed.

Running only A tests a hypothesis unconnected to any data. Running only B
assumes the panel's family is the answer, when the panel is 49 proteins
somebody chose rather than a survey of what works.

**Include AS17 or AS9 as a positive control.** If the assay cannot
distinguish a known producer, nothing below it is interpretable — the same
reason AS17 is staged in the cofold inputs.

Dossiers for group A: `results/CANDIDATES.md`. Full ranking for group B:
`results/anl_similarity_to_actives.tsv`.
