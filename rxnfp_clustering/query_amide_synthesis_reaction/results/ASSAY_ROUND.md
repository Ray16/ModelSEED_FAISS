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
AS23 produce 9.28, 1.91 and 1.45 µg/ml. BLASTing all three against the whole
of UniProtKB — not just this project's table — gives:

| protein | organism | aa | % id | to |
|---|---|---:|---:|---|
| **A0A7Y0L7E5** | Sulfobacillus harzensis | 524 | **65.3** | AS23 |
| **G8U0Y4** | Sulfobacillus acidophilus (strain  | 524 | **65.3** | AS23 |
| **A0A3E0HG69** | Kutzneria buriramensis | 505 | **51.4** | AS9 |
| **R4SVL2** | Amycolatopsis keratiniphila | 521 | **50.3** | AS9 |

**None of these 150 hits is among this project's 54,199 candidates, and the
reason is structural.** They are annotated as long-chain-fatty-acid–CoA
ligases (53) and acyl-CoA synthetases (25) — that is, for **thioester**
chemistry. A reaction-anchored pipeline finds enzymes through the reaction
they are annotated to catalyse, so it cannot reach one whose annotation
describes the wrong reaction. The panel proteins are annotated that way
themselves, and three of them make the nylon amide bond regardless.

This table could offer at best 30.6% identity to a measured producer.
BLAST reaches **65.3%**. Full ranking: `results/blast_actives_ranked.tsv`.

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
