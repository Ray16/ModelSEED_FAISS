# First assay round: ten proteins, three hypotheses

Three readings of this project's evidence produce disjoint lists — A and B
share at most 8.3% sequence identity, and C is a different protein fold and
a third the size of either — so at least two are substantially wrong and
nothing here can say which. Spreading the round across all three is the
honest hedge.

| | basis | fold | size |
|---|---|---|---|
| **A** | substrate chemistry, from annotation | PF06276 / PF01225 / PF02655 / PF00120 | 363–612 aa |
| **B** | sequence identity to a measured producer | PF00501;PF13193 (ANL) | 505–524 aa |
| **C** | measured kinetics on the acceptor chemistry | PF00583 (GNAT) | 161–172 aa |

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

**Not one of these 150 hits was reachable by any annotation-derived route —
all 54,199 candidates as the set then stood — and the reason is structural.**
(They are candidates *now*: the `active_homolog` route was built precisely to
admit them, and brought in 822 at ≥40% identity. The disjointness below is
therefore a statement about what annotation can reach, not about the current
candidate set.) They are annotated as long-chain-fatty-acid–CoA
ligases (53) and acyl-CoA synthetases (25) — that is, for **thioester**
chemistry. A reaction-anchored pipeline finds enzymes through the reaction
they are annotated to catalyse, so it cannot reach one whose annotation
describes the wrong reaction. The panel proteins are annotated that way
themselves, and three of them make the nylon amide bond regardless.

This table could offer at best 30.6% identity to a measured producer.
BLAST reaches **65.3%**. Full ranking: `results/blast_actives_ranked.tsv`.

## C. Measured acceptor chemistry — the half HMDA actually sits on

Groups A and B both reason about the **donor**. HMDA is an **acceptor**, and
SABIO-RK's 11,445 measured parameters answer the acceptor question directly:
which enzymes have published kinetics acylating a free primary amine on an
aliphatic polyamine. Exactly one EC does — **2.3.1.57, diamine
N-acetyltransferase** — with K_M on spermidine of 50–130 µM, a substrate
K_M rather than an inhibition constant, and k_cat/K_M 3.75×10⁵ M⁻¹s⁻¹.

| protein | organism | aa | why |
|---|---|---:|---|
| **P21340** paiA BSU32150 | Bacillus subtilis (strain 168) | 172 | EC 2.3.1.57; reviewed; structure 1TIQ; bacterial, trivially expressible |
| **Q9HL57** paiA Ta0374 | Thermoplasma acidophilum | 161 | EC 2.3.1.57; reviewed; **four** structures (3F0A/3FIX/3K9U/3NE7); thermophilic archaeon |

Both arrived through this project's own reaction screen and then scored
**0.555** on substrate distance — the best anything in this EC reaches is
0.6367, and the floor is 0.75 — so no evidence axis
here surfaces them, and 2 of the EC's 1,567 candidates are shortlisted. They
are in this round because a *measured* K_M on polyamine acylation is
evidence of a different kind from anything in A or B, not because the
ranking liked them.

**Two caveats, and they are the reason this is a third of the round rather
than the whole of it.** This is CoA-dependent acyl transfer, not
ATP-dependent ligation: it implies a route through adipoyl-CoA rather than
free adipate, which is a different process decision upstream. And the
measured acceptors are longer branched polyamines — spermidine and spermine
— with nothing measured on putrescine or HMDA, so acceptance of a short
unbranched C6 diamine is untested. What is established is the mechanism and
the pocket, not the substrate.

Also note what is *not* here: **no enzyme anywhere in the 11,445 parameters
has measured kinetics on two rungs of the free-diamine series** —
ethylenediamine, 1,3-diaminopropane, putrescine, cadaverine, HMDA. That is
the fifth independent source to reproduce this project's central finding,
and the first to do it with measured kinetics rather than annotation. See
`SUBSTRATE_WALK.md`, which also records a convincing near-miss (EC 2.3.1.87,
an apparent C2→C6 acyl walk that is entirely bisubstrate-analogue Ki values).

## The two annotation/homology groups are disjoint on *both* axes

The BLAST hits were put through the same machinery as every other candidate
here. Not one has substrate-level evidence — and this was re-tested at 13×
the sample size precisely because a null result on 150 could be chance:

```
deep search, all 3 producers vs UniProtKB  2,952 distinct hits
  >= 45% identity to a measured producer     259
  >= 40%                                     822
  already among the 54,199 annotation-derived   0
  of the 259, with an amide-forming
  curated reaction or EC ceiling               0
```

All 54 that carry an EC are **EC 6.2** — acid–thiol ligase, which forms a
**thioester with CoA**. The amide screen scores those zero by construction,
and correctly: forming a thioester is not forming an amide.

So:

| | substrate evidence | identity to a measured producer |
|---|---|---|
| **group A** | 4 independent lines each | ~8% |
| **group B** | **zero** | **50–65%** |

Neither group has both, and that is the whole problem in one table. The
proteins that demonstrably make this bond are annotated for *different
chemistry*, so annotation-based retrieval — the basis of this entire
pipeline — cannot reach them, and the chemistry screen correctly refuses them
when it does.

**That is not a bug to fix. It is the reason the assay round decides
something no further computation can.**

## Why all three

Group A has mechanistic support and no measured support. Group B has the only
measured support on the diad and reaches just ~30% identity to it — far
enough that function is not guaranteed. Group C has measured support for the
*acceptor half-reaction* and none on the diad, from a different fold.

Running only A tests a hypothesis unconnected to any data. Running only B
assumes the panel's family is the answer, when the panel is 49 proteins
somebody chose rather than a survey of what works. Running only C tests a
pocket and a mechanism on a substrate nobody has measured. Each failure mode
is different, which is the whole argument for spending the round across
them.

C is also the cheapest to run: 161 and 172 residues against 363–612 for A
and ~520 for B, both reviewed, both with solved structures, one of them
thermophilic. If a round has to be cut, cut it last rather than first.

**Include AS17 or AS9 as a positive control.** If the assay cannot
distinguish a known producer, nothing below it is interpretable — the same
reason AS17 is staged in the cofold inputs.

Dossiers for group A: `results/CANDIDATES.md`. Full ranking for group B:
`results/anl_similarity_to_actives.tsv`. Evidence and caveats for group C:
`results/SUBSTRATE_WALK.md`; the underlying kinetics are in
`results/sabio_kinetics.tsv`.
