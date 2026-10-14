# The acceptor-chemistry cohort

HMDA is an **acceptor**. These are the candidates whose EC has at least one *measured* kinetic parameter for acylating a free primary amine on an aliphatic polyamine — the target half-reaction, established experimentally rather than inferred from annotation.

| EC | enzyme | parameters | acceptors measured |
|---|---|---:|---|
| 2.3.1.57 | diamine N-acetyltransferase | 10 | Spermidine,Spermine |
| 6.3.1.8 | glutathionylspermidine synthase | 1 | Spermidine |
| 6.3.1.9 | trypanothione synthase | 11 | Spermidine |

**1,751 candidates**, 1,104 of them prokaryotic and of expressible length, against **15** on the bench shortlist.

Two things a reader should hold onto. The **kinetic** evidence is per EC, so every member earns it equally and it alone cannot rank one above another — which is why this is a cohort file and not an evidence axis, the same objection that keeps curated ATP binding out of the convergence count. Pocket capacity, below, cuts the cohort hard and is what the ordering uses, but it turns out to be family-inherited rather than per-protein, so it mostly separates ECs too. And the members are largely orthologs of one another: treat this as three chemistries with deep membership, not as 1,751 independent leads.

**168 of these have pocket room for BOTH monomers.** 1,458 have room for HMDA; only 168 also have room for adipate, and nylon-6,6 needs both. This is the sharpest cut available on the cohort, and the two paragraphs after it say exactly how much weight it will bear — less than it first appears.

Every one of them is an **ATP-dependent ligase** (166 EC 6.3.1.8, 2 EC 6.3.1.9) and not one is a GNAT acetyltransferase. Read that carefully, because it is weaker than it looks: the GNATs have real amine-side spans (3, 5, 8, 12) but `acid_span` is **0 for all 1,542**, and absence of a measurement is not absence of a pocket. It is consistent with a CoA-dependent acetyltransferase having no free-acid subsite — its donor arrives as a thioester — but no acid-bearing ligand may simply have been crystallised in that family. The conclusion rests on the mechanism; the geometry does not independently confirm it.

Nor is this a per-protein signal, though it is computed per protein. All 166 passing members of EC 6.3.1.8 carry identical spans (amine 19, acid 9) because the span is inherited from a ligand bound in the family's structures. It separates families, hence mostly ECs — 0% of 2.3.1.57, 87% of 6.3.1.8, 11% of 6.3.1.9 — and the 24 that fail inside 6.3.1.8 have spans of zero, which is missing data rather than a small pocket. They are sorted first below regardless, since a family-level reason is still a reason.

Ki values are excluded when selecting the ECs. A Ki says a molecule binds, which for a bisubstrate analogue says nothing about turnover — see the EC 2.3.1.87 near-miss in `SUBSTRATE_WALK.md`.

Full ranking in `ACCEPTOR_COHORT.tsv`; the first 25 follow.

| protein | EC | gene | organism | aa | both monomers | reviewed | PDB | prok |
|---|---|---|---|---:|---|---|---|---|
| `P0AES0` | 6.3.1.8 | gss | Escherichia coli (strain K12) | 619 | **yes** | yes | yes | yes |
| `P0AES1` | 6.3.1.8 | gss | Shigella flexneri | 619 | **yes** | yes | no | yes |
| `O60993` | 6.3.1.9 | TRS | Crithidia fasciculata | 652 | **yes** | yes | no | no |
| `P90518` | 6.3.1.8 | GSP | Crithidia fasciculata | 719 | **yes** | yes | no | no |
| `Q9GT49` | 6.3.1.9 | TRS | Trypanosoma cruzi (strain CL Brene | 647 | **yes** | yes | no | no |
| `A0A084ZP18` | 6.3.1.8 | gsp | Trabulsiella guamensis ATCC 49490 | 623 | **yes** | no | no | yes |
| `A0A085JCS5` | 6.3.1.8 | gsp | Tatumella ptyseos ATCC 33301 | 626 | **yes** | no | no | yes |
| `A0A089Q039` | 6.3.1.8 | JT31_15700 | Cedecea neteri | 620 | **yes** | no | no | yes |
| `A0A095TAM0` | 6.3.1.8 | HA49_11005 | Tatumella morbirosei | 625 | **yes** | no | no | yes |
| `A0A097R6J7` | 6.3.1.8 | AT03_19415 | Hafnia alvei FB1 | 625 | **yes** | no | no | yes |
| `A0A0D7LGP7` | 6.3.1.8 | gss | Citrobacter freundii | 619 | **yes** | no | no | yes |
| `A0A0F6B6Q8` | 6.3.1.8 | gsp | Salmonella typhimurium (strain 140 | 618 | **yes** | no | no | yes |
| `A0A0G4JQI7` | 6.3.1.8 | BN1221_00545c | Brenneria goodwinii | 623 | **yes** | no | no | yes |
| `A0A0H2VCD6` | 6.3.1.8 | gsp | Escherichia coli O6:H1 (strain CFT | 620 | **yes** | no | no | yes |
| `A0A0H2Z2U2` | 6.3.1.8 | gsp | Escherichia coli O1:K1 / APEC | 620 | **yes** | no | no | yes |
| `A0A0H3CTC1` | 6.3.1.8 | ECL_04317 | Enterobacter cloacae subsp. cloaca | 620 | **yes** | no | no | yes |
| `A0A0H3ENL6` | 6.3.1.8 | NRG857_14830 | Escherichia coli O83:H1 (strain NR | 619 | **yes** | no | no | yes |
| `A0A0H3FLD5` | 6.3.1.8 | EAE_03450 | Klebsiella aerogenes (strain ATCC  | 621 | **yes** | no | no | yes |
| `A0A0H3NFY5` | 6.3.1.8 | gsp | Salmonella typhimurium (strain SL1 | 618 | **yes** | no | no | yes |
| `A0A0J5L0Q4` | 6.3.1.8 | ABW06_19105 | Pluralibacter gergoviae (Enterobac | 619 | **yes** | no | no | yes |
| `A0A0L0GR58` | 6.3.1.8 | GM31_01900 | Trabulsiella odontotermitis | 623 | **yes** | no | no | yes |
| `A0A0N0Z8P6` | 6.3.1.8 | M992_1310 | Moellerella wisconsensis ATCC 3501 | 628 | **yes** | no | no | yes |
| `A0A0X4EHT1` | 6.3.1.8 | AWI28_21670 | Enterobacter genomosp. O | 620 | **yes** | no | no | yes |
| `A0A1B7K6N2` | 6.3.1.8 | M989_00827 | Kluyvera georgiana ATCC 51603 | 621 | **yes** | no | no | yes |
| `A0A1B7L8G1` | 6.3.1.8 | A9B99_02825 | Mangrovibacter phragmitis | 625 | **yes** | no | no | yes |
