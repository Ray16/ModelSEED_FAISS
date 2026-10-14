# The acceptor-chemistry cohort

HMDA is an **acceptor**. These are the candidates whose EC has at least one *measured* kinetic parameter for acylating a free primary amine on an aliphatic polyamine — the target half-reaction, established experimentally rather than inferred from annotation.

| EC | enzyme | parameters | acceptors measured |
|---|---|---:|---|
| 2.3.1.57 | diamine N-acetyltransferase | 10 | Spermidine,Spermine |
| 6.3.1.8 | glutathionylspermidine synthase | 1 | Spermidine |
| 6.3.1.9 | trypanothione synthase | 11 | Spermidine |

**1,751 candidates**, 1,104 of them prokaryotic and of expressible length, against **15** on the bench shortlist.

Two things a reader should hold onto. The **kinetic** evidence is per EC, so every member earns it equally and it alone cannot rank one above another — which is why this is a cohort file and not an evidence axis, the same objection that keeps curated ATP binding out of the convergence count. (Pocket capacity, below, is per protein and does discriminate; that is what the ordering uses.) And the members are largely orthologs of one another: treat this as three chemistries with deep membership, not as 1,751 independent leads.

**168 of these have pocket room for BOTH monomers**, and that is the only discriminator available inside a cohort defined by a per-EC property — the kinetics are identical for every member, pocket capacity is per protein. 1,458 have room for HMDA; only 168 also have room for adipate, and nylon-6,6 needs both.

The split is mechanistically self-consistent, which is why it is worth trusting. Every one of them is an **ATP-dependent ligase** (166 EC 6.3.1.8, 2 EC 6.3.1.9) and not one is a GNAT acetyltransferase — a pocket built around a C2 acetyl group has no room for a C6 diacid, so the geometry agrees with the chemistry without having been told to. They are sorted first below.

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
