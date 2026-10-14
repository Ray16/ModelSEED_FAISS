# The acceptor-chemistry cohort

HMDA is an **acceptor**. These are the candidates whose EC has at least one *measured* kinetic parameter for acylating a free primary amine on an aliphatic polyamine — the target half-reaction, established experimentally rather than inferred from annotation.

| EC | enzyme | parameters | acceptors measured |
|---|---|---:|---|
| 2.3.1.57 | diamine N-acetyltransferase | 10 | Spermidine,Spermine |
| 6.3.1.8 | glutathionylspermidine synthase | 1 | Spermidine |
| 6.3.1.9 | trypanothione synthase | 11 | Spermidine |

**309 candidates**, 205 of them prokaryotic and of expressible length, against **15** on the bench shortlist.

Two things a reader should hold onto. The evidence is **per EC**, so every member earns it equally and this cannot rank within a class — which is why it is a cohort file and not an evidence axis, the same objection that keeps curated ATP binding out of the convergence count. And the members are largely orthologs of one another: treat this as three chemistries with deep membership, not as 309 independent leads.

Ki values are excluded when selecting the ECs. A Ki says a molecule binds, which for a bisubstrate analogue says nothing about turnover — see the EC 2.3.1.87 near-miss in `SUBSTRATE_WALK.md`.

Full ranking in `ACCEPTOR_COHORT.tsv`; the first 25 follow.

| protein | EC | gene | organism | aa | reviewed | PDB | prok |
|---|---|---|---|---:|---|---|---|
| `P0AES0` | 6.3.1.8 | gss | Escherichia coli (strain K12) | 619 | yes | yes | yes |
| `P0A951` | 2.3.1.57 | speG | Escherichia coli (strain K12) | 186 | yes | yes | yes |
| `Q9KL03` | 2.3.1.57 | speG | Vibrio cholerae serotype O1 (strai | 173 | yes | yes | yes |
| `P21340` | 2.3.1.57 | paiA | Bacillus subtilis (strain 168) | 172 | yes | yes | yes |
| `Q9HL57` | 2.3.1.57 | paiA | Thermoplasma acidophilum (strain A | 161 | yes | yes | yes |
| `Q96F10` | 2.3.1.57 | SAT2 | Homo sapiens (Human) | 170 | yes | yes | no |
| `Q92831` | 2.3.1.57 | KAT2B | Homo sapiens (Human) | 832 | yes | yes | no |
| `Q9JHD1` | 2.3.1.57 | Kat2b | Mus musculus (Mouse) | 813 | yes | yes | no |
| `P21673` | 2.3.1.57 | SAT1 | Homo sapiens (Human) | 171 | yes | yes | no |
| `P48026` | 2.3.1.57 | Sat1 | Mus musculus (Mouse) | 171 | yes | yes | no |
| `Q5CPU3` | 2.3.1.57 | SSAT | Cryptosporidium parvum (strain Iow | 152 | yes | yes | no |
| `O41136` | 2.3.1.57 | A654L | Paramecium bursaria Chlorella viru | 197 | yes | yes | no |
| `P0AES1` | 6.3.1.8 | gss | Shigella flexneri | 619 | yes | no | yes |
| `O60993` | 6.3.1.9 | TRS | Crithidia fasciculata | 652 | yes | no | no |
| `P90518` | 6.3.1.8 | GSP | Crithidia fasciculata | 719 | yes | no | no |
| `Q9GT49` | 6.3.1.9 | TRS | Trypanosoma cruzi (strain CL Brene | 647 | yes | no | no |
| `Q1LUC3` | 2.3.1.57 | kat2b | Danio rerio (Zebrafish) (Brachydan | 796 | yes | no | no |
| `A0A0M3KKU5` | 2.3.1.57 |  | Escherichia coli LY180 | 170 | no | yes | yes |
| `A0A437SL45` | 2.3.1.57 | BM74_11955 | Bacillus thuringiensis | 171 | no | yes | yes |
| `A0A5P8YIA2` | 2.3.1.57 | speG | Yersinia pestis | 181 | no | yes | yes |
| `Q83B40` | 2.3.1.57 | speG | Coxiella burnetii (strain RSA 493  | 170 | no | yes | yes |
| `A0A084ZP18` | 6.3.1.8 | gsp | Trabulsiella guamensis ATCC 49490 | 623 | no | no | yes |
| `A0A085JCS5` | 6.3.1.8 | gsp | Tatumella ptyseos ATCC 33301 | 626 | no | no | yes |
| `A0A089Q039` | 6.3.1.8 | JT31_15700 | Cedecea neteri | 620 | no | no | yes |
| `A0A7D4ABJ6` | 6.3.1.9 | ACTIVE_7860 | Actinomadura verrucosospora | 385 | no | no | yes |
