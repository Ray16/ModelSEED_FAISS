# The acceptor-chemistry cohort

HMDA is an **acceptor**. These are the candidates whose EC has at least one *measured* kinetic parameter for acylating a free primary amine on an aliphatic polyamine — the target half-reaction, established experimentally rather than inferred from annotation.

| EC | enzyme | parameters | acceptors measured |
|---|---|---:|---|
| 2.3.1.57 | diamine N-acetyltransferase | 10 | Spermidine,Spermine |
| 6.3.1.8 | glutathionylspermidine synthase | 1 | Spermidine |
| 6.3.1.9 | trypanothione synthase | 11 | Spermidine |

**1,751 candidates**, 1,104 of them prokaryotic and of expressible length, against **15** on the bench shortlist.

Two things a reader should hold onto. The evidence is **per EC**, so every member earns it equally and this cannot rank within a class — which is why it is a cohort file and not an evidence axis, the same objection that keeps curated ATP binding out of the convergence count. And the members are largely orthologs of one another: treat this as three chemistries with deep membership, not as 1,751 independent leads.

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
| `P39909` | 2.3.1.57 | bltD | Bacillus subtilis (strain 168) | 152 | yes | no | yes |
| `D4FZ53` | 2.3.1.57 | bltD | Bacillus subtilis subsp. natto (st | 152 | yes | no | yes |
| `P0A952` | 2.3.1.57 | speG | Escherichia coli O157:H7 | 186 | yes | no | yes |
| `P0DKR8` | 2.3.1.57 | speG | Wigglesworthia glossinidia brevipa | 174 | yes | no | yes |
| `O60993` | 6.3.1.9 | TRS | Crithidia fasciculata | 652 | yes | no | no |
| `P90518` | 6.3.1.8 | GSP | Crithidia fasciculata | 719 | yes | no | no |
| `Q9GT49` | 6.3.1.9 | TRS | Trypanosoma cruzi (strain CL Brene | 647 | yes | no | no |
| `Q6P8J2` | 2.3.1.57 | Sat2 | Mus musculus (Mouse) | 170 | yes | no | no |
| `Q7PCJ8` | 2.3.1.57 | SAT2 | Bos taurus (Bovine) | 170 | yes | no | no |
| `Q7PCJ9` | 2.3.1.57 | SAT2 | Sus scrofa (Pig) | 170 | yes | no | no |
| `Q1LUC3` | 2.3.1.57 | kat2b | Danio rerio (Zebrafish) (Brachydan | 796 | yes | no | no |
| `P49431` | 2.3.1.57 | Sat1 | Mus saxicola (Elliot's brown spiny | 171 | yes | no | no |
