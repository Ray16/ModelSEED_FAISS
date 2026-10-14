# Measured substrate walks, not substrate resemblance

Every chemistry score here asks how much an enzyme's native substrate resembles adipate + HMDA. This asks a different question: which enzymes have **published kinetics on more than one rung of a chain-length series**, and how far up it they have been taken. An enzyme whose native substrate is distant but whose activity has been walked toward C6 is what an engineering campaign wants, and no axis in this project can see it.

## omega-amino acid donor

| EC | enzyme | rungs measured | longest | reaches C6 |
|---|---|---|---:|---|
| 6.3.2.1 | pantoate---beta-alanine ligase | C2,C3,C4,C5 | C5 | **no** |
| 6.3.2.11 | carnosine synthase | C3,C4 | C4 | **no** |
| 2.3.1.65 | bile acid-CoA:amino acid N-acyltransferase | C2,C3 | C3 | **no** |

Measured values for the longest walk, EC 6.3.2.1:

| rung | substrate | parameter | value | unit |
|---|---|---|---:|---|
| C5 | 5-Aminovaleric acid | kcat/Km | 0.42 | M^(-1)*s^(-1) |
| C5 | 5-Aminovaleric acid | Km | 72.0 | mM |
| C2 | Glycine | kcat/Km | 0.6 | M^(-1)*s^(-1) |
| C2 | Glycine | Km | 84.0 | mM |
| C3 | beta-Alanine | Km | 0.8 | mM |
| C3 | beta-Alanine | kcat/Km | 4300.0 | mM^(-1)*s^(-1) |
| C3 | beta-Alanine | kcat/Km | 4.3 | mM^(-1)*s^(-1) |
| C3 | beta-Alanine | Km | 0.00015 | M |
| C4 | gamma-Aminobutyric acid | kcat/Km | 0.81 | M^(-1)*s^(-1) |
| C4 | gamma-Aminobutyric acid | Km | 335.0 | mM |

**Read the values, not the rungs - they say the opposite of what the rung count suggests.** The tempting story is that activity declines with chain length and flattens by C5, so C6 is worth measuring. The C2 row kills it: glycine is 0.6 M^-1 s^-1, against C4 at 0.81 and C5 at 0.42. Everything that is not beta-alanine sits at the same ~0.5 M^-1 s^-1 floor whether it is SHORTER or longer, so the flat C4-C5 segment is non-specific background, not tolerance extending along the chain. PanC is sharply specific for C3; it is a four-rung walk that demonstrates specificity, not flexibility, and it is not evidence that C6 would work.

Carnosine synthase is the better lead ON THIS AXIS even with fewer rungs: C3 Km 1.49-1.8 mM against C4 Km 8.8 mM is a five-fold degradation, not five orders of magnitude. No kcat is published for it, so catalytic efficiency cannot be compared and this is a Km argument only.

It is worse on every other axis a bench cares about, and the trade-off should be read whole. **All 326 carnosine synthases held here are eukaryotic** - there is not one bacterial or archaeal member - and they run 827-951 residues. PanC is 283-309 residues, bacterial, reviewed, and has solved structures. So the enzyme with the better substrate tolerance is the harder expression target by a wide margin, and the one that is trivial to express is the one whose walk shows specificity rather than flexibility.

## linear dicarboxylate

No enzyme has measured parameters on two or more rungs of this series.

## free diamine acceptor

No enzyme has measured parameters on two or more rungs of this series.

## polyamine acceptor

| EC | enzyme | rungs measured | longest | reaches C6 |
|---|---|---|---:|---|
| 2.3.1.57 | diamine N-acetyltransferase | C7,C10 | C10 | yes |

Measured values for the longest walk, EC 2.3.1.57:

| rung | substrate | parameter | value | unit |
|---|---|---|---:|---|
| C7 | Spermidine | Km | 52.0 | µM |
| C7 | Spermidine | Km | 130.0 | µM |
| C7 | Spermidine | kcat/Km | 24000.0 | M^(-1)*s^(-1) |
| C7 | Spermidine | Km | 50.0 | µM |
| C10 | Spermine | Km | 33.0 | µM |
| C10 | Spermine | Km | 35.0 | µM |
| C10 | Spermine | kcat/Km | 820000.0 | M^(-1)*s^(-1) |
| C10 | Spermine | Km | 11.0 | µM |

**Read the values, not the rungs - they say the opposite of what the rung count suggests.** The tempting story is that activity declines with chain length and flattens by C5, so C6 is worth measuring. The C2 row kills it: glycine is 0.6 M^-1 s^-1, against C4 at 0.81 and C5 at 0.42. Everything that is not beta-alanine sits at the same ~0.5 M^-1 s^-1 floor whether it is SHORTER or longer, so the flat C4-C5 segment is non-specific background, not tolerance extending along the chain. PanC is sharply specific for C3; it is a four-rung walk that demonstrates specificity, not flexibility, and it is not evidence that C6 would work.

Carnosine synthase is the better lead ON THIS AXIS even with fewer rungs: C3 Km 1.49-1.8 mM against C4 Km 8.8 mM is a five-fold degradation, not five orders of magnitude. No kcat is published for it, so catalytic efficiency cannot be compared and this is a Km argument only.

It is worse on every other axis a bench cares about, and the trade-off should be read whole. **All 326 carnosine synthases held here are eukaryotic** - there is not one bacterial or archaeal member - and they run 827-951 residues. PanC is 283-309 residues, bacterial, reviewed, and has solved structures. So the enzyme with the better substrate tolerance is the harder expression target by a wide margin, and the one that is trivial to express is the one whose walk shows specificity rather than flexibility.

## Candidates in these ECs

All three walk-ECs are near-invisible in the existing deliverables: 1,720 candidates between them, 8 on the bench shortlist, **0 in QUALIFIED.md** - because every evidence axis scores substrate RESEMBLANCE, which is exactly what these proteins fail (carnosine synthase 0.667 and PanC 0.425, against a 0.75 floor).

Listing the best 6 per EC rather than the class: EC 6.3.2.1 alone holds 1,390 candidates and admitting them wholesale is the mass-expansion refused everywhere else here. Ranked on what a bench needs - reviewed, then a solved structure, then a model, then substrate distance - not on `master_priority`, which is tuned for the resemblance axis these proteins fail.

### EC 6.3.2.1

| protein | gene | organism | len | reviewed | PDB | prokaryote |
|---|---|---|---:|---|---|---|
| `P31663` | panC | Escherichia coli (strain K12) | 283 | yes | yes | yes |
| `P9WIL4` | panC | Mycobacterium tuberculosis (strain C | 309 | yes | yes | yes |
| `P9WIL5` | panC | Mycobacterium tuberculosis (strain A | 309 | yes | yes | yes |
| `Q2FV22` | panC | Staphylococcus aureus (strain NCTC 8 | 283 | yes | yes | yes |
| `Q6G456` | panC | Bartonella henselae (strain ATCC 498 | 284 | yes | yes | yes |
| `Q6GDK5` | panC | Staphylococcus aureus (strain MRSA25 | 283 | yes | yes | yes |

### EC 6.3.2.11

| protein | gene | organism | len | reviewed | PDB | prokaryote |
|---|---|---|---:|---|---|---|
| `A5YM72` | CARNS1 | Homo sapiens (Human) | 827 | yes | no | **no** |
| `D3KCC4` | CARNS1 | Gallus gallus (Chicken) | 930 | yes | no | **no** |
| `Q6ZPS2` | Carns1 | Mus musculus (Mouse) | 827 | yes | no | **no** |
| `A0A287BLD0` | CARNS1 | Sus scrofa (Pig) | 921 | no | no | **no** |
| `A0A5G2QZP0` | CARNS1 | Sus scrofa (Pig) | 925 | no | no | **no** |
| `F1RUY9` | CARNS1 | Sus scrofa (Pig) | 951 | no | no | **no** |

### EC 2.3.1.65

| protein | gene | organism | len | reviewed | PDB | prokaryote |
|---|---|---|---:|---|---|---|
| `Q14032` | BAAT | Homo sapiens (Human) | 418 | yes | no | **no** |
| `Q63276` | Baat | Rattus norvegicus (Rat) | 420 | yes | no | **no** |
| `Q91X34` | Baat | Mus musculus (Mouse) | 420 | yes | no | **no** |
| `C5BVE1` | Bcav_2277 | Beutenbergia cavernae (strain ATCC B | 425 | no | no | yes |

### EC 2.3.1.57

| protein | gene | organism | len | reviewed | PDB | prokaryote |
|---|---|---|---:|---|---|---|
| `Q96F10` | SAT2 | Homo sapiens (Human) | 170 | yes | yes | **no** |
| `P21340` | paiA | Bacillus subtilis (strain 168) | 172 | yes | yes | yes |
| `Q9HL57` | paiA | Thermoplasma acidophilum (strain ATC | 161 | yes | yes | yes |
| `Q5CPU3` | SSAT | Cryptosporidium parvum (strain Iowa  | 152 | yes | yes | **no** |
| `Q92831` | KAT2B | Homo sapiens (Human) | 832 | yes | yes | **no** |
| `Q9JHD1` | Kat2b | Mus musculus (Mouse) | 813 | yes | yes | **no** |

## The acceptor side, and one near-miss worth not repeating

HMDA is an **acceptor**, so the donor series above is the less relevant half. On the acceptor side: **no enzyme anywhere in the 11,445 parameters has measured kinetics on two rungs of the free diamine series** - not ethylenediamine, 1,3-diaminopropane, putrescine, cadaverine or HMDA. That is the fifth independent source to reproduce this project's central finding, and the first to do it with measured kinetics rather than annotation.

The one acceptor-side walk is **EC 2.3.1.57, diamine N-acetyltransferase (SSAT)**, measured on spermidine and spermine. It is the best acceptor lead here and better than carnosine synthase on every practical axis: Km on spermidine 50-130 uM (a substrate Km, not an inhibition constant), kcat/Km 3.75e5 M^-1 s^-1, and of 1,567 candidates held **913 are prokaryotic**, against 0 of 326 for carnosine synthase. The representatives below include `P21340` (B. subtilis paiA, 172 aa) and `Q9HL57` (T. acidophilum paiA, 161 aa), both reviewed with solved structures. It scores 0.6367 on substrate distance, below the 0.75 floor, so no axis in this project sees it; 2 of the 1,567 are shortlisted.

Two caveats that stop this being a recommendation. It is **CoA-dependent acyl transfer, not ATP-dependent ligation** - a route to the nylon bond via adipoyl-CoA rather than free adipate, which is a different process decision. And its native acceptors are longer branched polyamines, not a simple C6 diamine; nothing has been measured on putrescine or HMDA, so acceptance of a short unbranched diamine is untested.

**The near-miss, recorded so the next reader does not spend the same hour on it.** EC 2.3.1.87 (aralkylamine N-acetyltransferase) appears in the data with an acyl series running acetyl, propanoyl, pentanoyl and **hexanoyl** - C2 to C6, apparently a chain walk reaching exactly adipoyl length. It is not one. Every value is a **Ki**, and the species are `CoA-S-acyl tryptamine` bisubstrate analogues: transition-state mimics built by medicinal chemists to inhibit the enzyme, not substrates it turns over. What the series does show is that the acyl channel ACCOMMODATES a C6 chain (Ki 6.5 uM at hexanoyl against 0.048 uM at acetyl, ~135x weaker binding) - a pocket-capacity fact, not a catalytic one, and on an aromatic monoamine acceptor rather than a diamine.

