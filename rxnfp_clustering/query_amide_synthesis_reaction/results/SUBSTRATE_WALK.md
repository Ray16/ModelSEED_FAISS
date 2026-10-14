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

Carnosine synthase is the better lead on this axis even with fewer rungs: C3 Km 1.49-1.8 mM against C4 Km 8.8 mM is a five-fold degradation, not five orders of magnitude. No kcat is published for it, so catalytic efficiency cannot be compared and this is a Km argument only.

## linear dicarboxylate

No enzyme has measured parameters on two or more rungs of this series.

