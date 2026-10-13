# Candidates worth a bench slot

20 proteins, each supported by at least 3 independent observations that are not merely family membership. Ordered by how many lines agree, then by substrate distance.

**None of these has been assayed on nylon-6,6.** Convergence means several methods that fail differently happen to agree, which is a reason to spend a bench slot and not evidence that the enzyme works. With three measured positives in the reference panel, no ranking here can be validated.

---

## Q9RDS6 — Gamma-glutamylpolyamine synthetase GlnA2 (EC 6.3.1.-)

**Organism** Streptomyces coelicolor (strain ATCC BAA-471 / A3(2) / M145)  
**Gene** `glnA2 SCO2241`  
**Length** 453.0 aa  

**Why it is here** — 4 independent lines: acylates_a_free_diamine, own_reaction_chemistry, lead_family_verdict, bifunctional_ligand_in_the_pocket, independent_geometric_score

**Resolved**

- substrate distance **0.873** (own_curated_reaction)
- AlphaFold pLDDT **92.7**
- measured K_M **0.13 mM**
- curated ligand `CHEBI:29985`

**Native acceptor is a free di/polyamine** — the acceptor-side question for nylon-6,6

- `cadaverine` — C5 diamine - ONE methylene short of HMDA
- `putrescine` — C4 diamine
- `spermidine` — triamine
- `spermine` — tetraamine

**Catalytic activity** CATALYTIC ACTIVITY: Reaction=putrescine + L-glutamate + ATP = gamma-L-glutamylputrescine + ADP + phosphate + H(+); Xref=Rhea:RHEA:13633, ChEBI:CHEBI:15378, ChEBI:CHEBI:29985, ChEBI:CHEBI:30616, ChEBI:CHEBI:43474, ChEBI:CHEBI:58731, ChEBI:CHEBI:326268, ChEBI:CHEBI:456216; Evidence=; PhysiologicalDirection=left-to-right; Xref=Rhea:RHEA:13634; Evidence=; CATALYTIC ACTIVITY: Reaction=spermine + L-glut

**Function** FUNCTION: Involved in the catabolism of polyamines. Catalyzes the ATP-dependent gamma-glutamylation of polyamines. Substrates include putrescine, cadaverine, spermidine and spermine, with a preference for short-chain polyamine putrescine (PubMed:35409114). No complementation of the L-glutamine auxotrophy of an E.coli glnA mutant (PubMed:16932908). Together with GlnA3, enables survival of S.coelico

**Cofactor** COFACTOR: Name=Mg(2+); Xref=ChEBI:CHEBI:18420; Evidence=; Note=Binds 2 Mg(2+) ions per subunit.

**Family (UniProt)** SIMILARITY: Belongs to the glutamine synthetase family. .

## Q81RQ8 — Citryl-spermidine/3,4-dihydroxybenzoyl-citryl-spermidine:spermidine ligase (EC 6.3.2.-) (Petrobactin biosynthesis protei

**Organism** Bacillus anthracis  
**Gene** `asbB GBAA_1982`  
**Length** 612.0 aa  

**Why it is here** — 4 independent lines: acylates_a_free_diamine, own_reaction_chemistry, lead_family_verdict, adenylation_competent_structure, independent_geometric_score

**Resolved**

- substrate distance **0.789** (own_curated_reaction)
- AlphaFold pLDDT **96.1**
- structure `3TO3` at 2.382 Å
- in-line adenylation geometry 5.2 Å

**Native acceptor is a free di/polyamine** — the acceptor-side question for nylon-6,6

- `spermidine` — triamine

**Catalytic activity** CATALYTIC ACTIVITY: Reaction=N(8)-citryl-spermidine + spermidine + ATP = N(8),N'(8)-citryl-bis(spermidine) + AMP + diphosphate + H(+); Xref=Rhea:RHEA:63812, ChEBI:CHEBI:15378, ChEBI:CHEBI:30616, ChEBI:CHEBI:33019, ChEBI:CHEBI:57834, ChEBI:CHEBI:149586, ChEBI:CHEBI:149592, ChEBI:CHEBI:456215; Evidence=; CATALYTIC ACTIVITY: Reaction=N(1)-(3,4-dihydroxybenzoyl)-N(8)-citryl-spermidine + spermidine + A

**Function** FUNCTION: Involved in the biosynthesis of petrobactin, a catecholate siderophore that functions in both iron acquisition and virulence (PubMed:16875672, PubMed:17189355, PubMed:22408253). Catalyzes the ATP-dependent condensation of spermidine with N(8)-citryl-spermidine or N(1)-(3,4-dihydroxbenzoyl)-N(8)-citryl-spermidine, two intermediates in petrobactin biosynthesis pathway (PubMed:18758617, Pub

**Family (UniProt)** SIMILARITY: Belongs to the IucA/IucC family. .

## P9WJL3 — UDP-N-acetylmuramoyl-L-alanyl-D-glutamate--2,6-diaminopimelate ligase (EC 6.3.2.13) (Meso-A2pm-adding enzyme) (Meso-diam

**Organism** Mycobacterium tuberculosis (strain ATCC 25618 / H37Rv)  
**Gene** `murE Rv2158c MTCY270.10`  
**Length** 535.0 aa  

**Why it is here** — 4 independent lines: acylates_a_free_diamine, own_reaction_chemistry, lead_family_verdict, bifunctional_ligand_in_the_pocket, adenylation_competent_structure

**Resolved**

- substrate distance **0.764** (own_modelseed_reaction)
- its EC class reaches 0.764
- AlphaFold pLDDT **91.5**
- structure `2XJA` at 3.0 Å
- in-line adenylation geometry 6.2 Å
- crystallographic ligand `UAG`
- curated ligand `CHEBI:57791`

**Catalytic activity** CATALYTIC ACTIVITY: Reaction=UDP-N-acetyl-alpha-D-muramoyl-L-alanyl-D-glutamate + meso-2,6-diaminopimelate + ATP = UDP-N-acetyl-alpha-D-muramoyl-L-alanyl-gamma-D-glutamyl-meso-2,6-diaminopimelate + ADP + phosphate + H(+); Xref=Rhea:RHEA:23676, ChEBI:CHEBI:15378, ChEBI:CHEBI:30616, ChEBI:CHEBI:43474, ChEBI:CHEBI:57791, ChEBI:CHEBI:83900, ChEBI:CHEBI:83905, ChEBI:CHEBI:456216; EC=6.3.2.13; Evidence=

**Function** FUNCTION: Catalyzes the addition of meso-diaminopimelic acid to the nucleotide precursor UDP-N-acetylmuramoyl-L-alanyl-D-glutamate (UMAG) in the biosynthesis of bacterial cell-wall peptidoglycan. .

**Cofactor** COFACTOR: Name=Mg(2+); Xref=ChEBI:CHEBI:18420; Evidence=

**Location** SUBCELLULAR LOCATION: Cytoplasm .

**Family (UniProt)** SIMILARITY: Belongs to the MurCDEF family. MurE subfamily. .

## Q46E79 — 3-methyl-D-ornithine--L-lysine ligase (EC 6.3.2.59) ((3R)-3-methyl-D-ornithine:L-lysine ligase) (Pyrrolysine biosynthesi

**Organism** Methanosarcina barkeri (strain Fusaro / DSM 804)  
**Gene** `pylC Mbar_A0837`  
**Length** 363.0 aa  

**Why it is here** — 4 independent lines: acylates_a_free_diamine, lead_family_verdict, bifunctional_ligand_in_the_pocket, adenylation_competent_structure, independent_geometric_score

**Resolved**

- substrate distance **0.667** (own_modelseed_reaction)
- its EC class reaches 0.667
- AlphaFold pLDDT **94.8**
- structure `4FFL` at 1.5 Å
- in-line adenylation geometry 6.61 Å
- crystallographic ligand `0TK`
- curated ligand `CHEBI:32551;CHEBI:57668`

**Catalytic activity** CATALYTIC ACTIVITY: Reaction=(3R)-3-methyl-D-ornithine + L-lysine + ATP = (3R)-3-methyl-D-ornithyl-N(6)-L-lysine + ADP + phosphate + H(+); Xref=Rhea:RHEA:32763, ChEBI:CHEBI:15378, ChEBI:CHEBI:30616, ChEBI:CHEBI:32551, ChEBI:CHEBI:43474, ChEBI:CHEBI:64642, ChEBI:CHEBI:64643, ChEBI:CHEBI:456216; EC=6.3.2.59; Evidence=; PhysiologicalDirection=left-to-right; Xref=Rhea:RHEA:32764; Evidence=

**Function** FUNCTION: Is required for the biosynthesis of pyrrolysine (By similarity). Catalyzes the ATP-dependent ligation between (3R)-3-methyl-D-ornithine and L-lysine, leading to (3R)-3-methyl-D-ornithyl-N6-L-lysine (Probable). .

**Cofactor** COFACTOR: Name=Mg(2+); Xref=ChEBI:CHEBI:18420; Evidence=; Note=Binds 2 magnesium ions per subunit.

**Family (UniProt)** SIMILARITY: Belongs to the PylC family. .

## Q9KZC7 — Gamma-glutamylpolyamine synthetase GlnA3 (EC 6.3.1.-)

**Organism** Streptomyces coelicolor (strain ATCC BAA-471 / A3(2) / M145)  
**Gene** `glnA3 SCO6962`  
**Length** 466.0 aa  

**Why it is here** — 3 independent lines: acylates_a_free_diamine, own_reaction_chemistry, lead_family_verdict, bifunctional_ligand_in_the_pocket

**Resolved**

- substrate distance **0.873** (own_curated_reaction)
- AlphaFold pLDDT **96.0**
- curated ligand `CHEBI:29985`

**Native acceptor is a free di/polyamine** — the acceptor-side question for nylon-6,6

- `cadaverine` — C5 diamine - ONE methylene short of HMDA
- `putrescine` — C4 diamine
- `spermidine` — triamine
- `spermine` — tetraamine

**Catalytic activity** CATALYTIC ACTIVITY: Reaction=spermine + L-glutamate + ATP = gamma-L-glutamylspermine + ADP + phosphate + H(+); Xref=Rhea:RHEA:73895, ChEBI:CHEBI:15378, ChEBI:CHEBI:29985, ChEBI:CHEBI:30616, ChEBI:CHEBI:43474, ChEBI:CHEBI:45725, ChEBI:CHEBI:193052, ChEBI:CHEBI:456216; Evidence=; PhysiologicalDirection=left-to-right; Xref=Rhea:RHEA:73896; Evidence=; CATALYTIC ACTIVITY: Reaction=spermidine + L-glutam

**Function** FUNCTION: Involved in the catabolism of polyamines. Catalyzes the ATP-dependent gamma-glutamylation of polyamines. Substrates include putrescine, cadaverine, spermidine and spermine, with a preference for long-chain polyamines spermidine and spermine (PubMed:28487688, PubMed:35409114). Is not able to compensate for the loss of glutamine synthetases (GSs) (PubMed:28487688). No complementation of th

**Cofactor** COFACTOR: Name=Mg(2+); Xref=ChEBI:CHEBI:18420; Evidence=; Note=Binds 2 Mg(2+) ions per subunit.

**Family (UniProt)** SIMILARITY: Belongs to the glutamine synthetase family. .

## Q2FWC5 — L-aspartate--L-methionine ligase (LdmS) (EC 6.3.2.-) (ATP-dependent dipeptide ligase) (ATP-grasp enzyme) (L-Asp-L-Met li

**Organism** Staphylococcus aureus (strain NCTC 8325 / PS 47)  
**Gene** `SAOUHSC_02373`  
**Length** 397.0 aa  

**Why it is here** — 3 independent lines: own_reaction_chemistry, lead_family_verdict, bifunctional_ligand_in_the_pocket, adenylation_competent_structure

**Resolved**

- substrate distance **0.869** (own_modelseed_reaction)
- AlphaFold pLDDT **94.2**
- structure `7R8P` at 1.37 Å
- in-line adenylation geometry 4.0 Å
- measured K_M **0.31 mM**
- crystallographic ligand `CIT`

**Catalytic activity** CATALYTIC ACTIVITY: Reaction=L-aspartate + L-methionine + ATP = L-aspartyl-L-methionine + ADP + phosphate + H(+); Xref=Rhea:RHEA:76447, ChEBI:CHEBI:15378, ChEBI:CHEBI:29991, ChEBI:CHEBI:30616, ChEBI:CHEBI:43474, ChEBI:CHEBI:57844, ChEBI:CHEBI:195235, ChEBI:CHEBI:456216; Evidence=

**Function** FUNCTION: L-amino acid ligase, which preferentially catalyzes the formation of L-aspartyl-L-methionine dipeptide from L-aspartate and L-methionine in the presence of ATP. Less active with L-asparagine and L-methionine as substrates. Less active with L-aspartate and either L-phenylalanine, L-valine, L-leucine or L-isoleucine as substrates. Decreased activity when L-methionine is substituted with se

**Cofactor** COFACTOR: Name=Mg(2+); Xref=ChEBI:CHEBI:18420; Evidence=

## I3R584 — Gamma-glutamylputrescine synthetase (Gamma-Glu-Put synthetase) (EC 6.3.1.11) (Glutamate--putrescine ligase)

**Organism** Haloferax mediterranei (strain ATCC 33500 / DSM 1411 / JCM 8866 / NBRC 14739 / NCIMB 2177   
**Gene** `glnA2 HFX_1688 BM92_00670 C439_02042 E6P`  
**Length** 446.0 aa  

**Why it is here** — 3 independent lines: acylates_a_free_diamine, own_reaction_chemistry, lead_family_verdict, bifunctional_ligand_in_the_pocket

**Resolved**

- substrate distance **0.857** (own_modelseed_reaction)
- its EC class reaches 0.857
- AlphaFold pLDDT **94.6**
- curated ligand `CHEBI:29985`

**Native acceptor is a free di/polyamine** — the acceptor-side question for nylon-6,6

- `putrescine` — C4 diamine

**Catalytic activity** CATALYTIC ACTIVITY: Reaction=putrescine + L-glutamate + ATP = gamma-L-glutamylputrescine + ADP + phosphate + H(+); Xref=Rhea:RHEA:13633, ChEBI:CHEBI:15378, ChEBI:CHEBI:29985, ChEBI:CHEBI:30616, ChEBI:CHEBI:43474, ChEBI:CHEBI:58731, ChEBI:CHEBI:326268, ChEBI:CHEBI:456216; EC=6.3.1.11; Evidence=; PhysiologicalDirection=left-to-right; Xref=Rhea:RHEA:13634; Evidence=

**Function** FUNCTION: Involved in the breakdown of putrescine via the biosynthesis of gamma-L-glutamylputrescine. Not required for glutamine synthesis (PubMed:34439822). Is not able to compensate for the loss of glutamine synthetase (PubMed:32296946). .

**Cofactor** COFACTOR: Name=Mg(2+); Xref=ChEBI:CHEBI:18420; Evidence=; Note=Binds 2 Mg(2+) ions per subunit.

**Family (UniProt)** SIMILARITY: Belongs to the glutamine synthetase family. .

**BRENDA** `6.3.1.11;` — an entry exists; the API needs credentials this project does not have

**Literature** [22843593](https://pubmed.ncbi.nlm.nih.gov/22843593/), [25393412](https://pubmed.ncbi.nlm.nih.gov/25393412/), [32296946](https://pubmed.ncbi.nlm.nih.gov/32296946/), [34439822](https://pubmed.ncbi.nlm.nih.gov/34439822/)

## P78061 — Gamma-glutamylputrescine synthetase PuuA (Gamma-Glu-Put synthetase) (EC 6.3.1.11) (Glutamate--putrescine ligase)

**Organism** Escherichia coli (strain K12)  
**Gene** `puuA ycjK b1297 JW5201`  
**Length** 472.0 aa  

**Why it is here** — 3 independent lines: acylates_a_free_diamine, own_reaction_chemistry, lead_family_verdict, independent_geometric_score

**Resolved**

- substrate distance **0.857** (own_modelseed_reaction)
- its EC class reaches 0.857
- AlphaFold pLDDT **92.7**
- measured K_M **2.07 mM**

**Native acceptor is a free di/polyamine** — the acceptor-side question for nylon-6,6

- `putrescine` — C4 diamine
- `spermidine` — triamine
- `spermine` — tetraamine

**Catalytic activity** CATALYTIC ACTIVITY: Reaction=putrescine + L-glutamate + ATP = gamma-L-glutamylputrescine + ADP + phosphate + H(+); Xref=Rhea:RHEA:13633, ChEBI:CHEBI:15378, ChEBI:CHEBI:29985, ChEBI:CHEBI:30616, ChEBI:CHEBI:43474, ChEBI:CHEBI:58731, ChEBI:CHEBI:326268, ChEBI:CHEBI:456216; EC=6.3.1.11; Evidence=; PhysiologicalDirection=left-to-right; Xref=Rhea:RHEA:13634; Evidence=

**Function** FUNCTION: Involved in the breakdown of putrescine. Catalyzes the ATP-dependent gamma-glutamylation of putrescine, producing gamma-L-glutamylputrescine. Absolutely essential to utilize putrescine as both nitrogen and carbon sources and to decrease the toxicity of putrescine, which can lead to inhibition of cell growth and protein synthesis. In vitro is also able to use several diamines, and spermid

**Cofactor** COFACTOR: Name=Mg(2+); Xref=ChEBI:CHEBI:18420; Evidence=; Name=Mn(2+); Xref=ChEBI:CHEBI:29035; Evidence=; Note=Requires Mg(2+), that can be substituted by Mn(2+).

**Family (UniProt)** SIMILARITY: Belongs to the glutamine synthetase family. .

## E2JA32 — Fumarate--(S)-2,3-diaminopropanoate ligase (EC 6.3.2.46) (ATP-dependent fumaroyl-DAP ligase)

**Organism** Enterobacter agglomerans (Erwinia herbicola) (Pantoea agglomerans)  
**Gene** `ddaG`  
**Length** 396.0 aa  

**Why it is here** — 3 independent lines: acylates_a_free_diamine, own_reaction_chemistry, lead_family_verdict, independent_geometric_score

**Resolved**

- substrate distance **0.848** (own_modelseed_reaction)
- its EC class reaches 0.848
- AlphaFold pLDDT **88.3**
- measured K_M **0.55 mM**

**Catalytic activity** CATALYTIC ACTIVITY: Reaction=(S)-2,3-diaminopropanoate + fumarate + ATP = N(3)-fumaroyl-(S)-2,3-diaminopropanoate + AMP + diphosphate; Xref=Rhea:RHEA:44340, ChEBI:CHEBI:29806, ChEBI:CHEBI:30616, ChEBI:CHEBI:33019, ChEBI:CHEBI:57721, ChEBI:CHEBI:84330, ChEBI:CHEBI:456215; EC=6.3.2.46; Evidence=

**Function** FUNCTION: Involved in dapdiamide antibiotics biosynthesis. Ligates fumarate and 2,3-diaminopropionate (DAP) to form N-beta-fumaroyl-DAP. Can also form N-succinoyl-DAP from succinate and DAP, with lower efficiency. .

**BRENDA** `6.3.2.46;` — an entry exists; the API needs credentials this project does not have

**Literature** [20945916](https://pubmed.ncbi.nlm.nih.gov/20945916/), [19807062](https://pubmed.ncbi.nlm.nih.gov/19807062/)

## Q2FW70 — D-ornithine--citrate ligase (EC 6.3.2.58)

**Organism** Staphylococcus aureus (strain NCTC 8325 / PS 47)  
**Gene** `sfnaD SAOUHSC_02436`  
**Length** 658.0 aa  

**Why it is here** — 3 independent lines: acylates_a_free_diamine, own_reaction_chemistry, lead_family_verdict, independent_geometric_score

**Resolved**

- substrate distance **0.809** (own_modelseed_reaction)
- its EC class reaches 0.809
- AlphaFold pLDDT **94.4**

**Catalytic activity** CATALYTIC ACTIVITY: Reaction=D-ornithine + citrate + ATP = N(5)-[(S)-citryl]-D-ornithine + AMP + diphosphate + H(+); Xref=Rhea:RHEA:59136, ChEBI:CHEBI:15378, ChEBI:CHEBI:16947, ChEBI:CHEBI:30616, ChEBI:CHEBI:33019, ChEBI:CHEBI:57668, ChEBI:CHEBI:142972, ChEBI:CHEBI:456215; EC=6.3.2.58; Evidence=; PhysiologicalDirection=left-to-right; Xref=Rhea:RHEA:59137; Evidence=

**Function** FUNCTION: Involved in the biosynthesis of the siderophore staphyloferrin A. Catalyzes the ATP-dependent condensation of D-ornithine and citrate to form a citryl-D-ornithine intermediate. .

**Family (UniProt)** SIMILARITY: Belongs to the IucA/IucC family. .

## Q2G1M8 — 2-[(L-alanin-3-ylcarbamoyl)methyl]-3-(2-aminoethylcarbamoyl)-2-hydroxypropanoate synthase (EC 6.3.2.55) (Staphyloferrin 

**Organism** Staphylococcus aureus (strain NCTC 8325 / PS 47)  
**Gene** `sbnF SAOUHSC_00080`  
**Length** 616.0 aa  

**Why it is here** — 3 independent lines: acylates_a_free_diamine, own_reaction_chemistry, lead_family_verdict, independent_geometric_score

**Resolved**

- substrate distance **0.807** (own_curated_reaction)
- its EC class reaches 0.761
- AlphaFold pLDDT **91.1**

**Catalytic activity** CATALYTIC ACTIVITY: Reaction=2-[(2-aminoethylcarbamoyl)methyl]-2-hydroxybutanedioate + (S)-2,3-diaminopropanoate + ATP = 2-[(L-alanin-3-ylcarbamoyl)methyl]-3-(2-aminoethylcarbamoyl)-2-hydroxypropanoate + AMP + diphosphate; Xref=Rhea:RHEA:59128, ChEBI:CHEBI:30616, ChEBI:CHEBI:33019, ChEBI:CHEBI:57721, ChEBI:CHEBI:142970, ChEBI:CHEBI:142971, ChEBI:CHEBI:456215; EC=6.3.2.55; Evidence=; PhysiologicalD

**Function** FUNCTION: Catalyzes the condensation of L-2,3-diaminopropionic acid (L-Dap) and citryl-diaminoethane to form L-2,3-diaminopropionyl-citryl-diaminoethane, the third step in staphyloferrin B biosynthesis. .

**Family (UniProt)** SIMILARITY: Belongs to the IucA/IucC family. .

## Q2G1M9 — L-2,3-diaminopropanoate--citrate ligase (EC 6.3.2.54) (Staphyloferrin B biosynthesis protein SbnE)

**Organism** Staphylococcus aureus (strain NCTC 8325 / PS 47)  
**Gene** `sbnE SAOUHSC_00079`  
**Length** 578.0 aa  

**Why it is here** — 3 independent lines: acylates_a_free_diamine, own_reaction_chemistry, lead_family_verdict, independent_geometric_score

**Resolved**

- substrate distance **0.807** (own_modelseed_reaction)
- its EC class reaches 0.807
- AlphaFold pLDDT **92.6**
- measured K_M **0.99 mM**

**Catalytic activity** CATALYTIC ACTIVITY: Reaction=(S)-2,3-diaminopropanoate + citrate + ATP = 2-[(L-alanin-3-ylcarbamoyl)methyl]-2-hydroxybutanedioate + AMP + diphosphate; Xref=Rhea:RHEA:59124, ChEBI:CHEBI:16947, ChEBI:CHEBI:30616, ChEBI:CHEBI:33019, ChEBI:CHEBI:57721, ChEBI:CHEBI:142969, ChEBI:CHEBI:456215; EC=6.3.2.54; Evidence=; PhysiologicalDirection=left-to-right; Xref=Rhea:RHEA:59125; Evidence=

**Function** FUNCTION: Catalyzes the synthesis of citryl-L-2,3-diaminopropionic acid from L-2,3-diaminopropionic acid (L-Dap) and citrate, the first step in staphyloferrin B biosynthesis. .

**Family (UniProt)** SIMILARITY: Belongs to the IucA/IucC family. .

## I6Y0R5 — Dihydrofolate synthase/folylpolyglutamate synthase (DHFS / FPGS) (EC 6.3.2.12) (EC 6.3.2.17) (Folylpoly-gamma-glutamate 

**Organism** Mycobacterium tuberculosis (strain ATCC 25618 / H37Rv)  
**Gene** `folC Rv2447c`  
**Length** 487.0 aa  

**Why it is here** — 3 independent lines: own_reaction_chemistry, lead_family_verdict, bifunctional_ligand_in_the_pocket, adenylation_competent_structure

**Resolved**

- substrate distance **0.803** (own_modelseed_reaction)
- its EC class reaches 0.382
- AlphaFold pLDDT **92.5**
- structure `2VOS` at 2.0 Å
- in-line adenylation geometry 6.12 Å
- curated ligand `CHEBI:17839`

**Catalytic activity** CATALYTIC ACTIVITY: Reaction=7,8-dihydropteroate + L-glutamate + ATP = 7,8-dihydrofolate + ADP + phosphate + H(+); Xref=Rhea:RHEA:23584, ChEBI:CHEBI:15378, ChEBI:CHEBI:17839, ChEBI:CHEBI:29985, ChEBI:CHEBI:30616, ChEBI:CHEBI:43474, ChEBI:CHEBI:57451, ChEBI:CHEBI:456216; EC=6.3.2.12; Evidence=; CATALYTIC ACTIVITY: Reaction=(6S)-5,6,7,8-tetrahydrofolyl-(gamma-L-Glu)(n) + L-glutamate + ATP = (6S)-5,6

**Function** FUNCTION: Catalyzes the addition of a glutamate residue to dihydropteroate (7,8-dihydropteroate or H2Pte) to form dihydrofolate (7,8-dihydrofolate monoglutamate or H2Pte-Glu) (PubMed:23118010). Also catalyzes successive additions of L-glutamate to tetrahydrofolate, leading to folylpolyglutamate derivatives (By similarity). .; FUNCTION: Is involved in the bioactivation of the antituberculous drug p

**Cofactor** COFACTOR: Name=Mg(2+); Xref=ChEBI:CHEBI:18420; Evidence=; Note=Binds 2 Mg(2+) ions per subunit.

**Family (UniProt)** SIMILARITY: Belongs to the folylpolyglutamate synthase family. .

**Note** MISCELLANEOUS: Mutations within this gene in the H2Pte binding pocket are responsible for PAS resistance in M.tuberculosis clinical isolates and laboratory strains. FolC-linked PAS resistance is mediated by altered substrate specificity that results in the failure to generate levels of H2PtePAS-Glu that are necessary to inhibit DHFR (DfrA); therefore, a blockage of PAS bioactivation causes PAS res

**BRENDA** `6.3.2.12;` — an entry exists; the API needs credentials this project does not have

**Literature** [9634230](https://pubmed.ncbi.nlm.nih.gov/9634230/), [12657046](https://pubmed.ncbi.nlm.nih.gov/12657046/), [16754987](https://pubmed.ncbi.nlm.nih.gov/16754987/), [21969609](https://pubmed.ncbi.nlm.nih.gov/21969609/), [23779105](https://pubmed.ncbi.nlm.nih.gov/23779105/), [23118010](https://pubmed.ncbi.nlm.nih.gov/23118010/), [24366731](https://pubmed.ncbi.nlm.nih.gov/24366731/), [18566510](https://pubmed.ncbi.nlm.nih.gov/18566510/)

## P08192 — Dihydrofolate synthase/folylpolyglutamate synthase (DHFS / FPGS) (EC 6.3.2.12) (EC 6.3.2.17) (Folylpoly-gamma-glutamate 

**Organism** Escherichia coli (strain K12)  
**Gene** `folC dedC b2315 JW2312`  
**Length** 422.0 aa  

**Why it is here** — 3 independent lines: own_reaction_chemistry, lead_family_verdict, bifunctional_ligand_in_the_pocket, adenylation_competent_structure

**Resolved**

- substrate distance **0.803** (own_modelseed_reaction)
- its EC class reaches 0.382
- AlphaFold pLDDT **95.7**
- structure `1W78` at 1.82 Å
- in-line adenylation geometry 6.13 Å
- measured K_M **0.0006 mM**
- curated ligand `CHEBI:17839`

**Catalytic activity** CATALYTIC ACTIVITY: Reaction=7,8-dihydropteroate + L-glutamate + ATP = 7,8-dihydrofolate + ADP + phosphate + H(+); Xref=Rhea:RHEA:23584, ChEBI:CHEBI:15378, ChEBI:CHEBI:17839, ChEBI:CHEBI:29985, ChEBI:CHEBI:30616, ChEBI:CHEBI:43474, ChEBI:CHEBI:57451, ChEBI:CHEBI:456216; EC=6.3.2.12; Evidence=; CATALYTIC ACTIVITY: Reaction=(6S)-5,6,7,8-tetrahydrofolyl-(gamma-L-Glu)(n) + L-glutamate + ATP = (6S)-5,6

**Function** FUNCTION: Functions in two distinct reactions of the de novo folate biosynthetic pathway. Catalyzes the addition of a glutamate residue to dihydropteroate (7,8-dihydropteroate or H2Pte) to form dihydrofolate (7,8-dihydrofolate monoglutamate or H2Pte-Glu). Also catalyzes successive additions of L-glutamate to tetrahydrofolate or 10-formyltetrahydrofolate or 5,10-methylenetetrahydrofolate, leading t

**Cofactor** COFACTOR: Name=Mg(2+); Xref=ChEBI:CHEBI:18420; Evidence=; Note=Binds 2 Mg(2+) ions per subunit.

**Family (UniProt)** SIMILARITY: Belongs to the folylpolyglutamate synthase family. .

## A0A0F7RJ52 — Spermidine-citrate ligase (EC 6.3.2.-) (Petrobactin biosynthesis protein AsbA)

**Organism** Bacillus anthracis  
**Gene** `asbA GBAA_1981`  
**Length** 602.0 aa  

**Why it is here** — 3 independent lines: acylates_a_free_diamine, own_reaction_chemistry, lead_family_verdict, independent_geometric_score

**Resolved**

- substrate distance **0.789** (own_modelseed_reaction)
- AlphaFold pLDDT **93.4**
- measured K_M **5.65 mM**

**Native acceptor is a free di/polyamine** — the acceptor-side question for nylon-6,6

- `spermidine` — triamine

**Catalytic activity** CATALYTIC ACTIVITY: Reaction=spermidine + citrate + ATP = N(8)-citryl-spermidine + AMP + diphosphate + H(+); Xref=Rhea:RHEA:63808, ChEBI:CHEBI:15378, ChEBI:CHEBI:16947, ChEBI:CHEBI:30616, ChEBI:CHEBI:33019, ChEBI:CHEBI:57834, ChEBI:CHEBI:149586, ChEBI:CHEBI:456215; Evidence=

**Function** FUNCTION: Involved in the biosynthesis of petrobactin, a catecholate siderophore that functions in both iron acquisition and virulence (PubMed:16875672, PubMed:17189355, PubMed:22408253). Catalyzes the ATP-dependent condensation of citric acid and spermidine to form N(8)-citryl-spermidine (PubMed:17579415, PubMed:19259597, PubMed:22408253). It can also catalyze the condensation of several di- and 

**Family (UniProt)** SIMILARITY: Belongs to the IucA/IucC family. .

**Biotechnology** BIOTECHNOLOGY: The ability of the enzyme to catalyze condensation of citric acid with several di- and triamine analogs of spermidine suggests that it may be a versatile biocatalyst for the preparation of highly enantiomerically-enriched citrate derivatives via a novel type of desymmetrization process. .

**Literature** [18952800](https://pubmed.ncbi.nlm.nih.gov/18952800/), [14756782](https://pubmed.ncbi.nlm.nih.gov/14756782/), [16875672](https://pubmed.ncbi.nlm.nih.gov/16875672/), [17189355](https://pubmed.ncbi.nlm.nih.gov/17189355/), [17579415](https://pubmed.ncbi.nlm.nih.gov/17579415/), [19259597](https://pubmed.ncbi.nlm.nih.gov/19259597/), [22408253](https://pubmed.ncbi.nlm.nih.gov/22408253/), [27425635](https://pubmed.ncbi.nlm.nih.gov/27425635/)

## Q2FZP6 — UDP-N-acetylmuramoyl-L-alanyl-D-glutamate--L-lysine ligase (EC 6.3.2.7) (L-lysine-adding enzyme) (UDP-MurNAc-L-Ala-D-Glu

**Organism** Staphylococcus aureus (strain NCTC 8325 / PS 47)  
**Gene** `murE SAOUHSC_00954`  
**Length** 493.0 aa  

**Why it is here** — 3 independent lines: acylates_a_free_diamine, own_reaction_chemistry, lead_family_verdict, adenylation_competent_structure

**Resolved**

- substrate distance **0.784** (own_modelseed_reaction)
- its EC class reaches 0.784
- AlphaFold pLDDT **96.6**
- structure `4C12` at 1.8 Å
- in-line adenylation geometry 6.51 Å
- measured K_M **0.087 mM**

**Catalytic activity** CATALYTIC ACTIVITY: Reaction=UDP-N-acetyl-alpha-D-muramoyl-L-alanyl-D-glutamate + L-lysine + ATP = UDP-N-acetyl-alpha-D-muramoyl-L-alanyl-gamma-D-glutamyl-L-lysine + ADP + phosphate + H(+); Xref=Rhea:RHEA:17969, ChEBI:CHEBI:15378, ChEBI:CHEBI:30616, ChEBI:CHEBI:32551, ChEBI:CHEBI:43474, ChEBI:CHEBI:83900, ChEBI:CHEBI:83903, ChEBI:CHEBI:456216; EC=6.3.2.7; Evidence=

**Function** FUNCTION: Catalyzes the addition of L-lysine to the nucleotide precursor UDP-N-acetylmuramoyl-L-alanyl-D-glutamate (UMAG) in the biosynthesis of bacterial cell-wall peptidoglycan. Cannot use diaminopimelate as substrate (PubMed:10498701, PubMed:14114846, PubMed:20659527). Can accept L-ornithine as substrate, but the efficiency is 400-fold lower than that with L-lysine (PubMed:20659527). Seems to h

**Cofactor** COFACTOR: Name=Mg(2+); Xref=ChEBI:CHEBI:18420; Evidence=

**Location** SUBCELLULAR LOCATION: Cytoplasm .

**Family (UniProt)** SIMILARITY: Belongs to the MurCDEF family. MurE subfamily. .

## Q9WY79 — UDP-N-acetylmuramoyl-L-alanyl-D-glutamate--LD-lysine ligase (EC 6.3.2.37) (EC 6.3.2.7) (D-lysine-adding enzyme) (L-lysin

**Organism** Thermotoga maritima (strain ATCC 43589 / DSM 3109 / JCM 10099 / NBRC 100826 / MSB8)  
**Gene** `murE TM_0237`  
**Length** 490.0 aa  

**Why it is here** — 3 independent lines: acylates_a_free_diamine, own_reaction_chemistry, lead_family_verdict, adenylation_competent_structure

**Resolved**

- substrate distance **0.784** (own_modelseed_reaction)
- its EC class reaches 0.784
- AlphaFold pLDDT **94.5**
- structure `4BUB` at 2.9 Å
- in-line adenylation geometry 6.37 Å
- measured K_M **0.45 mM**

**Catalytic activity** CATALYTIC ACTIVITY: Reaction=UDP-N-acetyl-alpha-D-muramoyl-L-alanyl-D-glutamate + L-lysine + ATP = UDP-N-acetyl-alpha-D-muramoyl-L-alanyl-gamma-D-glutamyl-L-lysine + ADP + phosphate + H(+); Xref=Rhea:RHEA:17969, ChEBI:CHEBI:15378, ChEBI:CHEBI:30616, ChEBI:CHEBI:32551, ChEBI:CHEBI:43474, ChEBI:CHEBI:83900, ChEBI:CHEBI:83903, ChEBI:CHEBI:456216; EC=6.3.2.7; Evidence=; CATALYTIC ACTIVITY: Reaction=UD

**Function** FUNCTION: Catalyzes the addition of both L- and D-lysine to the nucleotide precursor UDP-N-acetylmuramoyl-L-alanyl-D-glutamate (UMAG) in the biosynthesis of bacterial cell-wall peptidoglycan. Is also able to use meso-diaminopimelate as the amino acid substrate in vitro, although much less efficiently. .

**Location** SUBCELLULAR LOCATION: Cytoplasm .

**Family (UniProt)** SIMILARITY: Belongs to the MurCDEF family. MurE subfamily. .

## P22188 — UDP-N-acetylmuramoyl-L-alanyl-D-glutamate--2,6-diaminopimelate ligase (EC 6.3.2.13) (Meso-A2pm-adding enzyme) (Meso-diam

**Organism** Escherichia coli (strain K12)  
**Gene** `murE b0085 JW0083`  
**Length** 495.0 aa  

**Why it is here** — 3 independent lines: acylates_a_free_diamine, own_reaction_chemistry, lead_family_verdict, bifunctional_ligand_in_the_pocket

**Resolved**

- substrate distance **0.764** (own_modelseed_reaction)
- its EC class reaches 0.764
- AlphaFold pLDDT **96.2**
- structure `7B61` at 1.65 Å
- measured K_M **0.036 mM**
- crystallographic ligand `API;UAG`
- curated ligand `CHEBI:57791`

**Catalytic activity** CATALYTIC ACTIVITY: Reaction=UDP-N-acetyl-alpha-D-muramoyl-L-alanyl-D-glutamate + meso-2,6-diaminopimelate + ATP = UDP-N-acetyl-alpha-D-muramoyl-L-alanyl-gamma-D-glutamyl-meso-2,6-diaminopimelate + ADP + phosphate + H(+); Xref=Rhea:RHEA:23676, ChEBI:CHEBI:15378, ChEBI:CHEBI:30616, ChEBI:CHEBI:43474, ChEBI:CHEBI:57791, ChEBI:CHEBI:83900, ChEBI:CHEBI:83905, ChEBI:CHEBI:456216; EC=6.3.2.13; Evidence=

**Function** FUNCTION: Catalyzes the addition of meso-diaminopimelic acid to the nucleotide precursor UDP-N-acetylmuramoyl-L-alanyl-D-glutamate (UMAG) in the biosynthesis of bacterial cell-wall peptidoglycan. Is also able to use many meso-diaminopimelate analogs as substrates, although much less efficiently, but not L-lysine. .

**Cofactor** COFACTOR: Name=Mg(2+); Xref=ChEBI:CHEBI:18420; Evidence=

**Location** SUBCELLULAR LOCATION: Cytoplasm .

**Family (UniProt)** SIMILARITY: Belongs to the MurCDEF family. MurE subfamily. .

## Q819Q0 — UDP-N-acetylmuramoyl-L-alanyl-D-glutamate--2,6-diaminopimelate ligase (EC 6.3.2.13) (Meso-A2pm-adding enzyme) (Meso-diam

**Organism** Bacillus cereus (strain ATCC 14579 / DSM 31 / CCUG 7414 / JCM 2152 / NBRC 15305 / NCIMB 93  
**Gene** `murE BC_3914`  
**Length** 491.0 aa  

**Why it is here** — 3 independent lines: acylates_a_free_diamine, own_reaction_chemistry, lead_family_verdict, bifunctional_ligand_in_the_pocket

**Resolved**

- substrate distance **0.764** (own_modelseed_reaction)
- its EC class reaches 0.764
- AlphaFold pLDDT **95.6**
- measured K_M **0.13 mM**
- curated ligand `CHEBI:57791`

**Catalytic activity** CATALYTIC ACTIVITY: Reaction=UDP-N-acetyl-alpha-D-muramoyl-L-alanyl-D-glutamate + meso-2,6-diaminopimelate + ATP = UDP-N-acetyl-alpha-D-muramoyl-L-alanyl-gamma-D-glutamyl-meso-2,6-diaminopimelate + ADP + phosphate + H(+); Xref=Rhea:RHEA:23676, ChEBI:CHEBI:15378, ChEBI:CHEBI:30616, ChEBI:CHEBI:43474, ChEBI:CHEBI:57791, ChEBI:CHEBI:83900, ChEBI:CHEBI:83905, ChEBI:CHEBI:456216; EC=6.3.2.13; Evidence=

**Function** FUNCTION: Catalyzes the addition of meso-diaminopimelic acid to the nucleotide precursor UDP-N-acetylmuramoyl-L-alanyl-D-glutamate (UMAG) in the biosynthesis of bacterial cell-wall peptidoglycan. .

**Cofactor** COFACTOR: Name=Mg(2+); Xref=ChEBI:CHEBI:18420; Evidence=

**Location** SUBCELLULAR LOCATION: Cytoplasm .

**Family (UniProt)** SIMILARITY: Belongs to the MurCDEF family. MurE subfamily. .

## A0QF47 — UDP-N-acetylmuramoyl-L-alanyl-D-glutamate--2,6-diaminopimelate ligase (EC 6.3.2.13) (Meso-A2pm-adding enzyme) (Meso-diam

**Organism** Mycobacterium avium (strain 104)  
**Gene** `murE MAV_2331`  
**Length** 520.0 aa  

**Why it is here** — 3 independent lines: acylates_a_free_diamine, own_reaction_chemistry, lead_family_verdict, bifunctional_ligand_in_the_pocket

**Resolved**

- substrate distance **0.764** (own_modelseed_reaction)
- its EC class reaches 0.764
- curated ligand `CHEBI:57791`

**Catalytic activity** CATALYTIC ACTIVITY: Reaction=UDP-N-acetyl-alpha-D-muramoyl-L-alanyl-D-glutamate + meso-2,6-diaminopimelate + ATP = UDP-N-acetyl-alpha-D-muramoyl-L-alanyl-gamma-D-glutamyl-meso-2,6-diaminopimelate + ADP + phosphate + H(+); Xref=Rhea:RHEA:23676, ChEBI:CHEBI:15378, ChEBI:CHEBI:30616, ChEBI:CHEBI:43474, ChEBI:CHEBI:57791, ChEBI:CHEBI:83900, ChEBI:CHEBI:83905, ChEBI:CHEBI:456216; EC=6.3.2.13; Evidence=

**Function** FUNCTION: Catalyzes the addition of meso-diaminopimelic acid to the nucleotide precursor UDP-N-acetylmuramoyl-L-alanyl-D-glutamate (UMAG) in the biosynthesis of bacterial cell-wall peptidoglycan. .

**Cofactor** COFACTOR: Name=Mg(2+); Xref=ChEBI:CHEBI:18420; Evidence=

**Location** SUBCELLULAR LOCATION: Cytoplasm .

**Family (UniProt)** SIMILARITY: Belongs to the MurCDEF family. MurE subfamily. .

